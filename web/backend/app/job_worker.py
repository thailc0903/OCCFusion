"""Separate queue worker. Model execution only occurs in the pinned rav25 CLI."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .config import REPO
from .database import utc_now
from .evaluations import digest
from .job_results import JobResults, file_hash, file_records, verify_publication
from .jobs import Jobs, TERMINAL
from .review import canonical
from .store import contained, InvalidArtifact
from .task_types import task_of,DETECTION


def process_ticks(pid):
    try:
        value=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if value[0]=='Z' else value[19]
    except (OSError,IndexError):return None


@contextmanager
def gpu_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield None;return
        try:yield stream.fileno()
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


class Commands:
    mode='real'

    def inference(self,job,paths):
        p=json.loads(job['profile_json'])
        cli='tools/infer_centerpoint_mini.py' if task_of(p)==DETECTION else 'tools/infer_nuscenes_mini.py'
        return [p['python'],'-u',str(REPO/cli),
            '--config',p['config'],'--checkpoint',p['checkpoint'],'--data-root',p['data_root'],
            '--scene-name',job['scene_name'],'--frame-index',str(job['frame_index']),
            '--device',p['device'],'--out-dir',str(paths['prediction'])]

    def evaluation(self,job,paths):
        p=json.loads(job['profile_json'])
        if task_of(p)==DETECTION:return None
        return [p['python'],'-u',str(REPO/'tools/evaluate_occupancy.py'),
            '--run-dir',str(paths['prediction']),'--gt-root',p['gt_root'],'--data-root',p['data_root'],
            '--split',job['split'],'--verify-reference','--out',str(paths['evaluation']/'evaluation.json')]


class Worker:
    def __init__(self,database,settings,commands=None,poll_s=.25):
        self.db,self.settings=database,settings;self.jobs=Jobs(database,settings)
        self.commands=commands or Commands();self.results=JobResults(settings)
        self.id=uuid.uuid4().hex;self.stop_requested=False;self.poll_s=poll_s;self.last_beat=0
        with self.db.transaction() as c:
            c.execute('INSERT INTO inference_workers VALUES(?,?,?,?,?,?,?)',
                (self.id,os.getpid(),process_ticks(os.getpid()) or '',utc_now(),time.time(),'active',self.commands.mode))

    def beat(self,force=False):
        if force or time.monotonic()-self.last_beat>2:
            with self.db.connect() as c:c.execute('UPDATE inference_workers SET heartbeat_at=?,status=? WHERE id=?',(time.time(),'active',self.id))
            self.last_beat=time.monotonic()

    def phase(self,identifier,phase,message):
        with self.db.transaction() as c:
            c.execute('UPDATE inference_jobs SET phase=? WHERE id=?',(phase,identifier))
            self.jobs.event(c,identifier,phase,message)
        self.beat()

    def terminal(self,identifier,status,error=None,result=None,evaluation_status=None):
        with self.db.transaction() as c:
            c.execute('''UPDATE inference_jobs SET status=?,phase=?,finished_at=?,error_json=?,result_json=?,
                evaluation_status=?,child_pid=NULL,child_started_ticks=NULL WHERE id=?''',
                (status,status,utc_now(),canonical(error) if error else None,canonical(result) if result else None,evaluation_status,identifier))
            self.jobs.event(c,identifier,status,error['message'] if error else 'Validated job completed; review remains unregistered.')

    def recover(self):
        """Called only with the OS lane lock. Never relaunch an interrupted CLI."""
        with self.db.connect() as c:
            running=c.execute("SELECT * FROM inference_jobs WHERE status='running'").fetchall()
        for job in running:
            with self.db.connect() as c:owner=c.execute('SELECT * FROM inference_workers WHERE id=?',(job['worker_id'],)).fetchone()
            if owner and process_ticks(owner['pid'])==owner['started_ticks']:
                return False
            if job['child_pid'] and process_ticks(job['child_pid'])==job['child_started_ticks']:
                return False
            paths=self.jobs.paths(job['id'])
            if paths['run'].is_dir() and not job['cancel_requested']:
                try:
                    marker=verify_publication(paths['run'],job['id'])
                    if marker['profile_fingerprint']!=json.loads(job['profile_json'])['fingerprint']:
                        raise InvalidArtifact('Recovered publication profile mismatch')
                    meta,report,files=self.results.prediction(paths['run'],job)
                    if task_of(json.loads(job['profile_json']))==DETECTION:
                        status='pending_dataset_evaluation'
                    else:
                        status,evaluation=self.results.evaluation(paths['eval_run']/'evaluation.json',job,meta,files)
                        if status!=marker['evaluation']['status'] or marker['result']['metrics']!=evaluation.get('metrics'):
                            raise InvalidArtifact('Recovered result/evaluation mismatch')
                    with self.db.transaction() as c:
                        self.jobs.event(c,job['id'],'recovered','Recovered an already complete, hash-verified publication without rerunning inference.',actor_type='recovery')
                    self.terminal(job['id'],'succeeded',result=marker['result'],evaluation_status=status)
                    continue
                except (OSError,ValueError,KeyError,TypeError) as error:
                    message='Published output could not be recovered: '+str(error)
            else:message='Worker stopped before a complete publication. Partial output retained; inference is not automatically retried.'
            self.terminal(job['id'],'canceled' if job['cancel_requested'] else 'failed',
                error=dict(stage='worker_interrupted',type='WorkerInterrupted',message=message))
        return True

    def claim(self):
        with self.db.transaction() as c:
            if c.execute("SELECT 1 FROM inference_jobs WHERE status='running'").fetchone():return None
            row=c.execute("SELECT * FROM inference_jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if not row:return None
            c.execute("UPDATE inference_jobs SET status='running',phase='claimed',started_at=?,worker_id=?,execution_mode=? WHERE id=?",
                (utc_now(),self.id,self.commands.mode,row['id']))
            self.jobs.event(c,row['id'],'claimed','Worker claimed job under the exclusive GPU lane lock.')
            return self.jobs.row(row['id'],c)

    def preflight(self,job):
        p=json.loads(job['profile_json'])
        if digest(canonical({k:v for k,v in p.items() if k!='fingerprint'}).encode())!=p['fingerprint']:
            raise InvalidArtifact('Stored job profile fingerprint mismatch')
        if task_of(p)!=DETECTION and not any(self.jobs.paths(job['id'])['eval_run'].is_relative_to(root.resolve()) for root in self.settings.evaluation_directories):
            raise InvalidArtifact('Job evaluation output is outside configured evaluation roots')
        if not Path(p['python']).is_file() or not Path(p['checkpoint']).is_file():raise FileNotFoundError('Profile Python/checkpoint is missing')
        if Path(p['checkpoint']).stat().st_size!=p['checkpoint_bytes']:raise InvalidArtifact('Checkpoint file size differs from pinned profile')
        for name,expected in p['source_hashes'].items():
            if file_hash(contained(REPO/name,REPO))!=expected:raise InvalidArtifact('Pinned model/config/CLI source changed: '+name)
        for name,expected in p['dataset_table_hashes'].items():
            if file_hash(contained(Path(p['data_root'])/p['dataset_version']/(name+'.json'),Path(p['data_root'])))!=expected:
                raise InvalidArtifact('Pinned dataset metadata changed: '+name)
        if self.commands.mode=='real' and file_hash(Path(p['checkpoint']))!=p['checkpoint_sha256']:
            raise InvalidArtifact('Checkpoint SHA-256 differs from pinned profile')

    def stop_child(self,child):
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                # A blocked OS process must keep the GPU lane occupied until
                # it really exits. Never claim cancellation and unlock early.
                while child.poll() is None:
                    self.beat();time.sleep(self.poll_s)

    def command(self,job,argv,phase,paths,lock_fd):
        with self.db.transaction() as c:
            current=self.jobs.row(job['id'],c)
            if current['cancel_requested'] or self.stop_requested:raise InterruptedError('Job cancellation or worker shutdown requested')
            if phase=='inference':
                if current['inference_launch_count']!=0:raise RuntimeError('Inference launch intent already exists; refusing a duplicate')
                c.execute('UPDATE inference_jobs SET inference_launch_count=1 WHERE id=?',(job['id'],))
            self.jobs.event(c,job['id'],phase+'_launch',canonical(dict(argv=argv,execution_mode=self.commands.mode)))
        shim=[str(REPO/'.venv-web/bin/python'),str(REPO/'tools/job_subprocess.py'),'--parent-pid',str(os.getpid()),'--',*argv]
        child=None
        with (paths['stage']/(phase+'.log')).open('xb') as log:
            try:
                child=subprocess.Popen(shim,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(lock_fd,),env=dict(os.environ,PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/occfusion-matplotlib'))
                with self.db.transaction() as c:
                    c.execute('UPDATE inference_jobs SET child_pid=?,child_started_ticks=? WHERE id=?',
                        (child.pid,process_ticks(child.pid),job['id']))
                while child.poll() is None:
                    self.beat()
                    if self.jobs.row(job['id'])['cancel_requested'] or self.stop_requested:
                        self.stop_child(child);raise InterruptedError('CLI process group stopped after cancellation/worker shutdown')
                    time.sleep(self.poll_s)
                if self.jobs.row(job['id'])['cancel_requested'] or self.stop_requested:raise InterruptedError('Cancellation received before validation/publication')
                return child.returncode
            finally:
                if child is not None:self.stop_child(child)
                with self.db.connect() as c:c.execute('UPDATE inference_jobs SET child_pid=NULL,child_started_ticks=NULL WHERE id=?',(job['id'],))

    def publish(self,job,paths,meta,report,evaluation_status,evaluation):
        content=(paths['evaluation']/'evaluation.json').read_bytes()
        result=dict(run_id=job['run_id'],evaluation_directory=job['evaluation_id'],sample_token=job['sample_token'],
            evaluation_status=evaluation_status,metrics=evaluation.get('metrics'),timings_s=report['timings_s'],
            precision=report['precision'],tf32=report['tf32'],allocator=report['vram'],review_registered=False,approved=False)
        marker=dict(schema_version=1,status='complete',job_id=job['id'],run_id=job['run_id'],sample_token=job['sample_token'],
            execution_mode=self.commands.mode,profile_fingerprint=json.loads(job['profile_json'])['fingerprint'],
            artifacts=file_records(paths['prediction']),evaluation=dict(directory=job['evaluation_id'],status=evaluation_status,
                sha256=digest(content),bytes=len(content)),result=result,completed_at_utc=utc_now())
        marker['fingerprint_sha256']=digest(canonical(marker).encode())
        self.phase(job['id'],'publishing','Publishing validated prediction/evaluation without creating review approvals.')
        with self.db.transaction() as c:
            current=self.jobs.row(job['id'],c)
            if current['cancel_requested'] or self.stop_requested:raise InterruptedError('Canceled before publication')
            if paths['run'].exists() or paths['eval_run'].exists():raise FileExistsError('Job output already exists; it will never be overwritten')
            (paths['prediction']/'job_result.json').write_text(canonical(marker)+'\n')
            paths['evaluation'].rename(paths['eval_run'])
            paths['prediction'].rename(paths['run'])
            verify_publication(paths['run'],job['id'])
            c.execute("UPDATE inference_jobs SET status='succeeded',phase='succeeded',finished_at=?,result_json=?,evaluation_status=? WHERE id=?",
                (utc_now(),canonical(result),evaluation_status,job['id']))
            self.jobs.event(c,job['id'],'succeeded','Validated artifacts are available in the viewer; no review was automatically registered or approved.')

    def publish_detection(self,job,paths,meta,report):
        result=dict(run_id=job['run_id'],task=DETECTION,sample_token=job['sample_token'],box_count=report['box_count'],
            evaluation_status='pending_dataset_evaluation',metrics=report['metrics'],timings_s=report['timings_s'],precision=report['precision'],
            tf32=report['tf32'],allocator=report['vram'],review_registered=False,approved=False)
        marker=dict(schema_version=1,status='complete',task=DETECTION,job_id=job['id'],run_id=job['run_id'],sample_token=job['sample_token'],
            execution_mode=self.commands.mode,profile_fingerprint=json.loads(job['profile_json'])['fingerprint'],artifacts=file_records(paths['prediction'],DETECTION),
            result=result,completed_at_utc=utc_now())
        marker['fingerprint_sha256']=digest(canonical(marker).encode())
        self.phase(job['id'],'publishing','Publishing detection bundle; dataset evaluation pending, no automatic approval.')
        with self.db.transaction() as c:
            if self.jobs.row(job['id'],c)['cancel_requested'] or self.stop_requested:raise InterruptedError('Canceled before detection publication')
            if paths['run'].exists():raise FileExistsError('Detection output exists; overwrite refused')
            (paths['prediction']/'job_result.json').write_text(canonical(marker)+'\n');paths['prediction'].rename(paths['run'])
            verify_publication(paths['run'],job['id'])
            c.execute("UPDATE inference_jobs SET status='succeeded',phase='succeeded',finished_at=?,result_json=?,evaluation_status=? WHERE id=?",(utc_now(),canonical(result),'pending_dataset_evaluation',job['id']))
            self.jobs.event(c,job['id'],'succeeded','Detection available; mAP/NDS pending dataset evaluation; review unregistered.')

    def execute(self,job,lock_fd):
        paths=self.jobs.paths(job['id']);phase='preflight'
        try:
            paths['stage'].mkdir(parents=True,exist_ok=False)
            self.phase(job['id'],phase,'Verifying pinned profile/model/config/dataset bytes before launching the CLI.')
            self.preflight(job)
            phase='inference';self.phase(job['id'],phase,'Separate CLI running; model/checkpoint stay outside the web process.')
            code=self.command(job,self.commands.inference(job,paths),phase,paths,lock_fd)
            if code:
                error=dict(stage='inference',type='CLIExitError',message=f'Inference CLI exited with code {code}',exit_code=code)
                report=paths['prediction']/'report.json'
                if report.is_file():error.update(json.loads(report.read_text()).get('error',{}))
                self.terminal(job['id'],'failed',error=error);return
            phase='artifact_validation';self.phase(job['id'],phase,'Checking prediction token, geometry, probability, calibration and pinned provenance.')
            meta,report,files=self.results.prediction(paths['prediction'],job)
            if task_of(json.loads(job['profile_json']))==DETECTION:
                self.publish_detection(job,paths,meta,report);return
            phase='evaluation';self.phase(job['id'],phase,'CPU evaluator running against SurroundOcc; no GT is fed into inference.')
            code=self.command(job,self.commands.evaluation(job,paths),phase,paths,lock_fd)
            if code not in (0,2):raise RuntimeError(f'CPU evaluator exited with code {code}; prediction retained privately')
            phase='evaluation_validation';self.phase(job['id'],phase,'Binding evaluation by sample token, geometry, protocol and hashes.')
            status,evaluation=self.results.evaluation(paths['evaluation']/'evaluation.json',job,meta,files)
            self.publish(job,paths,meta,report,status,evaluation)
        except InterruptedError as error:
            canceled=bool(self.jobs.row(job['id'])['cancel_requested'])
            self.terminal(job['id'],'canceled' if canceled else 'failed',error=dict(stage=phase,type='Canceled' if canceled else 'WorkerStopped',message=str(error)))
        except Exception as error:
            self.terminal(job['id'],'failed',error=dict(stage=phase,type=type(error).__name__,message=str(error)))

    def run_once(self):
        self.beat()
        with gpu_lock(self.settings.gpu_lock_path) as fd:
            if fd is None or not self.recover():return False
            job=self.claim()
            if job is None:return False
            self.execute(job,fd)
            return True

    def run(self,max_jobs=None,once=False):
        count=0
        try:
            while not self.stop_requested:
                if self.run_once():count+=1
                if once or max_jobs is not None and count>=max_jobs:break
                time.sleep(self.poll_s)
            return count
        finally:
            with self.db.connect() as c:c.execute("UPDATE inference_workers SET status='stopped',heartbeat_at=? WHERE id=?",(time.time(),self.id))
