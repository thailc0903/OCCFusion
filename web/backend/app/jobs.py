"""Persistent one-frame job queue. Web process only reads metadata and SQLite."""
import json
import os
from pathlib import Path
import re
import time
import uuid

from fastapi import HTTPException

from .auth import require_role
from .config import REPO
from .database import utc_now
from .evaluations import decode, digest
from .review import canonical
from .store import contained, InvalidArtifact


TERMINAL={'succeeded','failed','canceled'}
JOB_ID=re.compile(r'^[0-9a-f]{32}$')


class Profiles:
    def __init__(self,settings):
        self.settings=settings

    def all(self):
        source=self.settings.job_profiles_path.resolve()
        value=decode(source.read_bytes())
        if value['schema_version']!=1: raise InvalidArtifact('Unsupported inference profiles schema')
        result={}
        for item in value['profiles']:
            profile=dict(item)
            if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',profile['id']) or profile['id'] in result:
                raise InvalidArtifact('Invalid or duplicate profile id')
            for name in ('python','config','checkpoint','data_root','gt_root'):
                path=Path(profile[name]).expanduser()
                path=source.parent/path if not path.is_absolute() else path
                # Resolving bin/python symlinks loses the venv's pyvenv.cfg.
                # Execute the configured interpreter path, not /usr/bin/python.
                profile[name]=os.path.abspath(path) if name=='python' else str(path.resolve())
            if profile['precision']!='float32' or profile['autocast'] is not False or profile['batch_size']!=1 or profile['device']!='cuda:0' or profile['tf32']!={'matmul':False,'cudnn':True}:
                raise InvalidArtifact('Profile must retain the verified FP32/C+L+R baseline')
            if not any(Path(profile['data_root'])==root.resolve() for root in self.settings.dataset_roots):
                raise InvalidArtifact('Profile dataset is outside configured dataset roots')
            profile['fingerprint']=digest(canonical(profile).encode())
            result[profile['id']]=profile
        return result

    def get(self,identifier):
        result=self.all()
        if identifier not in result: raise HTTPException(404,'Unknown inference profile')
        return result[identifier]

    def options(self):
        result=[]
        for profile in self.all().values():
            missing=[name for name in ('python','config','checkpoint') if not Path(profile[name]).is_file()]
            result.append(dict(id=profile['id'],label=profile['label'],precision=profile['precision'],
                device=profile['device'],tf32=profile['tf32'],geometry=profile['geometry'],
                checkpoint=Path(profile['checkpoint']).name,checkpoint_sha256=profile['checkpoint_sha256'],
                config=Path(profile['config']).name,fingerprint=profile['fingerprint'],available=not missing,
                unavailable_reason='Missing '+', '.join(missing) if missing else None))
        return result

    def scenes(self,identifier):
        profile=self.get(identifier);root=Path(profile['data_root'])/profile['dataset_version']
        samples={row['token']:row for row in json.loads((root/'sample.json').read_bytes())}
        scenes=json.loads((root/'scene.json').read_bytes());result=[]
        for scene in sorted(scenes,key=lambda row:row['name']):
            splits=[name for name,values in profile['splits'].items() if scene['name'] in values]
            if len(splits)!=1: raise InvalidArtifact('Scene must belong to one official mini split')
            frames=[];token=scene['first_sample_token'];seen=set()
            while token:
                if token in seen:raise InvalidArtifact('Cycle in nuScenes sample chain')
                seen.add(token);sample=samples[token]
                if sample['scene_token']!=scene['token']:raise InvalidArtifact('Sample chain crosses scene')
                frames.append(dict(frame_index=len(frames),sample_token=token,timestamp_us=sample['timestamp']))
                token=sample['next']
            if len(frames)!=scene['nbr_samples'] or frames[-1]['sample_token']!=scene['last_sample_token']:
                raise InvalidArtifact('Scene sample count mismatch')
            result.append(dict(name=scene['name'],scene_token=scene['token'],split=splits[0],frames=frames))
        return result


class Jobs:
    def __init__(self,database,settings):
        self.db,self.settings=database,settings
        self.profiles=Profiles(settings)
        self.staging=contained(settings.runs_root/'.inference-jobs',settings.runs_root)

    def paths(self,identifier):
        if not JOB_ID.fullmatch(identifier):raise HTTPException(404,'Unknown job id')
        stage=contained(self.staging/identifier,self.staging)
        return dict(stage=stage,prediction=stage/'prediction',evaluation=stage/'evaluation',
            run=contained(self.settings.runs_root/('infer_job_'+identifier),self.settings.runs_root),
            eval_run=contained(self.settings.runs_root/('eval_job_'+identifier),self.settings.runs_root))

    @staticmethod
    def event(connection,identifier,phase,message,actor_id=None,actor_type='worker'):
        connection.execute('''INSERT INTO inference_job_events(job_id,actor_id,actor_type,phase,message,created_at)
            VALUES(?,?,?,?,?,?)''',(identifier,actor_id,actor_type,phase,message,utc_now()))

    def create(self,user,profile_id,scene_name,frame_index,request_key):
        require_role(user,'engineer')
        profile=self.profiles.get(profile_id)
        scene=next((scene for scene in self.profiles.scenes(profile_id) if scene['name']==scene_name),None)
        if scene is None or not 0<=frame_index<len(scene['frames']):raise HTTPException(422,'Scene/frame is outside nuScenes mini metadata')
        frame=scene['frames'][frame_index];identifier=uuid.uuid4().hex
        with self.db.transaction() as connection:
            previous=connection.execute('SELECT * FROM inference_jobs WHERE owner_id=? AND request_key=?',(user['user_id'],request_key)).fetchone()
            if previous:
                if previous['profile_json']!=canonical(profile) or previous['scene_name']!=scene_name or previous['frame_index']!=frame_index:
                    raise HTTPException(409,'Request key was already used for different job parameters/profile')
                identifier=previous['id']
            else:
                connection.execute('''INSERT INTO inference_jobs(id,owner_id,request_key,profile_id,profile_json,scene_name,frame_index,
                    sample_token,split,status,phase,created_at,run_id,evaluation_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (identifier,user['user_id'],request_key,profile_id,canonical(profile),scene_name,frame_index,
                     frame['sample_token'],scene['split'],'queued','queued',utc_now(),'infer_job_'+identifier,'eval_job_'+identifier))
                self.event(connection,identifier,'queued','Engineer requested one-frame inference; no model runs in the web process.',user['user_id'],'engineer')
        return self.describe(user,identifier)

    def row(self,identifier,connection=None):
        if not JOB_ID.fullmatch(identifier):raise HTTPException(404,'Unknown job id')
        if connection is None:
            with self.db.connect() as c:return self.row(identifier,c)
        value=connection.execute('SELECT * FROM inference_jobs WHERE id=?',(identifier,)).fetchone()
        if value is None:raise HTTPException(404,'Unknown job id')
        return value

    @staticmethod
    def owner(user,job):
        require_role(user,'engineer')
        if job['owner_id']!=user['user_id']:raise HTTPException(403,'Only the requesting engineer can access this job')

    def describe(self,user,identifier):
        with self.db.connect() as c:
            job=self.row(identifier,c);self.owner(user,job)
            events=[dict(row) for row in c.execute('SELECT * FROM inference_job_events WHERE job_id=? ORDER BY id',(identifier,))]
            worker=c.execute('SELECT * FROM inference_workers WHERE id=?',(job['worker_id'],)).fetchone() if job['worker_id'] else None
        profile=json.loads(job['profile_json'])
        return dict(id=identifier,profile_id=job['profile_id'],profile_label=profile['label'],scene_name=job['scene_name'],
            frame_index=job['frame_index'],sample_token=job['sample_token'],split=job['split'],status=job['status'],phase=job['phase'],
            created_at=job['created_at'],started_at=job['started_at'],finished_at=job['finished_at'],
            cancel_requested=bool(job['cancel_requested']),execution_mode=job['execution_mode'],
            worker_stale=bool(worker and time.time()-worker['heartbeat_at']>30),
            inference_launch_count=job['inference_launch_count'],evaluation_status=job['evaluation_status'],
            result=json.loads(job['result_json']) if job['result_json'] else None,error=json.loads(job['error_json']) if job['error_json'] else None,
            events=events,result_url='/?run='+job['run_id'] if job['status']=='succeeded' else None,
            precision=profile['precision'],tf32=profile['tf32'],config=Path(profile['config']).name,
            checkpoint=Path(profile['checkpoint']).name,checkpoint_sha256=profile['checkpoint_sha256'],
            review_note='Job completion does not register, submit or approve a review revision.')

    def list(self,user):
        require_role(user,'engineer')
        with self.db.connect() as c:
            rows=c.execute('SELECT id FROM inference_jobs WHERE owner_id=? ORDER BY created_at DESC LIMIT 100',(user['user_id'],)).fetchall()
            workers=[dict(id=row['id'],status=row['status'],execution_mode=row['execution_mode'],
                heartbeat_age_s=max(0,time.time()-row['heartbeat_at'])) for row in c.execute(
                "SELECT * FROM inference_workers WHERE heartbeat_at>? AND status='active'",(time.time()-30,))]
        return dict(jobs=[self.describe(user,row['id']) for row in rows],workers=workers)

    def cancel(self,user,identifier):
        with self.db.transaction() as c:
            job=self.row(identifier,c);self.owner(user,job)
            if job['status'] in TERMINAL:raise HTTPException(409,'A finished job cannot be canceled or retried')
            if job['status']=='queued':
                c.execute("UPDATE inference_jobs SET status='canceled',phase='canceled',cancel_requested=1,finished_at=? WHERE id=?",(utc_now(),identifier))
                message='Canceled while queued; no inference launched.'
            else:
                c.execute('UPDATE inference_jobs SET cancel_requested=1 WHERE id=?',(identifier,))
                message='Cancellation requested; worker will terminate its CLI process group before releasing the GPU lock.'
            self.event(c,identifier,'cancel_requested',message,user['user_id'],'engineer')
        return self.describe(user,identifier)

    def log(self,user,identifier,kind,offset=0):
        self.owner(user,self.row(identifier))
        if kind not in ('inference','evaluation'):raise HTTPException(404,'Unknown log stream')
        path=contained(self.paths(identifier)['stage']/(kind+'.log'),self.staging)
        if not path.is_file():return dict(text='',next_offset=offset)
        with path.open('rb') as stream:
            stream.seek(min(offset,path.stat().st_size));content=stream.read(65536);next_offset=stream.tell()
        return dict(text=content.decode('utf-8',errors='replace'),next_offset=next_offset)
