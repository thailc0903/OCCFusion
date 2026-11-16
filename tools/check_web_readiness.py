#!/usr/bin/env python3
"""Read-only WSL readiness: no main import, DB creation, user/job, worker or model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from urllib.request import urlopen

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def file_hash(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):value.update(data)
    return value.hexdigest()


def build_info(folder,origin):
    folder=Path(folder)
    if not (folder/'BUILD_ID').is_file():raise ValueError('Frontend BUILD_ID missing')
    value=json.loads((folder/'routes-manifest.json').read_text())['rewrites']
    rules=value if isinstance(value,list) else sum(value.values(),[])
    destinations=[r['destination'] for r in rules if r['source']=='/api/:path*']
    if destinations!=[origin.rstrip('/')+'/api/:path*']:
        raise ValueError('Frontend API rewrite mismatch: '+repr(destinations))
    return dict(build_id=(folder/'BUILD_ID').read_text().strip(),api_origin=origin,dist_directory=str(folder))


def database_info(path,profiles):
    path=Path(path)
    if not path.is_file():return dict(exists=False,accounts=0,running=0,queued=0,jobs=[],workers=[])
    connection=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    try:
        tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        jobs=[]
        if 'inference_jobs' in tables:
            for row in connection.execute('SELECT id,status,phase,profile_id,profile_json,inference_launch_count FROM inference_jobs ORDER BY created_at'):
                saved=json.loads(row['profile_json'])
                jobs.append(dict(id=row['id'],status=row['status'],phase=row['phase'],profile_id=row['profile_id'],
                    launch_count=row['inference_launch_count'],saved_profile_fingerprint=saved.get('fingerprint'),
                    current_profile_matches=saved.get('fingerprint')==profiles.get(row['profile_id'],{}).get('fingerprint')))
        workers=[dict(r) for r in connection.execute('SELECT id,pid,status,heartbeat_at FROM inference_workers')] if 'inference_workers' in tables else []
        return dict(exists=True,schema=connection.execute('PRAGMA user_version').fetchone()[0],
            accounts=connection.execute('SELECT count(*) FROM users').fetchone()[0] if 'users' in tables else 0,
            running=sum(j['status']=='running' for j in jobs),queued=sum(j['status']=='queued' for j in jobs),jobs=jobs,workers=workers)
    finally:connection.close()


def metadata_probe(interpreter):
    # Distribution metadata does not import Torch or initialize CUDA.
    code=("import sys,json,importlib.metadata as m; "
        "names=['torch','mmcv','mmengine','mmdet','mmdet3d','nuscenes-devkit','numpy']; "
        "installed={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}; "
        "print(json.dumps(dict(python=sys.version.split()[0],prefix=sys.prefix,"
        "packages={n:installed.get(n) for n in names},torch_imported='torch' in sys.modules)))")
    result=subprocess.run([str(interpreter),'-I','-B','-c',code],capture_output=True,text=True,timeout=30,check=True)
    value=json.loads(result.stdout)
    if value['torch_imported'] or any(v is None for v in value['packages'].values()):
        raise ValueError('Model distribution metadata incomplete or probe imported Torch')
    return value


def run(args):
    from web.backend.app.config import Settings
    if args.config:os.environ['VIEWER_CONFIG']=str(args.config)
    settings=Settings.load();checks=[]
    def add(name,status,evidence):checks.append(dict(name=name,status=status,evidence=evidence))
    def attempt(name,action,required=True):
        try:
            value=action();add(name,'pass',value);return value
        except (OSError,ValueError,KeyError,TypeError,ImportError,sqlite3.Error,subprocess.SubprocessError) as error:
            add(name,'fail' if required else 'warning',str(error));return None
    add('web_interpreter','pass',dict(executable=os.path.abspath(sys.executable),prefix=sys.prefix,version=sys.version.split()[0]))
    def dependencies():
        result={}
        for line in (ROOT/'web/backend/requirements.lock').read_text().splitlines():
            if not line or line.startswith('#'):continue
            name,expected=line.split('==');actual=importlib.metadata.version(name)
            if actual!=expected:raise ValueError(name+': installed '+actual+', locked '+expected)
            result[name]=actual
        return result
    attempt('web_dependencies_locked',dependencies)
    def frontend_dependencies():
        node=ROOT/'.cache/toolchains/node/bin/node'
        executable=str(node) if node.is_file() else shutil.which('node')
        if not executable:raise ValueError('Node.js missing')
        version=subprocess.run([executable,'--version'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
        if not 24<=int(version.lstrip('v').split('.')[0])<27:raise ValueError('Node must satisfy project engines >=24 <27: '+version)
        package=json.loads((ROOT/'web/frontend/package.json').read_bytes())
        installed={}
        for name,expected in {**package['dependencies'],**package['devDependencies']}.items():
            actual=json.loads((ROOT/'web/frontend/node_modules'/name/'package.json').read_bytes())['version']
            if actual!=expected:raise ValueError(name+': installed '+actual+', locked '+expected)
            installed[name]=actual
        return dict(node=version,packages=installed)
    attempt('frontend_dependencies_locked',frontend_dependencies)
    from web.backend.app.jobs import Profiles
    profiles=attempt('profiles',lambda:Profiles(settings).all()) or {}
    for check in checks:
        if check['name']=='profiles' and check['status']=='pass':
            check['evidence']=[dict(id=p['id'],task=p.get('task','occupancy'),fingerprint=p['fingerprint'],
                python=p['python'],config=p['config'],checkpoint=p['checkpoint'],data_root=p['data_root']) for p in profiles.values()]
    inspected={}
    def verify_profile(profile):
        for name in ('python','config','checkpoint'):
            if not Path(profile[name]).is_file():raise ValueError('Missing '+name+': '+profile[name])
        if args.full:
            paths={**{ROOT/relative:expected for relative,expected in profile['source_hashes'].items()},
                Path(profile['checkpoint']):profile['checkpoint_sha256']}
            for path,expected in paths.items():
                if str(path) not in inspected:inspected[str(path)]=file_hash(path)
                if inspected[str(path)]!=expected:raise ValueError('Profile file hash mismatch: '+str(path))
            if Path(profile['checkpoint']).stat().st_size!=profile['checkpoint_bytes']:raise ValueError('Checkpoint size mismatch')
        return dict(files_present=True,hashes_checked=args.full,precision=profile['precision'],tf32=profile['tf32'])
    for identifier,profile in profiles.items():
        attempt('profile_files:'+identifier,lambda p=profile:verify_profile(p),args.mode=='inference')
    for interpreter in sorted({p['python'] for p in profiles.values()}):
        attempt('model_interpreter_metadata:'+interpreter,lambda p=interpreter:metadata_probe(p),args.mode=='inference')
    def dataset():
        summaries=[]
        for profile in profiles.values():
            root=Path(profile['data_root'])/profile['dataset_version']
            for table,expected in profile['dataset_table_hashes'].items():
                if file_hash(root/(table+'.json'))!=expected:raise ValueError('Dataset table hash mismatch: '+table)
            val=[s for s in Profiles(settings).scenes(profile['id']) if s['split']=='mini_val']
            summaries.append(dict(profile=profile['id'],metadata_tables=len(profile['dataset_table_hashes']),
                mini_val_scenes=[s['name'] for s in val],mini_val_samples=sum(len(s['frames']) for s in val)))
        return summaries
    attempt('dataset_metadata',dataset)
    database=attempt('database_read_only',lambda:database_info(settings.state_directory/'viewer.sqlite3',profiles))
    if database:
        if not database['accounts']:add('accounts','warning','No account exists; readiness does not create one.')
        pending=[j['id'] for j in database['jobs'] if j['status'] in ('queued','running') and not j['current_profile_matches']]
        if pending:add('queued_profile_identity','fail' if args.mode=='inference' else 'warning',
            dict(job_ids=pending,note='Retain the exact existing VIEWER_CONFIG/profile/root path spelling, including case. Do not rewrite queued jobs.'))
        if database['running']:add('running_jobs','warning','Inference running; wait before restart. No stop/retry performed.')
    def artifacts():
        if args.full:
            from web.backend.app.store import RunStore
            from web.backend.app.benchmarks import BenchmarkStore
            store=BenchmarkStore(settings,RunStore(settings));result=[]
            for item in store.list_benchmarks():
                snapshot=store.snapshot(item['benchmark_id'])
                result.append(dict(id=item['benchmark_id'],task=item.get('task','occupancy'),status=snapshot.detail['status'],
                    valid_frames=snapshot.detail['successful_frame_count'],expected_frames=snapshot.detail['expected_frame_count'],
                    fingerprint=snapshot.fingerprint))
            return result
        return [dict(id=p.name,kind=json.loads((p/'manifest.json').read_text()).get('kind'))
            for root in settings.benchmark_directories if root.is_dir() for p in root.iterdir() if (p/'manifest.json').is_file()]
    attempt('saved_benchmarks_hashes' if args.full else 'saved_benchmarks_inventory',artifacts)
    def camera_sources():
        policy=settings.camera_policy_path.read_bytes();count=0
        for path in (settings.state_directory/'cameras/sources').glob('*.json'):
            pin=json.loads(path.read_bytes());evidence=pin['evidence']
            if hashlib.sha256(policy).hexdigest()!=pin['policy_file']['sha256']:raise ValueError('Camera policy hash changed')
            matches=[root/evidence['filename'] for root in settings.dataset_roots if (root/evidence['filename']).is_file()]
            if len(matches)!=1 or file_hash(matches[0])!=evidence['source_image']['sha256']:raise ValueError('Camera source hash mismatch: '+evidence['channel'])
            count+=1
        return dict(registered_source_count=count,privacy_review_not_assumed=True)
    cam=attempt('camera_registered_sources',camera_sources)
    if cam and not cam['registered_source_count']:add('camera_demo','warning','Source provenance not registered in this state; registration never marks Engineer privacy-reviewed.')
    dist=ROOT/'web/frontend'/args.dist_dir
    attempt('frontend_build',lambda:build_info(dist,args.api_origin))
    sources=list((ROOT/'web/frontend/components').rglob('*.tsx'))+list((ROOT/'web/frontend/lib').rglob('*.ts'))+list((ROOT/'web/frontend/app').rglob('*.tsx'))+[ROOT/'web/frontend/app/globals.css',ROOT/'web/frontend/next.config.mjs']
    if (dist/'BUILD_ID').exists() and max(p.stat().st_mtime_ns for p in sources)>(dist/'BUILD_ID').stat().st_mtime_ns:
        add('frontend_source_age','warning','Build is older than frontend source; rebuild before deployment.')
    for port in (args.backend_port,args.frontend_port):
        with socket.socket() as probe:occupied=probe.connect_ex(('127.0.0.1',port))==0
        failed=args.ports=='free' and occupied or args.ports=='serving' and not occupied
        add('port:'+str(port),'fail' if failed else 'pass',dict(listening=occupied,expectation=args.ports))
    if args.ports=='serving':
        def health():
            with urlopen('http://127.0.0.1:'+str(args.backend_port)+'/api/health',timeout=10) as response:value=json.load(response)
            expected=hashlib.sha256(str(settings.state_directory).encode()).hexdigest()[:16]
            if value['workspace_id']!=expected or value['model_modules_loaded']:raise ValueError('Health workspace/model-module mismatch')
            return value
        attempt('health_workspace',health)
    forbidden=[n for n in sys.modules if n in ('web.backend.app.main','torch','mmcv','mmdet3d','occfusion')]
    add('read_only_contract','fail' if forbidden else 'pass',dict(forbidden_modules=forbidden,database_open_mode='ro',
        model_construction=False,worker_started=False,download=False))
    status='not_ready' if any(c['status']=='fail' for c in checks) else 'ready_with_warnings' if any(c['status']=='warning' for c in checks) else 'ready'
    return dict(status=status,mode=args.mode,config=os.environ.get('VIEWER_CONFIG',str(ROOT/'web/viewer.config.json')),
        state=str(settings.state_directory),runs_root=str(settings.runs_root),profile_file=str(settings.job_profiles_path),checks=checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--mode',choices=('view','inference'),default='view')
    parser.add_argument('--full',action='store_true',help='Hash profile files and validate saved benchmark snapshots on CPU')
    parser.add_argument('--ports',choices=('either','free','serving'),default='either')
    parser.add_argument('--backend-port',type=int,default=8000);parser.add_argument('--frontend-port',type=int,default=3000)
    parser.add_argument('--dist-dir',default=os.environ.get('VIEWER_NEXT_DIST_DIR','.next'))
    parser.add_argument('--api-origin',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    try:result=run(args)
    except (OSError,ValueError,ImportError,KeyError) as error:result=dict(status='not_ready',error=str(error),main_imported='web.backend.app.main' in sys.modules)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 1 if result['status']=='not_ready' else 0


if __name__=='__main__':raise SystemExit(main())
