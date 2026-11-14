"""Test-only CLI/worker: copy genuine saved artifacts, never synthesize predictions."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

from web.backend.app.config import REPO, Settings
from web.backend.app.database import Database
from web.backend.app.job_worker import Worker

SOURCE=REPO/'work_dirs/infer_mini_val_scene0103_f1'
EVALUATION=REPO/'work_dirs/eval_mini_val_scene0103_f1/evaluation.json'


class FixtureCommands:
    mode='fixture'

    def __init__(self,inference_mode='success',evaluation_mode='success'):
        self.inference_mode,self.evaluation_mode=inference_mode,evaluation_mode

    def inference(self,job,paths):
        return [sys.executable,'-u','-m','web.backend.tests.job_support','copy-prediction',
            '--target',str(paths['prediction']),'--mode',self.inference_mode]

    def evaluation(self,job,paths):
        return [sys.executable,'-u','-m','web.backend.tests.job_support','copy-evaluation',
            '--target',str(paths['evaluation']/'evaluation.json'),'--mode',self.evaluation_mode]


def settings_to_file(settings,path):
    value=asdict(settings)
    path.write_text(json.dumps(value,default=str))


def read_settings(path):
    value=json.loads(Path(path).read_text())
    for key in ('runs_root','state_root','job_profiles_path','gpu_lock_path'):
        if value.get(key):value[key]=Path(value[key])
    for key in ('dataset_roots','evaluation_roots','benchmark_roots'):
        if value.get(key) is not None:value[key]=tuple(Path(p) for p in value[key])
    value['allowed_origins']=tuple(value['allowed_origins'])
    return Settings(**value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['copy-prediction','copy-evaluation','worker'])
    parser.add_argument('--target');parser.add_argument('--mode',default='success')
    args=parser.parse_args()
    if args.action=='worker':
        settings=read_settings(args.target)
        Worker(Database(settings.state_directory),settings,FixtureCommands(args.mode)).run(max_jobs=1)
        return
    target=Path(args.target);target.parent.mkdir(parents=True,exist_ok=True)
    print('TEST FIXTURE: copying real saved artifacts; no model or GPU execution.',flush=True)
    if args.mode=='slow':
        print('Waiting in fake CLI for queue/cancel/restart tests.',flush=True)
        time.sleep(90)
    if args.action=='copy-prediction':
        target.mkdir(exist_ok=False)
        if args.mode=='fail':
            (target/'report.json').write_text(json.dumps(dict(status='oom',error=dict(stage='forward',type='OutOfMemoryError',message='Injected test-only CLI failure'))))
            raise SystemExit(1)
        names=('metadata.json',) if args.mode=='missing' else ('occupancy.npz','metadata.json','report.json','bev.png','resolved_config.py')
        for name in names:shutil.copyfile(SOURCE/name,target/name)
        if args.mode=='wrong-token':
            path=target/'metadata.json';value=json.loads(path.read_text());value['sample_token']='0'*32;path.write_text(json.dumps(value))
    else:
        value=json.loads(EVALUATION.read_text())
        if args.mode=='wrong-token':value['frames'][0]['sample_token']='0'*32
        if args.mode=='wrong-hash':value['frames'][0]['prediction_sha256']='0'*64
        if args.mode=='success':shutil.copyfile(EVALUATION,target)
        else:target.write_text(json.dumps(value))


if __name__=='__main__':main()
