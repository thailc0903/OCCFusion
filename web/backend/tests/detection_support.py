"""Synthetic boxes only for isolated software tests, never reported as real output."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

from web.backend.app.config import REPO
from web.backend.app.detections import CONVENTION

def make_fixture(target,empty=False):
    target=Path(target);target.mkdir(parents=True,exist_ok=False)
    source=REPO/'work_dirs/centerpoint_check_cpu'
    meta=json.loads((source/'metadata.json').read_text());report=json.loads((source/'report.json').read_text())
    sha=hashlib.sha256((REPO/'tools/infer_centerpoint_mini.py').read_bytes()).hexdigest()
    for value in (meta,report):
        value.update(status='success',fixture=True,precision='float32',autocast=False);value['code']['inference_script_sha256']=sha
    report.update(check_only=False,tf32=dict(matmul=False,cudnn=True),vram=dict(peak_allocated_bytes=0,peak_reserved_bytes=0),box_count=0 if empty else 2)
    report['timings_s'].update(forward=0,pipeline_total=0)
    boxes=[] if empty else [dict(center=[5.,10.,-1.],dimensions=[4.,2.,2.],yaw=1.5707963267948966,velocity=[1.,0.],score=.95,label=0),dict(center=[-4.,15.,0.],dimensions=[1.,1.,2.],yaw=0.,velocity=[0.,0.],score=.2,label=8)]
    doc=dict(schema_version=1,task='detection',fixture=True,sample_token=meta['sample_token'],timestamp_us=meta['timestamp_us'],reference_timestamp_us=meta['reference_timestamp_us'],coordinate_system=meta['coordinate_system'],class_names=meta['class_names'],box_convention=CONVENTION,boxes=boxes,provenance={k:meta[k] for k in ('config','checkpoint','code','framework')},postprocessing=dict(nms_type='circle',config_score_threshold=.1,display_filter_applied=False))
    for name,value in [('metadata.json',meta),('report.json',report),('detections.json',doc)]:
        (target/name).write_text(json.dumps(value,indent=2)+'\n')
    shutil.copyfile(source/'resolved_config.py',target/'resolved_config.py')

class FixtureDetectionCommands:
    mode='fixture'
    def inference(self,job,paths):
        return [sys.executable,'-m','web.backend.tests.detection_support','--target',str(paths['prediction'])]
    def evaluation(self,job,paths):raise AssertionError('Detection must not call occupancy evaluation')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--target',required=True);args=parser.parse_args()
    make_fixture(args.target);print('SYNTHETIC DETECTION FIXTURE; no model/CLI/GPU inference')
