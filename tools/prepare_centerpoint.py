#!/usr/bin/env python3
"""Fetch the exact MMDetection3D v1.4.0 CenterPoint Pillar02/circle pair."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'ckpt/centerpoint_pillar02_v140'
CONFIG='configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py'
BASE='https://raw.githubusercontent.com/open-mmlab/mmdetection3d/v1.4.0/'
WEIGHTS='https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'

def fetch(relative,url):
    path=ASSETS/relative;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        pending=path.with_suffix(path.suffix+'.pending')
        with urllib.request.urlopen(url,timeout=90) as response,pending.open('wb') as out:
            while block:=response.read(4*1024*1024):out.write(block)
        pending.rename(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    records[relative]=dict(url=url,sha256=digest,bytes=path.stat().st_size)
    return path

def config(relative):
    if relative in records:return
    path=fetch(relative,BASE+relative)
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='_base_' for n in node.targets):
            values=ast.literal_eval(node.value)
            for item in ([values] if isinstance(values,str) else values):
                target=(path.parent/item).resolve();assert target.is_relative_to(ASSETS)
                config(str(target.relative_to(ASSETS)))

if __name__=='__main__':
    records={};ASSETS.mkdir(parents=True,exist_ok=True)
    fetch('metafile.yml',BASE+'configs/centerpoint/metafile.yml');config(CONFIG)
    fetch('model.pth',WEIGHTS)
    assert records['model.pth']['sha256'].startswith('191a3822'),'Checkpoint does not match official model-zoo checksum prefix'
    result=dict(schema_version=1,task='detection',framework='mmdetection3d',version='1.4.0',config=CONFIG,checkpoint='model.pth',model_zoo=BASE+'configs/centerpoint/metafile.yml',files=records)
    (ASSETS/'assets.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
