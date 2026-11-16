"""Readiness/launcher checks cannot initialize app, DB or GPU worker."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
from tools.check_web_readiness import build_info,database_info,metadata_probe
from web.backend.app.config import REPO


def test_saved_build_rewrite_is_checked_before_startup(tmp_path):
    (tmp_path/'BUILD_ID').write_text('test-build')
    manifest=tmp_path/'routes-manifest.json'
    manifest.write_text(json.dumps(dict(rewrites=dict(beforeFiles=[dict(source='/api/:path*',destination='http://127.0.0.1:8007/api/:path*')],afterFiles=[],fallback=[]))))
    with pytest.raises(ValueError,match='rewrite mismatch'):build_info(tmp_path,'http://127.0.0.1:8000')
    assert build_info(tmp_path,'http://127.0.0.1:8007')['build_id']=='test-build'


def test_missing_database_is_not_created(tmp_path):
    path=tmp_path/'state/viewer.sqlite3'
    assert database_info(path,{})['exists'] is False
    assert not path.parent.exists()


def test_read_only_queue_check_retains_rows_and_detects_profile_mismatch(tmp_path):
    path=tmp_path/'qa.sqlite3'
    with sqlite3.connect(path) as c:
        c.executescript('CREATE TABLE users(id INTEGER); INSERT INTO users VALUES(1); CREATE TABLE inference_jobs(id,status,phase,profile_id,profile_json,inference_launch_count,created_at);')
        c.execute('INSERT INTO inference_jobs VALUES(?,?,?,?,?,?,?)',('qa','queued','queued','old',json.dumps(dict(fingerprint='old-path-case')),0,'2026-11-16'))
    before=path.read_bytes()
    result=database_info(path,{'old':dict(fingerprint='new-path-case')})
    assert result['queued']==1 and result['running']==0 and not result['jobs'][0]['current_profile_matches']
    assert path.read_bytes()==before and not list(tmp_path.glob('*-journal'))


@pytest.mark.parametrize('override',[{'VIEWER_START_WORKER':'bad'}, {'VIEWER_NEXT_DIST_DIR':'.next-camera-qa'}, {'VIEWER_API_ORIGIN':'http://127.0.0.1:8007'}])
def test_production_launcher_refuses_qa_or_invalid_worker_before_processes(tmp_path,override):
    env=dict(os.environ,VIEWER_STATE_ROOT=str(tmp_path/'state'),VIEWER_START_WORKER='0',VIEWER_NEXT_DIST_DIR='.next',
        VIEWER_API_ORIGIN='http://127.0.0.1:8000',**{})
    env.update(override)
    result=subprocess.run(['bash','tools/run_web_viewer.sh'],cwd=REPO,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and ('must be 0' in result.stderr or 'Production launcher requires' in result.stderr)
    assert not (tmp_path/'state').exists()


def test_model_probe_reads_metadata_without_importing_torch():
    result=metadata_probe('/home/hatis/.venvs/rav25/bin/python')
    assert result['torch_imported'] is False and result['packages']['mmdet3d']=='1.4.0'


def test_readiness_subprocess_has_no_app_import_or_database_creation(tmp_path):
    result=subprocess.run([sys.executable,'-B','tools/check_web_readiness.py','--dist-dir','.next-camera-qa',
        '--api-origin','http://127.0.0.1:8007'],cwd=REPO,
        env=dict(os.environ,VIEWER_STATE_ROOT=str(tmp_path/'nonexistent/state')),capture_output=True,text=True,timeout=60)
    value=json.loads(result.stdout)
    contract=next(c for c in value['checks'] if c['name']=='read_only_contract')
    assert contract['status']=='pass' and contract['evidence']['forbidden_modules']==[]
    assert not (tmp_path/'nonexistent').exists()
