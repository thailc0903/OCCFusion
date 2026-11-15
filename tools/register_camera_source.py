#!/usr/bin/env python3
"""Register hashes of verified release images; does not mark them privacy-reviewed."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    scope=parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--run-id');scope.add_argument('--benchmark-id')
    parser.add_argument('--sample-token')
    parser.add_argument('--state-root',type=Path,required=True,help='Explicit web state; use isolated QA for testing')
    args=parser.parse_args()
    if bool(args.benchmark_id)!=bool(args.sample_token):parser.error('--sample-token is required only for --benchmark-id')
    from web.backend.app.config import Settings
    from web.backend.app.database import Database
    from web.backend.app.store import RunStore
    from web.backend.app.review import Reviews
    from web.backend.app.benchmarks import BenchmarkStore
    from web.backend.app.cameras import CameraStore
    settings=replace(Settings.load(),state_root=args.state_root.resolve())
    store=RunStore(settings);reviews=Reviews(Database(settings.state_directory),settings,store)
    cameras=CameraStore(settings,store,reviews,BenchmarkStore(settings,store))
    identifier=cameras.identifier('run' if args.run_id else 'benchmark',args.run_id or args.benchmark_id,args.sample_token)
    result=cameras.register_sources(identifier)
    print(json.dumps(dict(status='registered',scope_id=identifier,images=result,human_review='not_reviewed',
        source='nuScenes publisher release policy; no automatic PII detection',model_modules_loaded='torch' in sys.modules),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
