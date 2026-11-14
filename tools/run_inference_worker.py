#!/usr/bin/env python3
"""Run the single-GPU job worker using .venv-web; CLI subprocess uses rav25."""
import argparse
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from web.backend.app.config import Settings
from web.backend.app.database import Database
from web.backend.app.job_worker import Worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--max-jobs',type=int)
    args=parser.parse_args()
    if args.max_jobs is not None and args.max_jobs<1:parser.error('--max-jobs must be positive')
    settings=Settings.load();worker=Worker(Database(settings.state_directory),settings)
    def stop(_signum,_frame):worker.stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print('Inference worker ready: separate CLI execution, one GPU lane; interrupted jobs are never automatically re-inferred.',flush=True)
    print('Worker processed',worker.run(max_jobs=args.max_jobs,once=args.once),'jobs',flush=True)


if __name__=='__main__':main()
