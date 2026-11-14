#!/usr/bin/env python3
"""Linux exec shim: parent death terminates the CLI; inherited FD keeps its lane locked."""
import argparse
import ctypes
import os
import signal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:parser.error('Missing CLI command')
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'Cannot establish parent-death protection')
    if os.getppid()!=args.parent_pid:raise SystemExit('Worker died before CLI exec; inference not launched')
    os.execv(command[0],command)


if __name__=='__main__':main()
