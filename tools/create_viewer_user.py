#!/usr/bin/env python3
"""Provision a local account without putting a password in argv or Git."""
import argparse
import getpass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.backend.app.auth import Identity
from web.backend.app.config import Settings
from web.backend.app.database import Database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', required=True)
    parser.add_argument('--role', choices=['engineer', 'reviewer'], required=True)
    parser.add_argument('--password-stdin', action='store_true', help='Read one password line from stdin (automation only).')
    args = parser.parse_args()
    if args.password_stdin:
        password = sys.stdin.readline().rstrip('\r\n')
    else:
        password = getpass.getpass('Password (12+ characters): ')
        if password != getpass.getpass('Confirm password: '):
            parser.error('Passwords do not match')
    settings = Settings.load()
    database = Database(settings.state_directory)
    try:
        user = Identity(database, settings).create_user(args.username, password, args.role)
    except ValueError as error:
        parser.error(str(error))
    print(f'Created {user["username"]} ({user["role"]}); database: {database.path}')


if __name__ == '__main__':
    main()
