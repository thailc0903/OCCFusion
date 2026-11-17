"""Selective SurroundOcc ZIP ingestion. Never extractall or overwrite existing GT."""
from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile

import numpy as np

from tools import evaluate_occupancy as occupancy
from tools.forecasting.protocol import file_record, write_json

MAX_GT_BYTES = 64 * 1024 * 1024
NAME = re.compile(r'[^/\\:]+__LIDAR_TOP__[0-9]+\.pcd\.bin\.npy\Z')


def contained(path, root):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes configured root: ' + str(path))
    return path


def validate_member(info):
    name = info.filename
    if info.orig_filename != name or '\x00' in info.orig_filename:
        raise ValueError('Truncated/unsafe original archive path')
    parts = PurePosixPath(name).parts
    if not name or '\\' in name or ':' in name or '\x00' in name or name.startswith('/') or '..' in parts or '.' in name.split('/'):
        raise ValueError('Unsafe archive path: ' + repr(name))
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if info.is_dir():
        if name not in ('nuscenes_occ/', 'nuscenes_occ/samples/') or kind not in (0, stat.S_IFDIR):
            raise ValueError('Unexpected archive directory/type: ' + name)
        return None
    if kind not in (0, stat.S_IFREG):
        raise ValueError('Archive symlink/special member forbidden: ' + name)
    if info.flag_bits & 1:
        raise ValueError('Encrypted member forbidden: ' + name)
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError('Unsupported compression: ' + name)
    if len(parts) != 3 or parts[:2] != ('nuscenes_occ', 'samples') or not NAME.fullmatch(parts[2]):
        raise ValueError('Unexpected SurroundOcc member layout: ' + name)
    if name != '/'.join(parts):
        raise ValueError('Noncanonical archive member path')
    if not 0 < info.file_size <= MAX_GT_BYTES:
        raise ValueError('GT member outside size limit: ' + name)
    return parts[2]


def inspect_archives(paths):
    """Validate the entire central directory, hash archives, do not extract any data."""
    index = defaultdict(list)
    reports = []
    for path in sorted(map(Path, paths)):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Archive must be a regular configured file')
        record = file_record(path)
        with zipfile.ZipFile(path) as archive:
            names = set()
            total = 0
            count = 0
            for info in archive.infolist():
                if info.filename in names:
                    raise ValueError('Duplicate archive member: ' + info.filename)
                names.add(info.filename)
                basename = validate_member(info)
                if basename:
                    count += 1; total += info.file_size
                    index[basename].append(dict(archive=str(path.resolve()), member=info.filename,
                        uncompressed_bytes=info.file_size, compressed_bytes=info.compress_size, crc32=f'{info.CRC:08x}'))
        reports.append(dict(path=str(path.resolve()), file=record, entries=len(names), npy_count=count,
                            declared_uncompressed_bytes=total, structure_status='valid',
                            crc_scope='CRC checked only for subsequently streamed selected/overlap members; unselected payload not decompressed'))
    if not reports:
        raise ValueError('No ZIP archive found')
    return index, reports


def member_bytes(entry):
    """Bounded streaming reads to EOF make zipfile verify the actual member CRC."""
    with zipfile.ZipFile(entry['archive']) as archive:
        info = archive.getinfo(entry['member'])
        validate_member(info)
        if info.file_size != entry['uncompressed_bytes'] or f'{info.CRC:08x}' != entry['crc32']:
            raise ValueError('Archive directory changed after inspection')
        with archive.open(info) as stream:
            content = stream.read(MAX_GT_BYTES + 1)
            if len(content) != info.file_size or len(content) > MAX_GT_BYTES or stream.read(1):
                raise ValueError('Archive member length mismatch')
    return content


def unique_content(entries):
    first = member_bytes(entries[0])
    fingerprint = occupancy.sha256(first)
    for entry in entries[1:]:
        if occupancy.sha256(member_bytes(entry)) != fingerprint:
            raise ValueError('Conflicting same-basename archive members: ' + entry['member'])
    return first


def decode_content(content):
    """Check NPY header/payload bounds before NumPy may allocate the declared shape."""
    stream = BytesIO(content)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError('Unsupported sparse NPY version')
    if len(shape) != 2 or shape[1] != 4 or shape[0] < 0 or dtype.kind not in 'iuf' or dtype.hasobject:
        raise ValueError('GT must be numeric N x 4, no object dtype')
    if stream.tell() + shape[0] * 4 * dtype.itemsize != len(content):
        raise ValueError('NPY header/payload length mismatch')
    rows = np.load(BytesIO(content), allow_pickle=False)
    dense, stats = occupancy.sparse_to_dense(rows)
    return rows, dense, stats


def target_path(root, basename):
    path = Path(root) / basename
    if path.is_symlink():
        raise ValueError('Symlink GT destination forbidden')
    return contained(path, root)


def stage_missing(index, requested, staging, gt_root, *, existing_index=None):
    """Validate all requested labels/conflicts before publishing even the first GT."""
    staging, gt_root = Path(staging).resolve(), Path(gt_root).resolve()
    if staging.exists() and any(staging.iterdir()):
        raise ValueError('Staging is not empty')
    staging.mkdir(parents=True, exist_ok=True)
    if existing_index is None:
        existing_index = occupancy.GroundTruthIndex(gt_root)
    selected = []
    seen = set()
    # Check all existing/archive overlaps, including entries not selected for import.
    overlaps = []
    for basename in sorted(set(index) & set(existing_index.files)):
        paths = existing_index.files[basename]
        if len(paths) != 1:
            raise ValueError('Ambiguous existing GT basename: ' + basename)
        content = unique_content(index[basename])
        actual = file_record(contained(paths[0], gt_root))
        expected = dict(sha256=occupancy.sha256(content), bytes=len(content))
        if actual != expected:
            raise ValueError('Existing GT conflict, no overwrite: ' + basename)
        overlaps.append(dict(basename=basename, file=actual, status='identical_kept'))
    for request in requested:
        basename = request['expected_gt_filename']
        if not NAME.fullmatch(basename) or basename in seen:
            raise ValueError('Unsafe/duplicate requested GT basename: ' + basename)
        seen.add(basename)
        if basename != Path(request['lidar_filename']).name + '.npy':
            raise ValueError('Requested GT filename does not match LiDAR basename')
        entries = index.get(basename, [])
        if not entries:
            raise FileNotFoundError('Requested mini_train GT absent from archive: ' + basename)
        content = unique_content(entries)
        rows, _, stats = decode_content(content)
        destination = contained(staging / basename, staging)
        with destination.open('xb') as stream:
            stream.write(content)
        record = dict(sha256=occupancy.sha256(content), bytes=len(content))
        _, existing = existing_index.lookup(request['lidar_filename'])
        if existing and file_record(existing) != record:
            raise ValueError('Existing requested GT conflict: ' + basename)
        selected.append(dict(**request, archive_entries=entries, staged_path=str(destination), file=record,
                             sparse_shape=list(rows.shape), sparse_dtype=str(rows.dtype), decoder_statistics=stats,
                             status='staged_valid', already_exists=bool(existing)))
    return dict(status='staged_valid', selected_count=len(selected), selected_bytes=sum(s['file']['bytes'] for s in selected),
                overlaps=overlaps, files=selected)


def publish_staged(staged, gt_root, receipt_dir):
    """Copy into new target temp files, then exclusive link; no staged/GT inode sharing."""
    gt_root = Path(gt_root).resolve()
    receipt_dir = Path(receipt_dir)
    if not gt_root.is_dir():
        raise ValueError('Existing GT destination directory required')
    # Preflight the entire batch again before any publication.
    for item in staged['files']:
        if file_record(item['staged_path']) != item['file']:
            raise ValueError('Staged GT hash changed')
        target = target_path(gt_root, item['expected_gt_filename'])
        if target.exists() and file_record(target) != item['file']:
            raise ValueError('Existing GT conflict before publish: ' + target.name)
    results = []
    for item in staged['files']:
        target = target_path(gt_root, item['expected_gt_filename'])
        if target.exists():
            status = 'identical_existing_kept'
        else:
            temporary = contained(gt_root / ('.' + target.name + '.' + uuid.uuid4().hex + '.pending'), gt_root)
            try:
                with Path(item['staged_path']).open('rb') as source, temporary.open('xb') as output:
                    shutil.copyfileobj(source, output)
                    output.flush(); os.fsync(output.fileno())
                if file_record(temporary) != item['file']:
                    raise ValueError('Copied GT hash mismatch')
                try:
                    target.hardlink_to(temporary)
                    status = 'installed'
                except FileExistsError:
                    if file_record(target) != item['file']:
                        raise ValueError('Concurrent GT conflict; no overwrite')
                    status = 'identical_existing_kept'
            finally:
                temporary.unlink(missing_ok=True)
        result = dict(sample_token=item['sample_token'], scene=item['scene'], frame_index=item['frame_index'],
                      destination=str(target), file=item['file'], status=status)
        write_json(receipt_dir / (item['sample_token'] + '.json'), result)
        results.append(result)
    return dict(status='complete', installed=sum(r['status']=='installed' for r in results),
                identical_existing_kept=sum(r['status']=='identical_existing_kept' for r in results), files=results)
