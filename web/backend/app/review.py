"""Fingerprint-bound review of immutable inference artifacts; no model calls."""
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import secrets
import zipfile

from fastapi import HTTPException
import numpy as np

from .auth import require_role
from .database import utc_now
from .evaluations import EvaluationStore
from .store import geometry, contained, InvalidArtifact


ARTIFACT_NAMES = ('occupancy.npz', 'metadata.json', 'report.json', 'bev.png')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


@dataclass
class Snapshot:
    files: dict
    manifest: dict
    metadata: dict
    report: dict

    @property
    def fingerprint(self):
        return self.manifest['fingerprint_sha256']


class Reviews:
    def __init__(self, database, settings, store):
        self.db, self.settings, self.store = database, settings, store
        self.manifest_root = settings.state_directory / 'manifests'
        self.evaluations = EvaluationStore(settings)

    def read_files(self, run_id):
        folder = self.store.folder(run_id)
        paths = {name: self.store.artifact(folder, name) for name in ARTIFACT_NAMES}
        try:
            return {name: path.read_bytes() for name, path in paths.items()}
        except FileNotFoundError as error:
            raise HTTPException(409, 'Run is missing one of the four required prediction artifacts') from error

    def snapshot(self, run_id, evaluation_id=None):
        files = self.read_files(run_id)
        try:
            def reject_constant(value):
                raise ValueError(f'Non-finite JSON value: {value}')
            metadata = json.loads(files['metadata.json'], parse_constant=reject_constant)
            report = json.loads(files['report.json'], parse_constant=reject_constant)
            if metadata['status'] != 'success' or report['status'] != 'success' or report.get('check_only'):
                raise ValueError('Only successful prediction runs can be registered')
            if metadata['result_source'] != 'prediction' or not metadata['sample_token'] or metadata['sample_token'] != report['sample_token']:
                raise ValueError('metadata/report sample tokens or prediction source disagree')
            if metadata['coordinate_system']['name'] != 'LIDAR_TOP at reference_timestamp_us' or metadata['coordinate_system']['unit'] != 'metres':
                raise ValueError('Unsupported prediction coordinate system')
            _, shape, _, _ = geometry(metadata)
            with np.load(BytesIO(files['occupancy.npz']), allow_pickle=False) as archive:
                if set(archive.files) != {'labels', 'max_softmax_probability'}:
                    raise ValueError('Unexpected NPZ schema; no embedded sample token is assumed')
                labels, probability = archive['labels'], archive['max_softmax_probability']
            classes = metadata['class_names']
            free = metadata['free_label']
            palette = np.asarray(metadata['palette'])
            if not isinstance(free, int) or not 0 <= free < len(classes) or classes[free] != 'free':
                raise ValueError('Invalid background class mapping')
            if palette.shape != (len(classes), 3) or not np.isfinite(palette).all() or np.any(palette < 0) or np.any(palette > 255):
                raise ValueError('Invalid semantic palette')
            if labels.shape != shape or probability.shape != shape or labels.dtype.kind not in 'iu':
                raise ValueError('Prediction array shapes/dtype disagree with metadata')
            if labels.min() < 0 or labels.max() >= len(classes):
                raise ValueError('Prediction labels are outside the class map')
            if probability.dtype.kind != 'f' or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
                raise ValueError('Prediction probability is not finite/in [0,1]')
            histogram = {str(int(k)): int(n) for k, n in zip(*np.unique(labels, return_counts=True))}
            if histogram != report['prediction']['predicted_voxel_counts'] or report['prediction']['logits_shape'] != [1, *shape, len(classes)]:
                raise ValueError('Prediction shape/class counts disagree with report')
            if not files['bev.png'].startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('BEV artifact is not a PNG')
            for key in ('frame_index', 'precision'):
                if metadata[key] != report[key]:
                    raise ValueError(f'{key} differs between metadata and report')
            if metadata['scene']['name'] != report['scene_name'] or metadata['scene']['index'] != report['scene_index']:
                raise ValueError('Scene differs between metadata and report')
            with self.store.lock:
                self.store.catalog.lidar(metadata)
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
            raise HTTPException(409, f'Cannot register this artifact bundle: {error}') from error
        core = dict(
            schema_version=1, run_id=run_id, sample_token=metadata['sample_token'],
            sample_token_sources=['metadata.json:sample_token', 'report.json:sample_token'],
            npz_contains_sample_token=False,
            artifacts={name: dict(sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)) for name, value in files.items()},
        )
        if evaluation_id is not None:
            try:
                item, summary = self.evaluations.require_match(evaluation_id, metadata, files)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise HTTPException(409, f'Evaluation cannot be included: {error}') from error
            files['evaluation.json'] = item['content']
            core.update(schema_version=2, evaluation=dict(
                id=evaluation_id, sample_token=summary['sample_token'],
                prediction_sha256=summary['prediction_sha256'], metric_unit='ratio_0_1',
                scope='single_frame', protocol_name=summary['protocol_name']))
            core['artifacts']['evaluation.json'] = dict(sha256=item['sha256'], bytes=item['bytes'])
        fingerprint = hashlib.sha256(canonical(core).encode('utf-8')).hexdigest()
        return Snapshot(files, dict(core, fingerprint_sha256=fingerprint), metadata, report)

    def manifest_path(self, run_id, fingerprint):
        # run_id has already passed the store's directory/ID validation.
        return contained(self.manifest_root / run_id / f'{fingerprint}.json', self.manifest_root)

    def write_manifest(self, snapshot):
        path = self.manifest_path(snapshot.manifest['run_id'], snapshot.fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (canonical(snapshot.manifest) + '\n').encode('utf-8')
        if path.exists():
            if path.read_bytes() != content:
                raise HTTPException(409, 'Existing immutable sidecar manifest has been changed')
        else:
            temporary = path.with_suffix(f'.{secrets.token_hex(8)}.tmp')
            try:
                temporary.write_bytes(content)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def current(connection, run_id):
        return connection.execute('''SELECT v.*, r.owner_id,u.username AS owner_username
            FROM runs r JOIN revisions v ON v.id=r.current_revision_id
            JOIN users u ON u.id=r.owner_id WHERE r.run_id=?''', (run_id,)).fetchone()

    @staticmethod
    def has_submission(connection, run_id):
        return connection.execute('''SELECT 1 FROM runs r JOIN revisions v ON v.id=r.current_revision_id
            WHERE r.run_id=? AND v.submitted_at IS NOT NULL''', (run_id,)).fetchone() is not None

    def require_view(self, user, run_id, connection=None):
        if user['role'] == 'engineer':
            return
        if connection is None:
            with self.db.connect() as current:
                return self.require_view(user, run_id, current)
        if not self.has_submission(connection, run_id):
            raise HTTPException(403, 'Reviewer can only view runs that have been submitted')

    @staticmethod
    def event(connection, revision, user, action, from_state, to_state, comment):
        connection.execute('''INSERT INTO review_events(
            revision_id,actor_id,actor_username,actor_role,action,from_state,to_state,comment,created_at,fingerprint)
            VALUES(?,?,?,?,?,?,?,?,?,?)''',
            (revision['id'], user['user_id'], user['username'], user['role'], action,
             from_state, to_state, comment, utc_now(), revision['fingerprint']))

    def revision_files(self, revision):
        files = self.read_files(revision['run_id'])
        manifest = json.loads(revision['manifest_json'])
        if manifest['schema_version'] == 2:
            try:
                files['evaluation.json'] = self.evaluations.read(manifest['evaluation']['id'])['content']
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise HTTPException(409, f'Revision evaluation is missing or invalid: {error}') from error
        return files

    def integrity(self, revision, files=None):
        manifest = json.loads(revision['manifest_json'])
        changed, error = [], None
        try:
            files = self.revision_files(revision) if files is None else files
            expected_names = set(ARTIFACT_NAMES) | ({'evaluation.json'} if manifest['schema_version'] == 2 else set())
            if set(manifest['artifacts']) != expected_names:
                raise HTTPException(409, 'Unsupported revision manifest artifacts')
            for name in manifest['artifacts']:
                expected = manifest['artifacts'][name]
                if len(files[name]) != expected['bytes'] or hashlib.sha256(files[name]).hexdigest() != expected['sha256']:
                    changed.append(name)
        except (HTTPException, OSError) as exception:
            changed = list(manifest['artifacts'])
            error = str(getattr(exception, 'detail', exception))
        sidecar_ok = False
        try:
            path = self.manifest_path(revision['run_id'], revision['fingerprint'])
            sidecar_ok = path.read_bytes() == (canonical(manifest) + '\n').encode('utf-8')
        except OSError:
            pass
        if manifest['schema_version'] == 2:
            with self.db.connect() as connection:
                link = connection.execute('SELECT * FROM revision_evaluations WHERE revision_id=?', (revision['id'],)).fetchone()
            expected = manifest['artifacts']['evaluation.json']
            if not link or link['evaluation_id'] != manifest['evaluation']['id'] or link['sha256'] != expected['sha256'] or link['byte_count'] != expected['bytes']:
                sidecar_ok = False
        return dict(artifact_matches=not changed, changed_artifacts=changed,
                    manifest_matches=sidecar_ok, error=error,
                    approval_valid=revision['status'] == 'approved' and not changed and sidecar_ok)

    def describe(self, user, run_id):
        with self.db.connect() as connection:
            self.require_view(user, run_id, connection)
            revision = self.current(connection, run_id)
            if not revision:
                return dict(registered=False, state='unregistered', events=[],
                            permissions=dict(register=user['role'] == 'engineer'))
            integrity = self.integrity(revision)
            owner = user['role'] == 'engineer' and revision['owner_id'] == user['user_id']
            intact = integrity['artifact_matches'] and integrity['manifest_matches']
            events = [dict(row) for row in connection.execute('''SELECT e.*,v.number AS revision_number
                FROM review_events e JOIN revisions v ON v.id=e.revision_id
                WHERE v.run_id=? ORDER BY e.id''', (run_id,))]
            history = []
            for row in connection.execute('SELECT * FROM revisions WHERE run_id=? ORDER BY number', (run_id,)):
                manifest = json.loads(row['manifest_json'])
                history.append(dict(id=row['id'], number=row['number'], state=row['status'],
                                    fingerprint=row['fingerprint'], manifest_version=manifest['schema_version'],
                                    evaluation_included='evaluation.json' in manifest['artifacts'],
                                    approved_at=row['approved_at'], approved_by=row['approved_by']))
            current_manifest = json.loads(revision['manifest_json'])
            return dict(
                registered=True, state=revision['status'], revision_id=revision['id'],
                revision_number=revision['number'], fingerprint=revision['fingerprint'],
                manifest=current_manifest, revisions=history,
                evaluation_included='evaluation.json' in current_manifest['artifacts'],
                owner=dict(id=revision['owner_id'], username=revision['owner_username']),
                created_at=revision['created_at'], approved_at=revision['approved_at'],
                approved_by=revision['approved_by'], events=events, **integrity,
                permissions=dict(
                    register=False, note=owner or user['role'] == 'reviewer',
                    submit=owner and revision['status'] == 'draft' and intact,
                    decide=user['role'] == 'reviewer' and revision['status'] == 'submitted' and intact,
                    revise=owner and revision['status'] != 'submitted',
                    export=integrity['approval_valid'],
                ),
            )

    def register(self, user, run_id, comment, new_revision=False, evaluation_id=None,
                 expected_revision_id=None, expected_fingerprint=None):
        require_role(user, 'engineer')
        attachment = evaluation_id is not None
        with self.db.transaction() as connection:
            previous = self.current(connection, run_id)
            if evaluation_id is not None and (not new_revision or not previous or previous['id'] != expected_revision_id or previous['fingerprint'] != expected_fingerprint):
                raise HTTPException(409, 'Evaluation attachment requires the current revision/fingerprint')
            if previous:
                if not new_revision:
                    raise HTTPException(409, 'Run is already registered; create a new draft revision')
                if previous['owner_id'] != user['user_id']:
                    raise HTTPException(403, 'Only the registering engineer can revise this run')
                if previous['status'] == 'submitted':
                    raise HTTPException(409, 'A submitted revision must receive a review decision first')
            elif new_revision:
                raise HTTPException(409, 'Register the run first')
            if previous and new_revision and evaluation_id is None:
                evaluation_id = json.loads(previous['manifest_json']).get('evaluation', {}).get('id')
            snapshot = self.snapshot(run_id, evaluation_id)
            self.write_manifest(snapshot)
            now = utc_now()
            if not previous:
                connection.execute('INSERT INTO runs(run_id,owner_id,created_at) VALUES(?,?,?)',
                                   (run_id, user['user_id'], now))
            number = previous['number'] + 1 if previous else 1
            cursor = connection.execute('''INSERT INTO revisions(run_id,number,fingerprint,manifest_json,status,created_at)
                VALUES(?,?,?,?,?,?)''', (run_id, number, snapshot.fingerprint, canonical(snapshot.manifest), 'draft', now))
            connection.execute('UPDATE runs SET current_revision_id=? WHERE run_id=?', (cursor.lastrowid, run_id))
            if evaluation_id is not None:
                entry = snapshot.manifest['artifacts']['evaluation.json']
                connection.execute('INSERT INTO revision_evaluations VALUES(?,?,?,?)',
                                   (cursor.lastrowid, evaluation_id, entry['sha256'], entry['bytes']))
            revision = self.current(connection, run_id)
            self.event(connection, revision, user, 'attach_evaluation' if attachment else 'revise' if previous else 'register',
                       previous['status'] if previous else None, 'draft', comment)
        return self.describe(user, run_id)

    def note(self, user, run_id, comment):
        with self.db.transaction() as connection:
            self.require_view(user, run_id, connection)
            revision = self.current(connection, run_id)
            if not revision:
                raise HTTPException(409, 'Register the run before adding notes')
            if user['role'] == 'engineer' and revision['owner_id'] != user['user_id']:
                raise HTTPException(403, 'Only the registering engineer can add engineer notes')
            self.event(connection, revision, user, 'note', revision['status'], revision['status'], comment)
        return self.describe(user, run_id)

    def transition(self, user, run_id, action, comment, revision_id, fingerprint):
        target_role = 'engineer' if action == 'submitted' else 'reviewer'
        require_role(user, target_role)
        with self.db.transaction() as connection:
            revision = self.current(connection, run_id)
            if not revision or revision['id'] != revision_id or revision['fingerprint'] != fingerprint:
                raise HTTPException(409, 'Review revision/fingerprint is stale; reload the run')
            if action == 'submitted':
                if revision['owner_id'] != user['user_id']:
                    raise HTTPException(403, 'Only the registering engineer can submit this run')
                if revision['status'] != 'draft':
                    raise HTTPException(409, 'Only a draft revision can be submitted')
            elif revision['status'] != 'submitted':
                raise HTTPException(409, 'Only a submitted revision can receive a review decision')
            integrity = self.integrity(revision)
            if not integrity['artifact_matches'] or not integrity['manifest_matches']:
                raise HTTPException(409, dict(reason='artifact_changed', **integrity))
            now = utc_now()
            if action == 'submitted':
                connection.execute('UPDATE revisions SET status=?,submitted_at=? WHERE id=?', (action, now, revision_id))
            else:
                connection.execute('UPDATE revisions SET status=?,approved_by=?,approved_at=? WHERE id=?',
                                   (action, user['user_id'] if action == 'approved' else None,
                                    now if action == 'approved' else None, revision_id))
            self.event(connection, revision, user, 'submit' if action == 'submitted' else 'decision',
                       revision['status'], action, comment)
        return self.describe(user, run_id)

    def export(self, user, run_id):
        # Build the archive from these exact checked bytes, never reopen sources
        # after the integrity check (avoids a check-then-read export race).
        with self.db.transaction() as connection:
            self.require_view(user, run_id, connection)
            revision = self.current(connection, run_id)
            if not revision or revision['status'] != 'approved':
                raise HTTPException(409, 'Only the current approved revision can be exported')
            files = self.revision_files(revision)
            integrity = self.integrity(revision, files)
            if not integrity['approval_valid']:
                raise HTTPException(409, dict(reason='approval_does_not_match_artifacts', **integrity))
            self.event(connection, revision, user, 'export', 'approved', 'approved', 'Exported a hash-verified approved bundle')
            events = [dict(row) for row in connection.execute('''SELECT e.*,v.number AS revision_number
                FROM review_events e JOIN revisions v ON v.id=e.revision_id
                WHERE v.run_id=? ORDER BY e.id''', (run_id,))]
            review_info = dict(
                schema_version=1, run_id=run_id, revision_id=revision['id'], revision_number=revision['number'],
                status='approved', fingerprint_sha256=revision['fingerprint'],
                approved_by=revision['approved_by'], approved_at=revision['approved_at'],
                exported_at=utc_now(), exported_by=dict(id=user['user_id'], username=user['username'], role=user['role']),
                metrics=json.loads(files['report.json']).get('metrics', {}), events=events,
            )
            manifest = json.loads(revision['manifest_json'])
            review_info.update(manifest_schema_version=manifest['schema_version'], metric_source='report.json')
            if 'evaluation.json' in files:
                review_info.update(metrics=json.loads(files['evaluation.json'])['metrics'],
                                   metric_source='evaluation.json', metric_unit='ratio_0_1')
            output = BytesIO()
            with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for name in manifest['artifacts']:
                    archive.writestr(name, files[name])
                archive.writestr('manifest.json', revision['manifest_json'] + '\n')
                archive.writestr('review.json', json.dumps(review_info, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            return output.getvalue(), revision['fingerprint']
