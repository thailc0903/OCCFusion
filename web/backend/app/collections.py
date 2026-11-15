"""Independent, persistent review/export of a benchmark collection snapshot."""
import json
import logging
from pathlib import Path
import secrets
import zipfile
import time

from fastapi import HTTPException

from .auth import require_role
from .database import utc_now
from .evaluations import digest
from .review import canonical
from .store import contained, InvalidArtifact


class CollectionReviews:
    def __init__(self, database, settings, benchmarks):
        self.db,self.settings,self.benchmarks=database,settings,benchmarks
        self.root=settings.state_directory/'collection-manifests'

    @staticmethod
    def current(connection, identifier):
        return connection.execute('''SELECT v.*,c.owner_id,u.username AS owner_username
            FROM collections c JOIN collection_revisions v ON v.id=c.current_revision_id
            JOIN users u ON u.id=c.owner_id WHERE c.collection_id=?''',(identifier,)).fetchone()

    @staticmethod
    def has_submission(connection, identifier):
        return connection.execute('''SELECT 1 FROM collections c JOIN collection_revisions v
            ON v.id=c.current_revision_id WHERE c.collection_id=? AND v.submitted_at IS NOT NULL''',
            (identifier,)).fetchone() is not None

    def require_view(self, user, identifier, connection=None):
        if user['role']=='engineer': return
        if connection is None:
            with self.db.connect() as c: return self.require_view(user,identifier,c)
        if not self.has_submission(connection,identifier):
            raise HTTPException(403,'Reviewer can only view submitted collections')

    def manifest_path(self, identifier, fingerprint):
        self.benchmarks.folder(identifier)
        return contained(self.root/identifier/(fingerprint+'.json'),self.root)

    def write_manifest(self, snapshot):
        path=self.manifest_path(snapshot.detail['benchmark_id'],snapshot.fingerprint)
        content=(canonical(snapshot.manifest)+'\n').encode()
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if path.read_bytes()!=content: raise HTTPException(409,'Collection sidecar has been changed')
        else:
            temporary=path.with_suffix('.'+secrets.token_hex(8)+'.tmp')
            try:
                temporary.write_bytes(content);temporary.replace(path)
            finally: temporary.unlink(missing_ok=True)

    @staticmethod
    def event(connection, revision, user, action, target, comment, from_state=...):
        connection.execute('''INSERT INTO collection_events(revision_id,actor_id,actor_username,actor_role,
            action,from_state,to_state,comment,created_at,fingerprint) VALUES(?,?,?,?,?,?,?,?,?,?)''',
            (revision['id'],user['user_id'],user['username'],user['role'],action,revision['status'] if from_state is ... else from_state,target,
             comment,utc_now(),revision['fingerprint']))

    def integrity(self, revision, snapshot=None):
        error=None;manifest=json.loads(revision['manifest_json']);artifact_ok=False
        try:
            snapshot=snapshot or self.benchmarks.snapshot(revision['collection_id'])
            artifact_ok=snapshot.manifest==manifest and snapshot.fingerprint==revision['fingerprint']
            if not artifact_ok: error='Collection artifact membership or hash changed'
        except (OSError,ValueError,KeyError,TypeError,IndexError) as exception:
            error=str(exception)
        try:
            sidecar_ok=self.manifest_path(revision['collection_id'],revision['fingerprint']).read_bytes()==(canonical(manifest)+'\n').encode()
        except (OSError,ValueError): sidecar_ok=False
        return dict(artifact_matches=artifact_ok,manifest_matches=sidecar_ok,error=error,
                    approval_valid=revision['status']=='approved' and artifact_ok and sidecar_ok)

    def describe(self, user, identifier, snapshot=None):
        with self.db.connect() as connection:
            self.require_view(user,identifier,connection)
            revision=self.current(connection,identifier)
            if not revision:
                return dict(registered=False,state='unregistered',events=[],permissions=dict(register=user['role']=='engineer'))
            integrity=self.integrity(revision,snapshot)
            owner=user['role']=='engineer' and revision['owner_id']==user['user_id']
            intact=integrity['artifact_matches'] and integrity['manifest_matches']
            events=[dict(row) for row in connection.execute('''SELECT e.*,v.number AS revision_number
                FROM collection_events e JOIN collection_revisions v ON v.id=e.revision_id
                WHERE v.collection_id=? ORDER BY e.id''',(identifier,))]
            revisions=[dict(id=row['id'],number=row['number'],state=row['status'],fingerprint=row['fingerprint'],
                approved_at=row['approved_at']) for row in connection.execute(
                'SELECT * FROM collection_revisions WHERE collection_id=? ORDER BY number',(identifier,))]
            return dict(registered=True,state=revision['status'],revision_id=revision['id'],revision_number=revision['number'],
                fingerprint=revision['fingerprint'],manifest=json.loads(revision['manifest_json']),revisions=revisions,
                owner=dict(id=revision['owner_id'],username=revision['owner_username']),events=events,**integrity,
                permissions=dict(register=False,note=owner or user['role']=='reviewer',
                    submit=owner and revision['status']=='draft' and intact,
                    decide=user['role']=='reviewer' and revision['status']=='submitted' and intact,
                    revise=owner and revision['status']!='submitted',export=integrity['approval_valid']))

    def register(self, user, identifier, comment, new_revision=False, expected_id=None, expected_fingerprint=None):
        require_role(user,'engineer')
        try: snapshot=self.benchmarks.snapshot(identifier)
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise HTTPException(409,f'Cannot snapshot collection: {error}') from error
        with self.db.transaction() as connection:
            previous=self.current(connection,identifier)
            if previous:
                if previous['owner_id']!=user['user_id']: raise HTTPException(403,'Only the owner can revise the collection')
                if not new_revision or previous['status']=='submitted': raise HTTPException(409,'Create a new draft after the current review decision')
                if previous['id']!=expected_id or previous['fingerprint']!=expected_fingerprint: raise HTTPException(409,'Collection revision/fingerprint is stale')
            elif new_revision: raise HTTPException(409,'Register the collection first')
            self.write_manifest(snapshot)
            now=utc_now()
            if not previous:
                connection.execute('INSERT INTO collections(collection_id,owner_id,created_at) VALUES(?,?,?)',(identifier,user['user_id'],now))
            cursor=connection.execute('''INSERT INTO collection_revisions(collection_id,number,fingerprint,manifest_json,status,created_at)
                VALUES(?,?,?,?,?,?)''',(identifier,previous['number']+1 if previous else 1,snapshot.fingerprint,canonical(snapshot.manifest),'draft',now))
            connection.execute('UPDATE collections SET current_revision_id=? WHERE collection_id=?',(cursor.lastrowid,identifier))
            self.event(connection,self.current(connection,identifier),user,'revise' if previous else 'register','draft',comment,
                       previous['status'] if previous else None)
        return self.describe(user,identifier,snapshot)

    def note(self, user, identifier, comment):
        with self.db.transaction() as connection:
            self.require_view(user,identifier,connection)
            revision=self.current(connection,identifier)
            if not revision: raise HTTPException(409,'Register the collection first')
            if user['role']=='engineer' and revision['owner_id']!=user['user_id']: raise HTTPException(403,'Only the collection owner can write engineer notes')
            self.event(connection,revision,user,'note',revision['status'],comment)
        return self.describe(user,identifier)

    def transition(self, user, identifier, target, comment, revision_id, fingerprint):
        require_role(user,'engineer' if target=='submitted' else 'reviewer')
        try: snapshot=self.benchmarks.snapshot(identifier)
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise HTTPException(409,f'Collection integrity failed: {error}') from error
        with self.db.transaction() as connection:
            revision=self.current(connection,identifier)
            if not revision or revision['id']!=revision_id or revision['fingerprint']!=fingerprint: raise HTTPException(409,'Collection revision/fingerprint is stale')
            if target=='submitted':
                if revision['owner_id']!=user['user_id']: raise HTTPException(403,'Only the owner may submit this collection')
                if revision['status']!='draft': raise HTTPException(409,'Only draft can be submitted')
            elif revision['status']!='submitted': raise HTTPException(409,'Only submitted can receive a decision')
            integrity=self.integrity(revision,snapshot)
            if not integrity['artifact_matches'] or not integrity['manifest_matches']: raise HTTPException(409,dict(reason='artifact_changed',**integrity))
            now=utc_now()
            if target=='submitted': connection.execute('UPDATE collection_revisions SET status=?,submitted_at=? WHERE id=?',(target,now,revision_id))
            else: connection.execute('UPDATE collection_revisions SET status=?,approved_by=?,approved_at=? WHERE id=?',
                (target,user['user_id'] if target=='approved' else None,now if target=='approved' else None,revision_id))
            self.event(connection,revision,user,'submit' if target=='submitted' else 'decision',target,comment)
        return self.describe(user,identifier,snapshot)

    def export(self, user, identifier):
        self.require_view(user,identifier)
        with self.db.connect() as connection:
            revision=self.current(connection,identifier)
            if not revision or revision['status']!='approved': raise HTTPException(409,'Collection revision must be approved before export')
        started=time.perf_counter()
        logger=logging.getLogger('uvicorn.error')
        logger.info('Collection %s export: validating saved artifacts',identifier)
        try: snapshot=self.benchmarks.snapshot(identifier)
        except (OSError,ValueError,KeyError,TypeError) as error:
            raise HTTPException(409,f'Collection changed before export: {error}') from error
        if not self.integrity(revision,snapshot)['approval_valid']: raise HTTPException(409,'Collection approval does not match current artifacts/sidecar')
        validation_s=time.perf_counter()-started
        logger.info('Collection %s export: validation complete in %.3fs',identifier,validation_s)
        directory=self.settings.state_directory/'collection-exports';directory.mkdir(parents=True,exist_ok=True)
        output=directory/(secrets.token_hex(16)+'.zip')
        try:
            # Bounded memory: each file's checked bytes are immediately put into
            # the private ZIP. Never reopen a file between hash and zip write.
            with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
                for name,path in snapshot.paths.items():
                    allowed=next((root for root in self.benchmarks.roots if path.is_relative_to(root)),None)
                    if allowed is None: raise HTTPException(409,'Export source is outside configured benchmark roots')
                    content=contained(path,allowed).read_bytes();expected=snapshot.manifest['artifacts'][name]
                    if len(content)!=expected['bytes'] or digest(content)!=expected['sha256']:
                        raise HTTPException(409,f'Collection artifact changed while exporting: {name}')
                    archive.writestr(name,content)
                logger.info('Collection %s export: %d verified source files packed in %.3fs',identifier,len(snapshot.paths),time.perf_counter()-started)
                with self.db.transaction() as connection:
                    current=self.current(connection,identifier)
                    if current['id']!=revision['id'] or current['status']!='approved' or current['fingerprint']!=snapshot.fingerprint:
                        raise HTTPException(409,'Collection revision changed during export')
                    if not self.integrity(current,snapshot)['manifest_matches']: raise HTTPException(409,'Collection sidecar changed during export')
                    self.event(connection,current,user,'export','approved','Export exact approved collection; partial coverage retained')
                    events=[dict(row) for row in connection.execute('''SELECT e.*,v.number AS revision_number
                        FROM collection_events e JOIN collection_revisions v ON v.id=e.revision_id
                        WHERE v.collection_id=? ORDER BY e.id''',(identifier,))]
                    review=dict(kind='benchmark_collection_review',collection_id=identifier,revision_number=current['number'],
                        fingerprint=snapshot.fingerprint,state='approved',approved_by=current['approved_by'],approved_at=current['approved_at'],
                        exported_at=utc_now(),exported_by=user['username'],events=events,
                        status=snapshot.detail['status'],expected_frame_count=snapshot.detail['expected_frame_count'],
                        successful_frame_count=snapshot.detail['successful_frame_count'],metrics=snapshot.detail['metrics'],
                        metric_unit=snapshot.detail.get('metric_unit','ratio_0_1'),metric_source=snapshot.detail.get('metric_source_export','benchmark/metrics_pooled.json'),
                        export_timings_s=dict(validation=validation_s,source_read_hash_zip=time.perf_counter()-started-validation_s),
                        latency_source=snapshot.detail.get('latency_source_export','benchmark/initial_report.json'),
                        playback_note=snapshot.detail['playback_note'])
                    if snapshot.detail.get('task')=='detection':
                        review.update(kind='detection_benchmark_collection_review',task='detection',
                            protocol_fingerprint=snapshot.manifest['protocol_fingerprint'])
                    if snapshot.detail.get('task')=='camera':
                        review.update(kind='camera_frame_collection_review',task='camera',
                            projection_protocol=snapshot.manifest['projection_protocol'],
                            source_images_included=False,metric_unit=None,metrics=None)
                    archive.writestr('manifest.json',canonical(snapshot.manifest)+'\n')
                    archive.writestr('review.json',canonical(review)+'\n')
                    archive.close()  # ZIP close failure rolls back the export event.
            logger.info('Collection %s export: ZIP ready in %.3fs',identifier,time.perf_counter()-started)
            return output,snapshot.fingerprint
        except BaseException:
            output.unlink(missing_ok=True)
            raise
