from typing import Literal
import sys
import hashlib

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field

from .auth import Identity, SESSION_COOKIE, require_role
from .config import Settings
from .database import Database
from .review import Reviews
from .store import InvalidArtifact, OutsideRoot, RunStore
from .benchmarks import BenchmarkStore
from .collections import CollectionReviews
from .jobs import Jobs
from .detections import DetectionStore


class LoginBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


class CommentBody(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    comment: str = Field(min_length=1, max_length=4000)


class TransitionBody(CommentBody):
    revision_id: int = Field(gt=0)
    fingerprint: str = Field(pattern=r'^[0-9a-f]{64}$')


class DecisionBody(TransitionBody):
    decision: Literal['approved', 'rejected', 'needs_changes']


class EvaluationBody(TransitionBody):
    evaluation_id: str = Field(pattern=r'^[0-9a-f]{64}$')


class JobBody(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    profile_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    scene_name: str = Field(pattern=r'^scene-[0-9]{4}$')
    frame_index: int = Field(ge=0,strict=True)
    request_key: str = Field(pattern=r'^[0-9a-f]{32}$')


def create_app(settings=None):
    settings = settings or Settings.load()
    app = FastAPI(title='RAV25 inference review', version='2.0.0')
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    store = RunStore(settings)
    database = Database(settings.state_directory)
    identity = Identity(database, settings)
    reviews = Reviews(database, settings, store)
    benchmarks = BenchmarkStore(settings, store)
    collections = CollectionReviews(database, settings, benchmarks)
    jobs=Jobs(database,settings)
    detections=DetectionStore(settings,store)
    app.state.store = store
    app.state.database, app.state.identity, app.state.reviews = database, identity, reviews
    app.state.benchmarks, app.state.collections = benchmarks, collections
    app.state.jobs=jobs
    app.state.detections=detections

    @app.middleware('http')
    async def authentication_and_csrf(request: Request, call_next):
        if not request.url.path.startswith('/api/'):
            return await call_next(request)
        session = identity.authenticate(request.cookies.get(SESSION_COOKIE))
        request.state.auth = session
        public = request.url.path in ('/api/health', '/api/auth/session', '/api/auth/login')
        if not session and not public:
            return JSONResponse(status_code=401, content={'detail': 'Authentication required'},
                                headers={'Cache-Control': 'no-store'})
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            try:
                identity.check_csrf(request, session)
            except HTTPException as error:
                return JSONResponse(status_code=error.status_code, content={'detail': error.detail},
                                    headers={'Cache-Control': 'no-store'})
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers.add_vary_header('Cookie')
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.exception_handler(OutsideRoot)
    def outside_root(_request: Request, error: OutsideRoot):
        return JSONResponse(status_code=403, content={'detail': str(error)})

    @app.exception_handler(FileNotFoundError)
    def not_found(_request: Request, error: FileNotFoundError):
        return JSONResponse(status_code=404, content={'detail': str(error)})

    @app.exception_handler(InvalidArtifact)
    def invalid(_request: Request, error: InvalidArtifact):
        return JSONResponse(status_code=422, content={'detail': str(error)})

    @app.exception_handler(KeyError)
    def missing_key(_request: Request, error: KeyError):
        return JSONResponse(status_code=422, content={'detail': f'Missing artifact field: {error}'})

    @app.exception_handler(ValueError)
    def invalid_value(_request: Request, error: ValueError):
        return JSONResponse(status_code=422, content={'detail': f'Invalid saved data: {error}'})

    @app.get('/api/health')
    def health():
        return dict(status='ok', mode='saved-artifact-replay', inference_enabled=False,
                    inference_scope='Web process only; GPU jobs execute in an external CLI worker',job_submission_enabled=True,
                    workspace_id=hashlib.sha256(str(settings.state_directory).encode()).hexdigest()[:16],
                    model_modules_loaded=any(name.split('.')[0] in ('torch','mmcv','mmdet3d','occfusion') for name in sys.modules),
                    cache_hits=store.hits, cache_misses=store.misses)

    @app.get('/api/auth/session')
    def session(request: Request, response: Response):
        return identity.bootstrap(request, response, request.state.auth)

    @app.post('/api/auth/login')
    def login(body: LoginBody, request: Request, response: Response):
        return identity.login(body.username, body.password, request, response)

    @app.post('/api/auth/logout')
    def logout(request: Request, response: Response):
        identity.logout(request, response)
        return dict(ok=True)

    @app.get('/api/runs')
    def runs(request: Request):
        user = request.state.auth
        items = store.list_runs()
        with database.connect() as connection:
            if user['role'] == 'reviewer':
                items = [item for item in items if reviews.has_submission(connection, item['run_id'])]
            for item in items:
                revision = reviews.current(connection, item['run_id'])
                item['review_state'] = revision['status'] if revision else 'unregistered'
        return dict(runs=items)

    def visible(request, run_id):
        store.folder(run_id)
        reviews.require_view(request.state.auth, run_id)

    @app.get('/api/runs/{run_id}')
    def run(run_id: str, request: Request):
        visible(request, run_id)
        detail = store.get(run_id).detail
        evaluation = dict(status='unavailable', reason='Run không có prediction thành công.', matched=None, candidates=[])
        if detail['prediction_available']:
            evaluation = reviews.evaluations.describe(run_id, reviews.read_files(run_id))
        return dict(detail, evaluation=evaluation, review=reviews.describe(request.state.auth, run_id))

    def payload(request, run_id, kind, media_type):
        visible(request, run_id)
        value = getattr(store.get(run_id), kind)
        if value is None:
            raise HTTPException(409, 'This run has no saved prediction; check-only/failed runs are not rendered')
        return Response(content=value, media_type=media_type,
                        headers={'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff'})

    @app.get('/api/runs/{run_id}/occupancy')
    def occupancy(run_id: str, request: Request):
        return payload(request, run_id, 'occupancy', 'application/octet-stream')

    @app.get('/api/runs/{run_id}/lidar')
    def lidar(run_id: str, request: Request):
        return payload(request, run_id, 'lidar', 'application/octet-stream')

    @app.get('/api/runs/{run_id}/bev')
    def bev(run_id: str, request: Request):
        return payload(request, run_id, 'bev', 'image/png')

    @app.get('/api/runs/{run_id}/review')
    def review(run_id: str, request: Request):
        visible(request, run_id)
        return reviews.describe(request.state.auth, run_id)

    @app.post('/api/runs/{run_id}/register')
    def register(run_id: str, body: CommentBody, request: Request):
        return reviews.register(request.state.auth, run_id, body.comment)

    @app.post('/api/runs/{run_id}/review/revise')
    def revise(run_id: str, body: CommentBody, request: Request):
        return reviews.register(request.state.auth, run_id, body.comment, new_revision=True)

    @app.post('/api/runs/{run_id}/review/notes')
    def note(run_id: str, body: CommentBody, request: Request):
        return reviews.note(request.state.auth, run_id, body.comment)

    @app.post('/api/runs/{run_id}/review/evaluation')
    def attach_evaluation(run_id: str, body: EvaluationBody, request: Request):
        return reviews.register(request.state.auth, run_id, body.comment, new_revision=True,
                                evaluation_id=body.evaluation_id, expected_revision_id=body.revision_id,
                                expected_fingerprint=body.fingerprint)

    @app.post('/api/runs/{run_id}/review/submit')
    def submit(run_id: str, body: TransitionBody, request: Request):
        return reviews.transition(request.state.auth, run_id, 'submitted', body.comment, body.revision_id, body.fingerprint)

    @app.post('/api/runs/{run_id}/review/decision')
    def decision(run_id: str, body: DecisionBody, request: Request):
        require_role(request.state.auth, 'reviewer')
        return reviews.transition(request.state.auth, run_id, body.decision, body.comment, body.revision_id, body.fingerprint)

    @app.post('/api/runs/{run_id}/export')
    def export(run_id: str, request: Request):
        content, fingerprint = reviews.export(request.state.auth, run_id)
        return Response(content, media_type='application/zip', headers={
            'Content-Disposition': f'attachment; filename="{run_id}-approved-{fingerprint[:12]}.zip"',
            'Cache-Control': 'no-store',
        })

    @app.get('/api/benchmarks')
    def benchmark_list(request: Request):
        items=benchmarks.list_benchmarks()
        with database.connect() as connection:
            if request.state.auth['role']=='reviewer':
                items=[item for item in items if collections.has_submission(connection,item['benchmark_id'])]
        return dict(benchmarks=items,replay=True,inference_enabled=False)

    def collection_visible(request, identifier):
        benchmarks.folder(identifier)
        collections.require_view(request.state.auth,identifier)

    @app.get('/api/benchmarks/{benchmark_id}')
    def benchmark_detail(benchmark_id: str, request: Request):
        collection_visible(request,benchmark_id)
        snapshot=benchmarks.snapshot(benchmark_id)
        return dict(snapshot.detail,review=collections.describe(request.state.auth,benchmark_id,snapshot))

    @app.get('/api/benchmarks/{benchmark_id}/frames/{sample_token}')
    def benchmark_frame(benchmark_id: str, sample_token: str, request: Request):
        collection_visible(request,benchmark_id)
        return benchmarks.frame(benchmark_id,sample_token)[0]

    @app.get('/api/benchmarks/{benchmark_id}/frames/{sample_token}/{kind}')
    def benchmark_payload(benchmark_id: str, sample_token: str, kind: Literal['occupancy','lidar','bev'], request: Request, bundle: str | None = None):
        collection_visible(request,benchmark_id)
        detail,loaded=benchmarks.frame(benchmark_id,sample_token)
        if not detail['prediction_available']: raise HTTPException(409,'Failed frame has no saved prediction; inference is disabled')
        if bundle is not None and bundle!=detail['bundle_fingerprint']: raise HTTPException(409,'Frame bundle changed; reload the saved frame')
        return Response(getattr(loaded,kind),media_type='image/png' if kind=='bev' else 'application/octet-stream')

    @app.get('/api/collections/{collection_id}/review')
    def collection_review(collection_id: str, request: Request):
        collection_visible(request,collection_id)
        return collections.describe(request.state.auth,collection_id)

    @app.post('/api/collections/{collection_id}/register')
    def collection_register(collection_id: str, body: CommentBody, request: Request):
        return collections.register(request.state.auth,collection_id,body.comment)

    @app.post('/api/collections/{collection_id}/review/revise')
    def collection_revise(collection_id: str, body: TransitionBody, request: Request):
        return collections.register(request.state.auth,collection_id,body.comment,True,body.revision_id,body.fingerprint)

    @app.post('/api/collections/{collection_id}/review/notes')
    def collection_note(collection_id: str, body: CommentBody, request: Request):
        return collections.note(request.state.auth,collection_id,body.comment)

    @app.post('/api/collections/{collection_id}/review/submit')
    def collection_submit(collection_id: str, body: TransitionBody, request: Request):
        return collections.transition(request.state.auth,collection_id,'submitted',body.comment,body.revision_id,body.fingerprint)

    @app.post('/api/collections/{collection_id}/review/decision')
    def collection_decision(collection_id: str, body: DecisionBody, request: Request):
        require_role(request.state.auth,'reviewer')
        return collections.transition(request.state.auth,collection_id,body.decision,body.comment,body.revision_id,body.fingerprint)

    @app.post('/api/collections/{collection_id}/export')
    def collection_export(collection_id: str, request: Request):
        path,fingerprint=collections.export(request.state.auth,collection_id)
        return FileResponse(path,media_type='application/zip',
            filename=f'{collection_id}-collection-approved-{fingerprint[:12]}.zip',
            background=BackgroundTask(path.unlink,missing_ok=True))

    @app.get('/api/jobs/options')
    def job_options(request: Request):
        require_role(request.state.auth,'engineer')
        profiles=jobs.profiles.options()
        return dict(profiles=profiles,scenes_by_profile={p['id']:jobs.profiles.scenes(p['id']) for p in profiles})

    @app.get('/api/detections')
    def detection_list(request: Request):
        values=detections.list()
        if request.state.auth['role']=='reviewer':
            with database.connect() as c:values=[v for v in values if reviews.has_submission(c,v['run_id'])]
        return dict(runs=values)

    @app.get('/api/detections/{run_id}')
    def detection_detail(run_id: str,request: Request,occupancy_run_id: str | None=None):
        reviews.require_view(request.state.auth,run_id)
        occupancy=None
        if occupancy_run_id is not None:
            visible(request,occupancy_run_id)
            occupancy=__import__('json').loads(store.artifact(store.folder(occupancy_run_id),'metadata.json').read_bytes())
        return dict(detections.get(run_id,occupancy),review=reviews.describe(request.state.auth,run_id))

    @app.get('/api/detections/{run_id}/lidar')
    def detection_lidar(run_id: str,request: Request):
        reviews.require_view(request.state.auth,run_id)
        folder=store.folder(run_id);detections.get(run_id)
        meta=__import__('json').loads((folder/'metadata.json').read_bytes())
        data,_=store.catalog.lidar(meta,meta['bounds'])
        return Response(data.tobytes(),media_type='application/octet-stream')

    @app.get('/api/runs/{run_id}/detections')
    def detection_matches(run_id: str,request: Request):
        visible(request,run_id)
        meta=__import__('json').loads(store.artifact(store.folder(run_id),'metadata.json').read_bytes())
        matches=[];rejected=[]
        for item in detections.list():
            if item['sample_token']!=meta['sample_token']:continue
            try:
                reviews.require_view(request.state.auth,item['run_id'])
                matches.append(detections.get(item['run_id'],meta))
            except HTTPException as e:
                if e.status_code!=403:raise
            except (ValueError,KeyError,OSError) as e:rejected.append(dict(run_id=item['run_id'],status='invalid',reason=str(e)))
        return dict(matches=matches,rejected=rejected,scope='Detection snapshots reviewed/exported separately from occupancy')

    @app.get('/api/jobs')
    def job_list(request: Request):
        return jobs.list(request.state.auth)

    @app.post('/api/jobs',status_code=202)
    def job_create(body: JobBody,request: Request):
        return jobs.create(request.state.auth,body.profile_id,body.scene_name,body.frame_index,body.request_key)

    @app.get('/api/jobs/{job_id}')
    def job_detail(job_id: str,request: Request):
        return jobs.describe(request.state.auth,job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def job_cancel(job_id: str,request: Request):
        return jobs.cancel(request.state.auth,job_id)

    @app.get('/api/jobs/{job_id}/log')
    def job_log(job_id: str,request: Request,kind: Literal['inference','evaluation']='inference',offset: int=0):
        if offset<0:raise HTTPException(422,'Log offset must be nonnegative')
        return jobs.log(request.state.auth,job_id,kind,offset)

    return app


app = create_app()
