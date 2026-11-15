"""Camera routes inherit parent run/frame visibility and existing CSRF middleware."""
from typing import Annotated
from fastapi import HTTPException, Request, Response, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field

from .auth import require_role


class PrivacyBody(BaseModel):
    model_config=ConfigDict(extra='forbid',str_strip_whitespace=True)
    camera_fingerprint:str=Field(pattern=r'^[0-9a-f]{64}$')
    previous_revision:str|None=Field(default=None,pattern=r'^[0-9a-f]{64}$')
    reviewed:bool=Field(strict=True)
    comment:str=Field(min_length=1,max_length=4000)
    masks:list[tuple[Annotated[int,Field(strict=True)],Annotated[int,Field(strict=True)],
        Annotated[int,Field(strict=True)],Annotated[int,Field(strict=True)]]]=Field(default_factory=list,max_length=64)


def register_camera_routes(app,cameras,reviews,visible,collection_visible,CommentBody,TransitionBody,DecisionBody):
    def context(request,identifier):
        token=request.path_params.get('sample_token')
        if token is None:visible(request,identifier)
        else:collection_visible(request,identifier)
        return cameras.identifier('run' if token is None else 'benchmark',identifier,token)

    def listing(identifier:str,request:Request):
        scope=context(request,identifier)
        value=cameras.listing(scope,request.url.path)
        with reviews.db.connect() as c:
            allowed=request.state.auth['role']=='engineer' or reviews.has_submission(c,scope)
        value['review']=reviews.describe(request.state.auth,scope) if allowed else dict(
            registered=False,state='not_submitted',permissions={},events=[])
        return value

    def image(identifier:str,channel:str,request:Request,version:str=Query(pattern=r'^[0-9a-f]{64}$')):
        scope=context(request,identifier)
        content,media=cameras.image(scope,channel,version,request.url.path.rsplit('/',2)[0])
        return Response(content,media_type=media,headers={'Content-Disposition':'inline'})

    def projection(identifier:str,channel:str,request:Request,version:str=Query(pattern=r'^[0-9a-f]{64}$')):
        scope=context(request,identifier)
        return cameras.project(scope,channel,version,request.url.path.rsplit('/',2)[0])

    def privacy(identifier:str,channel:str,body:PrivacyBody,request:Request):
        require_role(request.state.auth,'engineer')
        return cameras.review_image(context(request,identifier),channel,body,request.state.auth)

    def review(identifier:str,request:Request):
        return reviews.describe(request.state.auth,context(request,identifier))

    # Use concrete annotations (not postponed strings) for FastAPI's schema.
    def register(identifier:str,body:CommentBody,request:Request):
        return reviews.register(request.state.auth,context(request,identifier),body.comment)

    def revise(identifier:str,body:TransitionBody,request:Request):
        return reviews.register(request.state.auth,context(request,identifier),body.comment,True,body.revision_id,body.fingerprint)

    def note(identifier:str,body:CommentBody,request:Request):
        return reviews.note(request.state.auth,context(request,identifier),body.comment)

    def submit(identifier:str,body:TransitionBody,request:Request):
        return reviews.transition(request.state.auth,context(request,identifier),'submitted',body.comment,body.revision_id,body.fingerprint)

    def decision(identifier:str,body:DecisionBody,request:Request):
        require_role(request.state.auth,'reviewer')
        return reviews.transition(request.state.auth,context(request,identifier),body.decision,body.comment,body.revision_id,body.fingerprint)

    def export(identifier:str,request:Request):
        scope=context(request,identifier);path,fingerprint=reviews.export(request.state.auth,scope)
        return FileResponse(path,media_type='application/zip',filename=scope+'-'+fingerprint[:12]+'.zip',
            background=BackgroundTask(path.unlink,missing_ok=True))

    for prefix in ('/api/runs/{identifier}/cameras','/api/benchmarks/{identifier}/frames/{sample_token}/cameras'):
        for method,suffix,handler in (
            ('GET','',listing),('GET','/review',review),('POST','/register',register),
            ('POST','/review/revise',revise),('POST','/review/notes',note),
            ('POST','/review/submit',submit),('POST','/review/decision',decision),
            ('POST','/export',export),('GET','/{channel}/image',image),
            ('GET','/{channel}/projection',projection),('POST','/{channel}/privacy',privacy)):
            app.add_api_route(prefix+suffix,handler,methods=[method],
                name='camera_'+handler.__name__+('_run' if prefix.startswith('/api/runs') else '_frame'))
