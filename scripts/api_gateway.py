"""Explicit loopback bridge to configured local or HTTPS model endpoints."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError

LIMIT=64*1024*1024
SCHEMA='bemarkdown-local-model-api-v1'
PROMPTS={
    'formula':'Transcribe every visible formula and accompanying text. Use LaTeX; wrap math in $ delimiters when mixed with prose. Do not correct or invent content.',
    'text':'Transcribe all visible text and mathematical expressions in reading order. Preserve math with paired $ delimiters. Do not fill blank or masked areas. Output only the transcription.',
    'table':'Transcribe the entire table as one HTML table with explicit closing tags. Preserve empty cells, rowspan, colspan, and math in paired $ delimiters. No explanation or code fences.'}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('UPSTREAM_REDIRECT_FORBIDDEN')


def endpoint(value):
    u=urlsplit(value)
    if u.username or u.password or u.query or u.fragment or not u.hostname:
        raise ValueError('Endpoint must not contain credentials, query or fragment')
    if u.scheme!='https' and not (u.scheme=='http' and u.hostname in {'127.0.0.1','::1'}):
        raise ValueError('Remote endpoints require HTTPS; HTTP is loopback-only')
    return value


def post(spec,payload=None):
    headers={'Content-Type':'application/json'}
    if spec.get('key_env'):headers['Authorization']='Bearer '+os.environ[spec['key_env']]
    try:
        body = json.dumps(payload).encode() if payload is not None else None
        with build_opener(NoRedirect()).open(Request(endpoint(spec['url']),body,headers),timeout=spec.get('timeout_seconds',120)) as response:
            raw=response.read(LIMIT+1)
        if len(raw)>LIMIT:raise ValueError('UPSTREAM_RESPONSE_TOO_LARGE')
        return json.loads(raw)
    except HTTPError as exc:
        # Providers may echo credentials/request bodies; never expose that body.
        raise ValueError('UPSTREAM_HTTP_'+str(exc.code)) from None


class Gateway:
    def __init__(self,config):
        self.roles=config['roles']
        self.session_roles = {}
        self.sessions = {}
        self.closed_sessions = {}
        self._capability = None
        if set(self.roles)!={'layout','formula','table','text'}:raise ValueError('All four roles are required')
        for role,spec in self.roles.items():
            endpoint(spec['url'])
            if spec['protocol'] not in {'bemarkdown','chat-completions'}:raise ValueError('Unknown protocol')
            if spec['protocol']=='chat-completions':
                if role=='layout':raise ValueError('Layout requires the BeMarkdown region protocol, not free-form chat')
                if not spec.get('model'):raise ValueError('Model is required')
            if spec.get('key_env') and not os.environ.get(spec['key_env']):raise ValueError('Missing configured key environment variable')

    def capabilities(self):
        if self._capability is not None:
            return self._capability
        from bemarkdown.model_session import SESSION_SCHEMA
        max_items, max_bytes = 128, LIMIT
        for role, spec in self.roles.items():
            suffix = '/v1/' + role
            if spec['protocol'] != 'bemarkdown' or not spec['url'].endswith(suffix):
                continue
            base = spec['url'][:-len(suffix)]
            try:
                capability = post({**spec, 'url': base + '/v1/capabilities'})
            except ValueError as error:
                if str(error) in {'UPSTREAM_HTTP_404', 'UPSTREAM_HTTP_405', 'UPSTREAM_HTTP_501'}:
                    continue
                raise
            if capability.get('schema') != 'bemarkdown-model-api-capabilities-v1':
                raise ValueError('UPSTREAM_CAPABILITIES_INVALID')
            if capability.get('session_schema') != SESSION_SCHEMA:
                continue
            if role not in capability.get('session_roles', capability.get('roles', [])):
                continue
            count, size = capability.get('max_items'), capability.get('max_bytes')
            if type(count) is not int or not 1 <= count <= 128 or type(size) is not int or not 1024 <= size <= LIMIT:
                raise ValueError('UPSTREAM_CAPABILITY_LIMIT_INVALID')
            self.session_roles[role] = base
            max_items, max_bytes = min(max_items, count), min(max_bytes, size)
        self._capability = {'schema':'bemarkdown-model-api-capabilities-v1','roles':list(self.roles),
                            'session_schema':SESSION_SCHEMA, 'session_roles':list(self.session_roles),
                            'max_items':max_items, 'max_bytes':max_bytes, 'layout_batch_items':min(8,max_items)}
        return self._capability

    def session_action(self, action, payload):
        from bemarkdown.model_session import SESSION_SCHEMA
        self.capabilities()
        session_id = payload.get('session_id')
        if payload.get('schema') != SESSION_SCHEMA or not isinstance(session_id, str) or not re.fullmatch(r'[a-f0-9]{32}', session_id):
            raise ValueError('GATEWAY_SESSION_BINDING_INVALID')
        if action not in {'open','infer','status','close'}:
            raise ValueError('GATEWAY_SESSION_ACTION_INVALID')
        if session_id in self.closed_sessions:
            if action == 'close':
                return dict(self.closed_sessions[session_id]['receipt'])
            if action == 'open':
                raise ValueError('GATEWAY_SESSION_ALREADY_CLOSED')
        if action == 'open':
            if self.sessions and session_id not in self.sessions:
                raise ValueError('GATEWAY_ANOTHER_SESSION_ACTIVE')
            role = payload.get('role')
            if role not in self.session_roles:
                raise ValueError('UPSTREAM_ROLE_HAS_NO_SESSION_SUPPORT')
            if session_id in self.sessions and self.sessions[session_id] != role:
                raise ValueError('GATEWAY_SESSION_ROLE_CHANGED')
            self.sessions[session_id] = role
        role = self.sessions.get(session_id) or self.closed_sessions.get(session_id, {}).get('role')
        if role is None:
            raise ValueError('GATEWAY_SESSION_UNKNOWN')
        if action == 'infer' and payload.get('role') != role:
            raise ValueError('GATEWAY_SESSION_ROLE_CHANGED')
        spec = {**self.roles[role], 'url': self.session_roles[role] + '/v2/session/' + action}
        result = post(spec, payload)
        if result.get('schema') != SESSION_SCHEMA or result.get('session_id') != session_id:
            raise ValueError('UPSTREAM_SESSION_BINDING_INVALID')
        if action == 'close' and result.get('model_unloaded') is True:
            self.sessions.pop(session_id, None)
            self.closed_sessions[session_id] = {'role':role, 'receipt':result}
            while len(self.closed_sessions) > 64:
                del self.closed_sessions[next(iter(self.closed_sessions))]
        return result

    def close(self):
        failures = []
        from bemarkdown.model_session import SESSION_SCHEMA
        for session_id in list(self.sessions):
            try:
                self.session_action('close', {'schema':SESSION_SCHEMA,'session_id':session_id})
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('UPSTREAM_SESSION_CLOSE_UNCONFIRMED: ' + '; '.join(failures))

    def __call__(self,payload):
        from bemarkdown.pdf.local_model_api import decoded_image
        role=payload['role'];spec=self.roles[role]
        for item in payload['items']:decoded_image(item)
        if spec['protocol']=='bemarkdown':return post(spec,payload)
        items=[]
        for item in payload['items']:
            request={'model':spec['model'],'messages':[{'role':'user','content':[
                {'type':'text','text':spec.get('prompt',PROMPTS[role])},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+item['image_base64']}}]}],
                'max_tokens':spec.get('max_tokens',4096),'temperature':0}
            result=post(spec,request)
            choice=result['choices'][0]
            if choice.get('finish_reason') not in {None,'stop'}:raise ValueError('UPSTREAM_INCOMPLETE_OUTPUT')
            value=choice['message']['content']
            if not isinstance(value,str):raise ValueError('UPSTREAM_TEXT_REQUIRED')
            items.append({'id':item['id'],'text':value})
        # Config identity is explicitly not a checksum of inaccessible remote weights.
        fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        return {'schema':SCHEMA,'role':role,'request_id':payload['request_id'],'released':True,
                'identity':{'model_id':spec['model'],'model_fingerprint':fingerprint,
                    'fingerprint_scope':'DECLARED_REMOTE_CONFIGURATION_NOT_WEIGHTS'},'items':items}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--port',type=int,default=8765);p.add_argument('--token-env',default='BEMARKDOWN_GATEWAY_TOKEN')
    a=p.parse_args()
    from bemarkdown.local_model_server import create_server
    token=os.environ.get(a.token_env,'')
    if len(token)<16:p.error('Set a gateway token of at least 16 characters in '+a.token_env)
    server=create_server(('127.0.0.1',a.port),Gateway(json.loads(a.config.read_text('utf8'))),token=token)
    print(json.dumps({'ready':True,'port':server.server_port}),flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
