import threading
import uuid
from io import BytesIO
from PIL import Image
from test_public_api import gateway
from bemarkdown.local_model_server import create_server
from bemarkdown.model_session import SESSION_SCHEMA
from bemarkdown.pdf.local_model_api import LocalModelApi, SCHEMA, encoded_image


def test_mixed_v2_v1_forwarding_and_idempotent_close(monkeypatch):
    events=[]
    class Upstream:
        def capabilities(self):
            return dict(schema='bemarkdown-model-api-capabilities-v1',roles=['layout','formula','table','text'],
                        session_schema=SESSION_SCHEMA,session_roles=['layout'],max_items=128,max_bytes=64*1024**2)
        def session_action(self,action,payload):
            events.append(action)
            value=dict(schema=SESSION_SCHEMA,session_id=payload['session_id'])
            if action=='open':return dict(value,role='layout',state='OPEN')
            if action=='close':return dict(value,state='CLOSED',model_unloaded=True)
            return dict(self(payload),**value,request_complete=True,model_unloaded=False,released=False)
        def __call__(self,payload):
            events.append('v1-'+payload['role'])
            return dict(schema=SCHEMA,role=payload['role'],request_id=payload['request_id'],released=True,
                        identity={'model_id':'fixture','model_fingerprint':'a'*64},items=[{'id':x['id'],'text':'fixture'} for x in payload['items']])
    upstream=create_server(('127.0.0.1',0),Upstream(),token='upstream-key')
    t=threading.Thread(target=upstream.serve_forever);t.start()
    monkeypatch.setenv('UPSTREAM_KEY','upstream-key');monkeypatch.setenv('GATEWAY_KEY','gateway-key')
    bridge=gateway.Gateway({'roles':{role:{'protocol':'bemarkdown','url':f'http://127.0.0.1:{upstream.server_port}/v1/{role}','key_env':'UPSTREAM_KEY'} for role in ('layout','formula','table','text')}})
    server=create_server(('127.0.0.1',0),bridge,token='gateway-key');st=threading.Thread(target=server.serve_forever);st.start()
    try:
        data=BytesIO();Image.new('RGB',(10,10),'white').save(data,format='PNG')
        item={'id':'0',**encoded_image(data.getvalue())}
        client=LocalModelApi('layout',{'url':f'http://127.0.0.1:{server.server_port}/v1/layout','token_env':'GATEWAY_KEY'})
        client.request([item]);client.request([item]);sid=client.session_id;client.close()
        bridge.session_action('close',dict(schema=SESSION_SCHEMA,session_id=sid))
        text=LocalModelApi('text',{'url':f'http://127.0.0.1:{server.server_port}/v1/text','token_env':'GATEWAY_KEY'})
        text.request([item]);text.close()
        assert events.count('open')==1 and events.count('infer')==2 and events.count('close')==1
        assert events.count('v1-text')==1 and not bridge.sessions
        assert bridge.capabilities()['session_roles']==['layout']
    finally:
        server.shutdown();server.server_close();st.join();upstream.shutdown();upstream.server_close();t.join()
