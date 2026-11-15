import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,HTTPServer
from io import BytesIO

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gateway',ROOT/'scripts/api_gateway.py')
gateway=importlib.util.module_from_spec(spec);spec.loader.exec_module(gateway)


class ApiTest(unittest.TestCase):
    def test_remote_endpoint_boundary(self):
        for value in ['http://example.com/chat','https://user:key@example.com/chat','https://example.com/chat?key=secret']:
            with self.assertRaises(ValueError):gateway.endpoint(value)
        self.assertEqual(gateway.endpoint('https://example.com/chat'),'https://example.com/chat')

    def test_real_loopback_gateway_and_conversion(self):
        from PIL import Image
        from bemarkdown.local_model_server import create_server
        from bemarkdown.pdf.local_model_api import LocalModelApi,encoded_image,SCHEMA
        from bemarkdown.next_runtime import convert_document
        from bemarkdown.package_io import validate_package
        calls=[]
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path,self.headers.get('Authorization')))
                if self.path=='/layout':
                    data=base64.b64decode(payload['items'][0]['image_base64'])
                    with Image.open(BytesIO(data)) as image:w,h=image.size
                    result={**payload,'released':True,'identity':{'model_id':'fixture-layout','model_fingerprint':'a'*64},
                            'items':[{'id':payload['items'][0]['id'],'regions':[{'label':'text','bbox':[0,0,w,h],'reading_order':0}]}]}
                else:
                    assert payload['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
                    value={'formula':'x=1','table':'<table><tr><td>A</td></tr></table>','text':'PUBLIC_API_SMOKE'}[payload['model']]
                    result={'choices':[{'finish_reason':'stop','message':{'content':value}}]}
                data=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        upstream=HTTPServer(('127.0.0.1',0),Upstream);t=threading.Thread(target=upstream.serve_forever,daemon=True);t.start()
        os.environ['PUBLIC_TEST_KEY']='test-only-key'
        roles={'layout':{'protocol':'bemarkdown','url':f'http://127.0.0.1:{upstream.server_port}/layout'}}
        roles.update({role:{'protocol':'chat-completions','url':f'http://127.0.0.1:{upstream.server_port}/chat','model':role,'key_env':'PUBLIC_TEST_KEY'} for role in ['formula','table','text']})
        bridge=create_server(('127.0.0.1',0),gateway.Gateway({'roles':roles}),token='public-test-token-123456')
        bt=threading.Thread(target=bridge.serve_forever,daemon=True);bt.start()
        try:
            os.environ['PUBLIC_GATEWAY_TOKEN']='public-test-token-123456'
            config={'schema':SCHEMA,'roles':{role:{'url':f'http://127.0.0.1:{bridge.server_port}/v1/{role}','token_env':'PUBLIC_GATEWAY_TOKEN'} for role in roles}}
            image=Image.new('RGB',(100,50),'white');data=BytesIO();image.save(data,format='PNG')
            for role in roles:
                result=LocalModelApi(role,config['roles'][role]).request([{'id':'0',**encoded_image(data.getvalue())}])
                self.assertEqual(result[0]['id'],'0')
            with tempfile.TemporaryDirectory() as tmp:
                tmp=Path(tmp);source=tmp/'input.png';source.write_bytes(data.getvalue());c=tmp/'api.json';c.write_text(json.dumps(config),'utf8')
                converted=convert_document(source,tmp/'out',api_config=c)
                package=Path(converted.to_dict()['package_path'])
                self.assertTrue(validate_package(package).valid)
                self.assertIn('PUBLIC_API_SMOKE',(package/'document.md').read_text('utf8'))
            self.assertTrue(all(auth=='Bearer test-only-key' for path,auth in calls if path=='/chat'))
        finally:
            bridge.shutdown();bridge.server_close();bt.join();upstream.shutdown();upstream.server_close();t.join();os.environ.pop('PUBLIC_TEST_KEY',None)

    @unittest.skipUnless(os.name=='nt','Windows managed runtime')
    def test_external_policy_without_gpu(self):
        s=importlib.util.spec_from_file_location('cpu_policy',ROOT/'TOOLS/runtime_manager/external_resource_policy.py')
        p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
        sample=p.sample_device();self.assertEqual(sample['gpu_count'],0)
        policy=p.resolve_device_policy(sample['total_physical_bytes'],sample['commit_limit_bytes'],0)
        self.assertTrue(p.check_current(policy,sample)['allowed'])
        self.assertEqual(p.evaluate_admission(policy,sample,{'extra_private_bytes':0,'extra_physical_bytes':0,'extra_gpu_mib':1})['decision'],'CAPACITY_INSUFFICIENT')


if __name__=='__main__':unittest.main()
