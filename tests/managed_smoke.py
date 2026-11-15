"""Windows managed CLI smoke with synthetic inputs and a loopback model fixture."""
from pathlib import Path
from io import BytesIO
import importlib.util,json,os,runpy,subprocess,sys,tempfile,threading,time,zipfile
from PIL import Image
import fitz
from bemarkdown.local_model_server import create_server
from bemarkdown.pdf.local_model_api import decoded_image,SCHEMA
ROOT=Path(__file__).resolve().parents[1]


def backend(payload):
    items=[]
    for item in payload['items']:
        with Image.open(BytesIO(decoded_image(item))) as im:w,h=im.size
        row={'id':item['id']}
        if payload['role']=='layout':row['regions']=[{'label':'text','bbox':[0,0,w,h],'reading_order':0}]
        else:row['text']='MANAGED_API_FIXTURE'
        items.append(row)
    return {**payload,'items':items,'released':True,'identity':{'model_id':'synthetic-fixture','model_fingerprint':'b'*64}}


def main():
    assert os.name=='nt','Managed MCP is Windows-only'
    server=create_server(('127.0.0.1',0),backend,token='managed-fixture-token')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config=ROOT/'.local/model-api.json';previous=config.read_bytes() if config.exists() else None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);env=os.environ.copy();env['EKB_RUN_MANAGER_HOME']=str(tmp/'manager');env['SMOKE_TOKEN']='managed-fixture-token'
            config.write_text(json.dumps({'schema':SCHEMA,'roles':{role:{'url':f'http://127.0.0.1:{server.server_port}/v1/{role}','token_env':'SMOKE_TOKEN'} for role in ['layout','formula','table','text']}}),'utf8')
            image=tmp/'image.png';Image.new('RGB',(160,80),'white').save(image)
            pdf=tmp/'page.pdf'
            with fitz.open() as document:
                page=document.new_page(width=160,height=80);page.insert_text((10,30),'Synthetic test');document.save(pdf)
            from docx import Document
            native=tmp/'native.docx';d=Document();d.add_paragraph('Native text preserved');d.save(native)
            screenshots=tmp/'screenshots.docx';d=Document();d.add_picture(str(image));d.save(screenshots)
            try:
                for source in [native,image,pdf,screenshots]:
                    command=[sys.executable,'-I','-X','utf8',str(ROOT/'TOOLS/bemarkdown/convert.py'),'--source',str(source),'--output-root',str(tmp/'out'),'--mcp-root',str(ROOT),'--json']
                    result=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf8',timeout=90)
                    if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
                    value=json.loads(result.stdout);assert value['success'],value
                # MCP transport initialization/listing needs no vision challenge.
                proc=subprocess.Popen([sys.executable,'-I','-X','utf8',str(ROOT/'TOOLS/education_mcp/launch.py'),'--mcp-root',str(ROOT),'--workspace',str(tmp/'workspace')],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf8')
                messages='\n'.join(json.dumps({'jsonrpc':'2.0','id':i,'method':method,'params':params}) for i,method,params in [(1,'initialize',{'protocolVersion':'2025-11-25','clientInfo':{'name':'public-smoke','version':'1'}}),(2,'tools/list',{})])+'\n'
                stdout,stderr=proc.communicate(messages,timeout=30);assert proc.returncode==0,stderr
                replies=[json.loads(line) for line in stdout.splitlines()];assert len(replies)==2 and all('result' in x for x in replies),replies
                names={t['name'] for t in replies[1]['result']['tools']};assert 'bemarkdown_convert' in names and not any(n.startswith('semantic_retention') for n in names)
                print(json.dumps({'managed_inputs':4,'mcp_initialize_and_tools':True,'local_gpu_required':False,'model_fixture_only':True}))
            finally:
                boot=runpy.run_path(str(ROOT/'TOOLS/runtime_manager/bootstrap.py'));_,_,Client,_,identity=boot['runtime'](ROOT)
                client=Client(tmp/'manager')
                try:current=client.ping()
                except Exception:pass
                else:
                    assert not client.list(states=['RUNNING','STARTING','QUEUED','WAITING_RESOURCES'])
                    client.shutdown()
                    deadline=time.monotonic()+15
                    while identity(current['identity']['pid'])==current['identity']:
                        assert time.monotonic()<deadline;time.sleep(.1)
    finally:
        if previous is None:config.unlink(missing_ok=True)
        else:config.write_bytes(previous)
        server.shutdown();server.server_close();thread.join()


if __name__=='__main__':main()
