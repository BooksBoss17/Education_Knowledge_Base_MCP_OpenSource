"""Windows managed CLI smoke with synthetic inputs and a loopback model fixture."""
from pathlib import Path
from io import BytesIO
import importlib.util,json,os,runpy,subprocess,sys,tempfile,threading,time,uuid,zipfile
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
                jobs = []
                def rpc(method, **params):
                    request = {'jsonrpc':'2.0','id':1,'method':method,'params':params}
                    proc.stdin.write(json.dumps(request)+'\n')
                    proc.stdin.flush()
                    response = json.loads(proc.stdout.readline())['result']
                    assert not response.get('isError'), response
                    return response

                def run_action(action, task, **fields):
                    result = rpc('tools/call',name='runtime_jobs',arguments=dict(action=action,task_run_id=task,**fields))
                    return json.loads(result['content'][0]['text'])

                def finish(task, outcome):
                    result = run_action('end',task,outcome=outcome)
                    deadline = time.monotonic()+15
                    while result['state'] == 'CLOSING':
                        assert time.monotonic()<deadline
                        time.sleep(.25)
                        result = run_action('inspect',task)
                    return result

                try:
                    rpc('initialize',protocolVersion='2025-11-25',capabilities={},clientInfo={'name':'public-smoke','version':'1'})
                    tools = {t['name'] for t in rpc('tools/list')['tools']}
                    assert {'bemarkdown_convert','runtime_jobs','runtime_monitor','textbook_import'} <= tools
                    a, b = uuid.uuid4().hex, uuid.uuid4().hex
                    for task in (a,b):
                        jobs.append(task)
                        assert run_action('begin',task)['state'] == 'OPEN'
                    capture = run_action('inspect',a)['capture_id']
                    assert capture == run_action('inspect',b)['capture_id']
                    rpc('tools/call',name='knowledge_workspace',arguments={'action':'inspect','task_run_id':a})
                    first = finish(a,'complete')
                    assert first['state']=='COMPLETED' and not first['receipt']['monitor']['released_here']
                    second = finish(b,'pause')
                    assert second['state']=='PAUSED' and second['receipt']['monitor']['released_here']
                    assert run_action('resume',b)['capture_id'] != capture
                    assert finish(b,'complete')['state']=='COMPLETED'
                finally:
                    try:
                        for task in jobs:
                            if run_action('inspect',task)['state']=='OPEN':
                                finish(task,'cancel')
                    finally:
                        proc.stdin.close()
                        assert proc.wait(timeout=30)==0, proc.stderr.read()
                print(json.dumps({'managed_inputs':4,'mcp_initialize_and_tools':True,'shared_task_monitor_and_closeout':True,
                                  'local_gpu_required':False,'model_fixture_only':True}))
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
