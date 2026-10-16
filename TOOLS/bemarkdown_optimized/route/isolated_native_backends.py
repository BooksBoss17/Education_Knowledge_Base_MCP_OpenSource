"""Owned subprocess boundaries for native heavy backends; unchanged model APIs."""
import argparse,atexit,json,os,pathlib,secrets,subprocess,sys,time,traceback
from multiprocessing.connection import Client,Listener

_active=None
_table_staged_active=False
_native_isolation_active=False

class ResultView:
    def __init__(self,value):self.json=value

def normalized_result(value):
    if isinstance(value,(str,int,float,bool,bytes,type(None))):return value
    if isinstance(value,dict):return {k:normalized_result(v) for k,v in value.items()}
    if isinstance(value,list):return [normalized_result(x) for x in value]
    if isinstance(value,tuple):return tuple(normalized_result(x) for x in value)
    if hasattr(value,'json'):
        data=value.json() if callable(value.json) else value.json
        return {'__isolated_result_json__':json.loads(data) if isinstance(data,str) else normalized_result(data)}
    # Public NumPy/PIL objects and BeMarkdown data classes remain losslessly pickled.
    return value

def decoded_result(value):
    if isinstance(value,dict):
        if set(value)=={'__isolated_result_json__'}:return ResultView(value['__isolated_result_json__'])
        return {k:decoded_result(v) for k,v in value.items()}
    if isinstance(value,list):return [decoded_result(x) for x in value]
    if isinstance(value,tuple):return tuple(decoded_result(x) for x in value)
    return value

class RecProxy:
    def __init__(self,owner):self.owner=owner
    def predict(self,*args,**kwargs):return self.owner._request('rec_predict',args,kwargs)

class RemoteBackend:
    def __init__(self,kind,args=(),kwargs=None):
        self.kind=kind;self.arguments=args;self.options=kwargs or {};self._process=None;self._connection=None
        self._work=None;self._calls=0;self._starts=0;self._cached={}
    def _ensure(self):
        global _active
        if self._process is not None and self._process.poll() is None:return
        if _active is not None and _active is not self:_active._stop()
        self._starts+=1
        root=pathlib.Path(os.environ['TEMP']).resolve()/'isolated-backends';root.mkdir(exist_ok=True)
        self._work=root/(self.kind+'-'+secrets.token_hex(5));self._work.mkdir()
        token=secrets.token_bytes(32);config=self._work/'config.json'
        config.write_text(json.dumps({'kind':self.kind,'args':[str(x) if isinstance(x,pathlib.Path) else x for x in self.arguments],
            'kwargs':{k:str(v) if isinstance(v,pathlib.Path) else v for k,v in self.options.items()},
            'auth':token.hex(),'snapshot':str(pathlib.Path(__file__).resolve().parent),'attributes':self._cached},ensure_ascii=False),encoding='utf-8')
        self._log=(self._work/'worker.log').open('wb')
        self._process=subprocess.Popen([sys.executable,'-X','utf8',__file__,'--actor',str(config)],
            stdout=self._log,stderr=self._log,creationflags=subprocess.CREATE_NO_WINDOW)
        ready=self._work/'ready.json';started=time.monotonic()
        while not ready.exists():
            if self._process.poll() is not None:raise RuntimeError(f'ISOLATED_BACKEND_START_FAILED:{self.kind}:{self._work}')
            if time.monotonic()-started>300:raise TimeoutError('ISOLATED_BACKEND_START_TIMEOUT')
            time.sleep(.02)
        address=json.loads(ready.read_text(encoding='utf-8'))['address']
        self._connection=Client(tuple(address),authkey=token);_active=self
    def _request(self,method,args=(),kwargs=None):
        self._ensure();self._calls+=1
        self._connection.send({'method':method,'args':args,'kwargs':kwargs or {}})
        if not self._connection.poll(7200):raise TimeoutError('ISOLATED_BACKEND_REQUEST_TIMEOUT')
        value=self._connection.recv()
        if not value['ok']:
            if value.get('error_type')=='AttributeError':raise AttributeError(value['error'])
            raise RuntimeError(f"ISOLATED_BACKEND_ERROR:{self.kind}:{method}:{value['error']}:{self._work}")
        return decoded_result(value['value'])
    def __getattr__(self,name):
        if name.startswith('__'):raise AttributeError(name)
        if name=='_rec_model':return RecProxy(self)
        value=self._request('describe',(name,))
        if value['callable']:return lambda *a,**kw:self._request(name,a,kw)
        return value['value']
    @property
    def retained_by_owner(self):
        # close() ends the actor process; querying ownership must never restart it.
        return False
    @property
    def character_positions(self):return self._cached.get('character_positions',False)
    @character_positions.setter
    def character_positions(self,value):
        self._cached['character_positions']=value;self._request('setattr',('character_positions',value))
    def __call__(self,*a,**kw):return self._request('__call__',a,kw)
    def unload(self):
        value=self._request('unload') if self._process is not None else {'unload_status':'PROCESS_ALREADY_EXITED'}
        self._stop();return value
    def close(self):
        if self._process is not None:self._request('close')
        self._stop()
    def _stop(self):
        global _active
        if self._process is None:return
        if self._process.poll() is None:
            self._connection.send({'method':'STOP','args':(),'kwargs':{}})
            self._process.wait(timeout=60)
        if self._connection is not None:self._connection.close()
        self._log.close();self._process=None;self._connection=None
        if _active is self:_active=None

def close_active():
    if _active is not None:_active._stop()

def require_table_staged_activation():
    if not _table_staged_active:
        raise RuntimeError('TABLE_STAGES_REQUIRE_ACTIVE_ISOLATED_BACKENDS')

def require_native_backend_activation():
    if not _native_isolation_active:
        raise RuntimeError('NATIVE_LOCAL_REQUIRES_ACTIVE_ISOLATED_BACKENDS')

def activate(*,table_staged=False):
    global _table_staged_active,_native_isolation_active
    from bemarkdown import pdf_layout_runtime,pdf_content_router,formula_ocr
    from bemarkdown.pdf import production_runtime,table_runtime
    layout=lambda **kw:RemoteBackend('layout',kwargs=kw)
    ocr=lambda **kw:RemoteBackend('ocr',kwargs=kw)
    table=lambda *a,**kw:RemoteBackend('table',a,kw)
    formula=lambda **kw:RemoteBackend('formula',kwargs=kw)
    pdf_layout_runtime.FormalPaddleLayoutRuntime=layout
    production_runtime.FormalPaddleLayoutRuntime=layout
    pdf_content_router.PaddleXPdfOcrRuntime=ocr
    if not table_staged:
        table_runtime.RegistryBackedTableEvidenceProvider=table
        production_runtime.RegistryBackedTableEvidenceProvider=table
    formula_ocr.create_production_formulanet_runtime=formula
    _table_staged_active=bool(table_staged)
    _native_isolation_active=True
    atexit.register(lambda:_active._stop() if _active is not None else None)

def actor(config):
    data=json.loads(pathlib.Path(config).read_text(encoding='utf-8'));folder=pathlib.Path(config).parent
    sys.path.insert(0,data['snapshot'])
    kind=data['kind'];args=data['args'];kwargs=data['kwargs']
    if kind=='layout':
        from bemarkdown.pdf_layout_runtime import FormalPaddleLayoutRuntime
        runtime=FormalPaddleLayoutRuntime(*args,**kwargs)
    elif kind=='ocr':
        from bemarkdown.pdf_content_router import PaddleXPdfOcrRuntime
        runtime=PaddleXPdfOcrRuntime(*args,**kwargs)
    elif kind=='table':
        from bemarkdown.pdf.table_runtime import RegistryBackedTableEvidenceProvider
        runtime=RegistryBackedTableEvidenceProvider(*args,**kwargs)
    elif kind=='formula':
        from bemarkdown.formula_ocr import create_production_formulanet_runtime
        runtime=create_production_formulanet_runtime(**kwargs)
    else:raise ValueError('Unknown isolated backend')
    for name,value in data.get('attributes',{}).items():setattr(runtime,name,value)
    with Listener(('127.0.0.1',0),authkey=bytes.fromhex(data['auth'])) as listener:
        (folder/'ready.json').write_text(json.dumps({'address':listener.address,'pid':os.getpid()}),encoding='utf-8')
        with listener.accept() as connection:
            while True:
                request=connection.recv();method=request['method'];arguments=request['args'];options=request['kwargs']
                if method=='STOP':break
                try:
                    if method=='describe':
                        value=getattr(runtime,arguments[0]);value={'callable':callable(value),'value':None if callable(value) else value}
                    elif method=='setattr':setattr(runtime,*arguments);value=None
                    elif method=='rec_predict':value=[normalized_result(v) for v in runtime._rec_model.predict(*arguments,**options)]
                    else:value=getattr(runtime,method)(*arguments,**options)
                    connection.send({'ok':True,'value':normalized_result(value)})
                except BaseException as error:
                    traceback.print_exc();connection.send({'ok':False,'error_type':type(error).__name__,'error':type(error).__name__+': '+str(error)})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--actor',required=True);actor(parser.parse_args().actor)
