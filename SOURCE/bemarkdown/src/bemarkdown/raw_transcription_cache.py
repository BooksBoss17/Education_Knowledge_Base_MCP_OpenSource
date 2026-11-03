"""Raw model execution identity, separate from current interpretation and routing."""
import hashlib
import json
import re
from .agent_usage import add_step_usage

SCHEMA = 'flash-raw-model-request-v1'
# This protocol fixes the headless CLI and workspace-write image tool contract.
# A change to inference-affecting CLI/environment behavior must bump this schema;
# validator and route changes are audited separately and must not bump it.


def model_input_bytes(tasks):
    # This exact file is visible to the model. Code hashes are deliberately absent.
    return (json.dumps({'tasks':tasks},ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def request_key(request):
    return hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


def read_execution_events(path):
    """Keep only execution evidence, without accumulating tool image/text payloads."""
    events=[];sessions=[];usage={};models=[];steps=0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:event=json.loads(line)
            except ValueError:continue
            if event.get('type')=='session':sessions.append(event.get('sessionId'))
            if event.get('model'):models.append(event['model'])
            if event.get('type')=='status' and event.get('phase')=='step_end':
                add_step_usage(usage,event.get('usage',{}))
                steps+=1
            if event.get('type')=='tool_call' and event.get('tool')=='read_image':
                events.append({'type':'tool_call','tool':'read_image','callId':event.get('callId'),
                               'input':{'file_path':event.get('input',{}).get('file_path','')}})
            elif event.get('type')=='tool_result':
                record={'type':'tool_result','callId':event.get('callId'),'status':event.get('status')}
                result=event.get('result')
                if isinstance(result,str) and len(result)<8192 and '<type>image</type>' in result:
                    match=re.search(r'<path>([^\r\n<>]+)</path>',result)
                    if match:record['image_result_path']=match.group(1)
                events.append(record)
    return events,sessions,usage,models,steps
