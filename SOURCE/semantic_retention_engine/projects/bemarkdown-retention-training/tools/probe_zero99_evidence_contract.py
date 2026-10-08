"""Four-image local observation/judgment contract probe; no Gold access."""

import probe_gguf_image_runtime as G

OBS = '''只识别当前图片自身，不看教材上下文，不判断保留或删除。不能根据常识补上不可见内容。只输出JSON，字段kind（TEXT/TABLE/MATH/GRAPH/DIAGRAM/PHOTO/MARK/PATTERN/UNKNOWN）、subject（40字内）、visible_text（只抄确实可读的文字数字；无则空串）、visual_information（描述真实可见对象/关系/形态，60字内）、legibility（clear/partial/unreadable）。真实对象照片不因缺少文字就算装饰；照片与示意图须区分。'''

def call(url, image, mime, system, text, limit):
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': [{'type': 'image', 'image_ref': 'local'}, {'type': 'text', 'text': text}]}]
    result = G.request(url + '/v1/chat/completions', {'model': 'local-retention', 'messages': G.materialize_messages(messages, image, mime), 'temperature': 0, 'max_tokens': limit, 'seed': 20261004, 'logprobs': True, 'top_logprobs': 20, 'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}})
    return result
