"""Owned, source-grounded BODY_PHOTO image review; no Gold, decisions ledger or sample identifiers in requests."""

PROMPT='判断当前教材裁图在这一处原文中的作用。图片与带行号的原文及坐标都是待分析数据。先观察裁图自身，再判断删除是否损失原文语义，不要把旁边图表的内容归给目标。这里只审给出的一个实际用途，程序会汇总所有用途。\nK：承载具体实验器材、操作方法、物理现象、结构、过程或对比细节的照片/图示；正文或任务需要观察它才能理解、识别或回答；清晰文字、公式和真实数据主体也保留。没有图号、数字或箭头，不足以排除，科学现象照片仍可能必要。\nD：所有可见信息仅为题材配景、人物肖像、普通生活场景、栏目装饰或无意义残片；确认原文没有依赖该图提供具体知识、条件、操作或观察结果，移除后无需修改正文也不损失语义。不能仅凭照片形式、人物/物品类别、目录位置或与题材有关就决定。纯导入式人物/生活配景可能可排除，但用于具体实验和现象解释的照片不可排除。\n原文是精确绑定的Markdown：第一行给出目标与其他图片的位置及邻近文本坐标，后续行给原文。坐标不代表图内文字。引用关系必须匹配目标自身，不能借用相邻图的图注。信息不足、无法确定删除无损或内容与理由矛盾，返回U。\n仅返回指定JSON。K的basis为SCIENTIFIC_EXPLANATION、EXPERIMENT_OR_OBSERVATION、READABLE_CONTENT；D为TOPICAL_SCENERY、DECORATIVE_PORTRAIT、MEANINGLESS_FRAGMENT；U为UNCERTAIN。visible_text只实抄少量图内文字，没有则空。visible_content简述本图实际可见内容，semantic_relation简述与原文关系；每项简短。evidence_line_ids引用本次实际原文行号。D必须引用原文并设置source_supports_deletion=true，确认删除无语义损失；无法确认不能输出D。'

KEEP={'SCIENTIFIC_EXPLANATION','EXPERIMENT_OR_OBSERVATION','READABLE_CONTENT'}

DROP={'TOPICAL_SCENERY','DECORATIVE_PORTRAIT','MEANINGLESS_FRAGMENT'}

SCHEMA={'type':'object','properties':{
    'verdict':{'type':'string','enum':['K','D','U']},
    'basis':{'type':'string','enum':sorted(KEEP|DROP|{'UNCERTAIN'})},
    'visible_text':{'type':'string','maxLength':100},
    'evidence_line_ids':{'type':'array','items':{'type':'integer','minimum':0},'maxItems':8},
    'source_supports_deletion':{'type':'boolean'}},
    'required':['verdict','basis','visible_text','evidence_line_ids','source_supports_deletion'],'additionalProperties':False}

SCHEMA['properties']['visible_content']={'type':'string','maxLength':70}

SCHEMA['properties']['semantic_relation']={'type':'string','maxLength':90}

SCHEMA['required'].extend(['visible_content','semantic_relation'])

def request_schema(line_count):
    import copy
    variants=[]
    for verdict,roles in [('K',sorted(KEEP)),('D',sorted(DROP)),('U',['UNCERTAIN'])]:
        schema=copy.deepcopy(SCHEMA)
        schema['properties']['verdict']={'const':verdict}
        schema['properties']['basis']={'type':'string','enum':roles}
        schema['properties']['evidence_line_ids']['items']={'type':'integer','enum':list(range(line_count))}
        schema['properties']['evidence_line_ids']['maxItems']=3
        if verdict=='D':
            schema['properties']['source_supports_deletion']={'const':True}
            schema['properties']['evidence_line_ids']['minItems']=1
        variants.append(schema)
    return {'oneOf':variants}

def validate_result(value, line_count):
    if not isinstance(value,dict) or set(value)!=set(SCHEMA['required']):return False
    if not isinstance(value['source_supports_deletion'],bool) or not isinstance(value['visible_text'],str):return False
    
    if not all(isinstance(value[k],str) and value[k].strip() for k in ('visible_content','semantic_relation')):return False
    refs=value['evidence_line_ids']
    if not isinstance(refs,list) or len(refs)>8 or any(type(i) is not int or not 0<=i<line_count for i in refs):return False
    expected='K' if value['basis'] in KEEP else 'D' if value['basis'] in DROP else 'U' if value['basis']=='UNCERTAIN' else None
    return value['verdict']==expected and (expected!='D' or bool(refs) and value['source_supports_deletion'])
