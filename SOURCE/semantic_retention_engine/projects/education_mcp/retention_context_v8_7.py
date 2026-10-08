"""Label-free content observation and Markdown-grounded image retention."""

from collections import defaultdict,Counter

from pathlib import Path

import hashlib

import json

import re

import runpy

CORE=runpy.run_path(str(Path(__file__).with_name('context_retention.py')))

TITLES={'问题','思考','思考与讨论','讨论与交流','练习','练习与应用','实验与探究','实验探究',
        '自主活动','科学漫步','科学方法','科学书屋','小资料','拓展一步','拓展学习','实践活动',
'迷你实验室','物理聊吧','问题与练习','请提问','我的学习总结','单元自我评价'}

TITLES.update({'演示','做一做'})

TITLES.update({'迁移','方法点拨','策略提炼'})

FIG=re.compile(r'^\s*(?:图|表)\s*\d+(?:[.．\-－—–]\d+)?(?:\s|[\u3400-\u9fff]|$)')

PANEL=re.compile(r'^\s*[（(][a-zA-Z甲乙丙丁][)）]\s*[^。？！\n]{0,28}$')

def normalize(text):return re.sub(r'[\s#*_]','',text)

def gap(a,b):return max(a[0]-b[2],b[0]-a[2],0)+max(a[1]-b[3],b[1]-a[3],0)

class Context:
    def __init__(self,prepared):
        import pymupdf
        self.prepared=prepared
        self.package=Path(prepared['package'])
        self.raw=(self.package/'document.md').read_bytes()
        self.md_canonical=re.sub(r'[^\w\u3400-\u9fff]','',self.raw.decode('utf8'))
        if hashlib.sha256(self.raw).hexdigest()!=prepared['markdown_sha256']:raise ValueError('MD changed')
        if CORE['sha'](prepared['source']['path'])!=prepared['source']['sha256']:raise ValueError('Source changed')
        report=json.loads((self.package/'conversion_report.json').read_text('utf8'))
        self.pages=defaultdict(list)
        for span in report['markdown_render']['node_spans']:
            text=self.raw[span['byte_start']:span['byte_end']]
            if span.get('sha256') and hashlib.sha256(text).hexdigest()!=span['sha256']:raise ValueError('Span changed')
            self.pages[span['page_index']].append(dict(span,text=text.decode('utf8')))
        self.doc=pymupdf.open(prepared['source']['path'])
        self.native={}
        self.heading_counts=Counter()
        heading_pages=defaultdict(set)
        self.headings={}
        for page in self.pages:
            lines=[]
            # Only native text geometry is needed; do not decode all page images.
            for block in self.doc[page].get_text('dict',flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)['blocks']:
                for line in block.get('lines',[]):
                    text=''.join(s['text'] for s in line['spans']).strip()
                    if text:lines.append(dict(text=text,bbox=list(line['bbox'])))
            self.native[page]=lines
            md_headings=[]
            for span in self.pages[page]:
                first=next((normalize(s) for s in span['text'].splitlines() if normalize(s)),'')
                if first in TITLES:
                    b=list(span['bbox_pdf_pt']);b[3]=min(b[3],b[1]+22)
                    md_headings.append(dict(text=first,bbox=b))
                text=normalize(span['text']);b=span['bbox_pdf_pt'];h=self.doc[page].rect.height
                if 1<=len(text)<=40 and (b[3]/h<.17 or b[1]/h>.9):
                    if not FIG.match(span['text']) and not PANEL.match(span['text']):
                        lines.append(dict(text=text,bbox=list(b)))
            heads=[n for n in lines if normalize(n['text']) in TITLES]+md_headings
            self.headings[page]=heads
            for n in heads:heading_pages[normalize(n['text'])].add(page)
        self.heading_counts=Counter({text:len(pages) for text,pages in heading_pages.items()})
        self.margin_text_pages=defaultdict(set)
        for page,lines in self.native.items():
            h=self.doc[page].rect.height
            for n in lines:
                b=n['bbox'];norm=normalize(n['text'])
                if len(norm)<=40 and (b[3]/h<.17 or b[1]/h>.9):
                    key=(norm,'top' if b[3]/h<.17 else 'bottom')
                    self.margin_text_pages[key].add(page)
        self.corner_pages=defaultdict(set)
        for asset in prepared['assets']:
            for o in asset['occurrences']:
                b=o['bbox'];rect=self.doc[o['page']].rect
                if self.corner_key(b,rect):self.corner_pages[self.corner_key(b,rect)].add(o['page'])

    @staticmethod
    def corner_key(box,rect):
        width=box[2]-box[0];height=box[3]-box[1]
        if width<100 and height<65 and (box[3]/rect.height<=.10 or box[1]/rect.height>=.95):
            return tuple(round(v*50) for v in [box[0]/rect.width,box[1]/rect.height,width/rect.width,height/rect.height])
        return None

    def facts(self,asset,occurrence):
        box=occurrence['bbox'];page=occurrence['page'];rect=self.doc[page].rect
        spans=self.pages.get(page,[])
        captions=[]
        for n in spans:
            text=normalize(n['text'])
            # A paragraph mentioning a figure is not itself the figure caption.
            if len(text)>110 or not (FIG.match(n['text']) or PANEL.match(n['text'])):continue
            b=n['bbox_pdf_pt']
            if gap(box,b)<=75:captions.append(n)
        captions.sort(key=lambda n:gap(box,n['bbox_pdf_pt']))
        roles=[]
        corner=self.corner_key(box,rect)
        if corner and len(self.corner_pages[corner])>=5:
            roles.append(dict(role='REPEATED_SMALL_PAGE_EDGE_LAYOUT',distinct_pages=len(self.corner_pages[corner])))
        repeated=len({o['page'] for o in asset['occurrences']})>=3
        margin=(box[3]/rect.height<=.14 or box[1]/rect.height>=.92)
        if repeated and margin:
            companions=[n for n in self.native.get(page,[]) if gap(box,n['bbox'])<45 and
                (n['bbox'][3]/rect.height<.17 or n['bbox'][1]/rect.height>.91) and
                len(self.margin_text_pages[(normalize(n['text']),'top' if n['bbox'][3]/rect.height<.17 else 'bottom')])>=3]
            if companions:roles.append(dict(role='REPEATED_MARGIN_COMPANION',texts=[n['text'] for n in companions][:3]))
        for head in self.headings.get(page,[]):
            b=head['bbox'];overlap=min(box[3],b[3])-max(box[1],b[1])
            left=box[2]<=b[0]+10 and 0<=b[0]-box[2]<55
            behind=box[0]<=b[0] and box[2]>=b[2] and overlap>0
            if (left and overlap>0) or behind:
                roles.append(dict(role='COLUMN_HEADING_MARK',heading=normalize(head['text']),
                    heading_repeated_pages=self.heading_counts[normalize(head['text'])]))
        # Bind the target in a continuous page-local Markdown fragment.
        start=occurrence['byte_start'];end=occurrence['byte_end']
        page_start=min(n['byte_start'] for n in spans) if spans else start
        page_end=max(n['byte_end'] for n in spans) if spans else end
        before=self.raw[max(page_start,start-1800):start].decode('utf8',errors='ignore')
        after=self.raw[end:min(page_end,end+2400)].decode('utf8',errors='ignore')
        before=CORE['IMAGE'].sub('[OTHER_IMAGE]',before)[-600:]
        after=CORE['IMAGE'].sub('[OTHER_IMAGE]',after)[:850]
        # The renderer's target placeholder directly precedes a named column.
        # This remains observable MD structure even for scanned PDFs with no native text.
        first_lines=[normalize(line) for line in after.splitlines() if normalize(line)]
        if first_lines and first_lines[0] in TITLES:
            roles.append(dict(role='TARGET_IMMEDIATELY_PRECEDES_NAMED_COLUMN',heading=first_lines[0]))
        md=before+'\n[TARGET_IMAGE]\n'+after
        contents_text=normalize('\n'.join(n['text'] for n in spans)).lower()
        document_role='contents' if ('目录' in contents_text or 'contents' in contents_text or len(re.findall(r'[·.]{5,}\s*\d+',contents_text))>=6) else 'lesson'
        if page<=1 and re.search(r'普通高中教科书|普通高中课程标准实验教科书',contents_text):document_role='publication_cover'
        if page>=self.doc.page_count-2 and re.search(r'isbn|(?:印刷|出版|发行|定价|版次).{0,20}(?:印刷|出版|发行|定价|版次)',contents_text):document_role='publication_metadata'
        native_captions=[n for n in self.native.get(page,[]) if gap(box,n['bbox'])<=75 and n['bbox'][1]>=box[3]-8 and min(box[2],n['bbox'][2])-max(box[0],n['bbox'][0])>0]
        native_captions.sort(key=lambda n:(n['bbox'][1],n['bbox'][0]))
        native_text=''.join(n['text'] for n in native_captions[:12])
        canonical=lambda t:re.sub(r'[^\w\u3400-\u9fff]','',t)
        md_all=self.md_canonical
        cues=[]
        if re.search(r'图\s*\d+(?:[.．－-]\d+)+',native_text):
            match=re.search(r'(?:列车|火车|飞机|小车|小球|弹簧|支架|空调|风扇|鸽子).{0,24}(?:视为.{0,6}(?:点|质点)|建立.{0,10}坐标|参照|参考系)',native_text)
            if match and canonical(match.group()) in md_all:cues.append(dict(text=match.group(),source='MD-verified source-caption text with native page geometry',bbox=[n['bbox'] for n in native_captions[:12]]))
        return dict(source_kind=occurrence['source_kind'],page_number=page+1,document_role=document_role,source_usage_cues=cues,
            target_bounds=[round(v,2) for v in box],page_size=[rect.width,rect.height],
            full_page_like=(box[2]-box[0])/rect.width>.85 and (box[3]-box[1])/rect.height>.8,
            layout=roles,caption_candidates=[dict(text=n['text'][:150],distance=round(gap(box,n['bbox_pdf_pt']),1),
                relative='below' if n['bbox_pdf_pt'][1]>=box[1] else 'above') for n in captions[:3]],markdown=md)

    def close(self):self.doc.close()

    def linked_facts(self,asset,occurrence,base):
        """Source-MD spans with page/bbox provenance, no old decision or gold.

        MD byte adjacency is unreliable when the renderer writes text before
        separately extracted images. Keep source span order and expose geometry.
        """
        page=occurrence['page'];box=occurrence['bbox'];items=[]
        for p in [page,page-1,page+1]:
            for n in self.pages.get(p,[]):
                if n['kind'] in {'IMAGE','OTHER'}:continue
                text=CORE['IMAGE'].sub('[OTHER_IMAGE]',n['text']).strip()
                if not text:continue
                b=n['bbox_pdf_pt'];distance=gap(box,b) if p==page else 500
                items.append(dict(page=p+1,kind=n['kind'],bbox=[round(v,1) for v in b],text=text,
                    distance=distance,offset=n['byte_start']))
        # Exact nearby captions first; full text blocks on the same page next.
        items.sort(key=lambda n:(n['page']!=page+1,n['distance'],n['offset']))
        chosen=[];budget=0
        for n in items:
            remaining=2400-budget
            if remaining<=0 or len(chosen)>=24:break
            text=n['text'][:remaining];budget+=len(text)
            chosen.append({k:v for k,v in n.items() if k not in {'distance','offset','text'}}|dict(text=text))
        peers=[dict(bbox=[round(v,1) for v in o['bbox']],source_kind=o['source_kind'])
            for a in self.prepared['assets'] for o in a['occurrences'] if o['page']==page and o['id']!=occurrence['id']]
        peers.sort(key=lambda p:gap(box,p['bbox']))
        result={k:v for k,v in base.items() if k!='markdown'};result['linked_md_spans']=chosen;result['source_image_regions']=peers[:8]
        result['linked_md_spans']=[[n['page'],n['bbox'],n['text']] for n in chosen]
        result['context_method']='MD spans [page,bbox,text]; neighboring image regions carry no inferred content'
        return result
