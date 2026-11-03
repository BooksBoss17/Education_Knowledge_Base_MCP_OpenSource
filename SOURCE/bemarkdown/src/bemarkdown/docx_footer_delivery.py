"""Deliver source-backed literal text from actually referenced DOCX footers once.

Word pagination and dynamic PAGE fields remain explicitly uncertified. This
thin layer never treats XML branch duplication as extra rendered copies.
"""
import hashlib
import json
import posixpath
import re
import zipfile
from lxml import etree

W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
MC='http://schemas.openxmlformats.org/markup-compatibility/2006'
NS={'w':W,'r':R,'mc':MC}
VERSION='docx-referenced-footer-literal-v1'
DYNAMIC=re.compile(r'\b(?:PAGE|NUMPAGES|SECTIONPAGES)\b')


def _sha(data):return hashlib.sha256(data).hexdigest()


def append_referenced_footer_text(source_path, markdown):
    parser=etree.XMLParser(resolve_entities=False,no_network=True,load_dtd=False)
    records=[];pending=[];branches=[];parts=[]
    with zipfile.ZipFile(source_path) as z:
        body=etree.fromstring(z.read('word/document.xml'),parser)
        relationships=etree.fromstring(z.read('word/_rels/document.xml.rels'),parser)
        rels={r.get('Id'):r for r in relationships}
        for section in body.xpath('.//w:sectPr',namespaces=NS):
            for ref in section.xpath('./w:footerReference',namespaces=NS):
                rid=ref.get('{'+R+'}id');rel=rels[rid]
                if rel.get('TargetMode')=='External':
                    pending.append({'relationship_id':rid,'reason':'EXTERNAL_FOOTER_NOT_FETCHED'});continue
                part=posixpath.normpath(posixpath.join('word',rel.get('Target')))
                if part.startswith('../') or part not in z.namelist():raise ValueError('DOCX_FOOTER_RELATION_PATH')
                if part not in parts:parts.append(part)
        def walk(node,part,events):
            name=etree.QName(node).localname
            if name=='AlternateContent':
                choice=node.find('mc:Choice',NS);fallback=node.find('mc:Fallback',NS)
                selected=choice if choice is not None else fallback
                branches.append({'source_part':part,'source_locator':node.getroottree().getpath(node),
                    'selected_branch':'Choice' if choice is not None else 'Fallback',
                    'policy':'CURRENT_SCANNER_CHOICE_FIRST_LITERAL_TRAVERSAL',
                    'actual_overlay_visit':True,'not_all_XML_branches_counted':True})
                if selected is not None:walk(selected,part,events)
                return
            if name in {'del','instrText'}:return
            if name=='r' and node.xpath('./w:rPr/w:vanish[not(@w:val="0" or @w:val="false" or @w:val="off")]|./w:rPr/w:webHidden',namespaces=NS):return
            if name=='fldSimple':
                if DYNAMIC.search(node.get('{'+W+'}instr','')):
                    events.append({'kind':'DYNAMIC_FIELD_UNMEASURED','source_locator':node.getroottree().getpath(node)})
                    return
            if name=='p' and DYNAMIC.search(' '.join(node.xpath('.//w:instrText/text()',namespaces=NS))):
                # PAGE cached results are not literal footer text. Still walk
                # branches for an exact adoption ledger, without emitting them.
                for alt in node.xpath('.//mc:AlternateContent',namespaces=NS):
                    choice=alt.find('mc:Choice',NS)
                    branches.append({'source_part':part,'source_locator':alt.getroottree().getpath(alt),
                        'selected_branch':'Choice' if choice is not None else 'Fallback',
                        'policy':'CURRENT_SCANNER_CHOICE_FIRST_DYNAMIC_FIELD_INSPECTION',
                        'actual_overlay_visit':True,'dynamic_value_emitted':False})
                events.append({'kind':'DYNAMIC_FIELD_UNMEASURED','source_locator':node.getroottree().getpath(node)})
                return
            if name in {'drawing','pict','object','oMath','oMathPara'}:
                events.append({'kind':'UNSUPPORTED_VISUAL_OR_MATH_PRESERVED_PENDING',
                    'source_locator':node.getroottree().getpath(node)});return
            if name=='t':events.append({'kind':'TEXT','text':node.text or ''});return
            if name=='tab':events.append({'kind':'TEXT','text':'\t'});return
            if name in {'br','cr'}:events.append({'kind':'TEXT','text':'\n'});return
            for child in node:walk(child,part,events)
        for part in parts:
            xml=z.read(part);root=etree.fromstring(xml,parser)
            for paragraph in root.xpath('./w:p',namespaces=NS):
                events=[];walk(paragraph,part,events)
                locator=root.getroottree().getpath(paragraph)
                if any(e['kind']=='UNSUPPORTED_VISUAL_OR_MATH_PRESERVED_PENDING' for e in events):
                    pending.append({'source_part':part,'source_locator':locator,
                        'reason':'FOOTER_VISUAL_OR_MATH_REQUIRES_SOURCE_REVIEW'});continue
                text=''.join(e['text'] for e in events if e['kind']=='TEXT').strip()
                if not text:
                    if any(e['kind']=='DYNAMIC_FIELD_UNMEASURED' for e in events):
                        records.append({'source_part':part,'source_locator':locator,
                            'status':'DYNAMIC_FIELD_NOT_EVALUATED','not_page_count_guess':True})
                    continue
                # Match the normal literal-text escaping. No semantic repair.
                emitted=re.sub(r'([\\`*\[\]])',r'\\\1',text)
                prefix=('' if markdown.endswith('\n\n') else '\n' if markdown.endswith('\n') else '\n\n') if markdown else ''
                markdown+=prefix
                a=len(markdown.encode('utf-8'));markdown+=emitted
                b=len(markdown.encode('utf-8'))
                records.append({'source_part':part,'source_locator':locator,
                    'source_part_xml_sha256':_sha(xml),'source_text_sha256':_sha(text.encode('utf-8')),
                    'status':'LITERAL_FOOTER_DELIVERED_ONCE_PER_REFERENCED_PART',
                    'output_utf8_byte_range':[a,b],'output_text_sha256':_sha(emitted.encode('utf-8')),
                    'source_occurrence_uid':_sha((part+':'+locator).encode('utf-8'))})
    if markdown and not markdown.endswith('\n'):markdown+='\n'
    digest=_sha(markdown.encode('utf-8'))
    for row in records:
        if 'output_utf8_byte_range' in row:
            a,b=row['output_utf8_byte_range']
            assert _sha(markdown.encode('utf-8')[a:b])==row['output_text_sha256']
            row['render_occurrence_uid']=_sha((digest+':'+row['source_occurrence_uid']+':'+str(a)).encode('utf-8'))
    return markdown,{'version':VERSION,'records':records,'pending':pending,'adopted_branches':branches,
        'referenced_parts':parts,'markdown_sha256':digest,'word_pagination_certified':False,
        'emission_policy':'INCLUDE_ONCE_PER_REFERENCED_PART_NOT_ONCE_PER_WORD_PAGE','gold_read':False}
