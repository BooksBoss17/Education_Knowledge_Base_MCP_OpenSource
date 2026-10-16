"""Reuse independently reviewed whole-target scope only against bound render evidence.

This certifies which output belongs to a target. It does not certify OCR accuracy
or turn a global text-alignment proposal into character geometry.
"""
import hashlib
import json
import math
from pathlib import Path
from PIL import Image

VERSION='independent-pdf-target-scopes-v1'
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


class ReviewedPdfTargetScopes:
    def __init__(self,index_path):
        self.path=Path(index_path).resolve();raw=self.path.read_bytes();index=json.loads(raw.decode('utf-8'))
        self.index_sha256=hashlib.sha256(raw).hexdigest()
        if index.get('version')!=VERSION:raise ValueError('PDF_SCOPE_INDEX_VERSION')
        self.records=[]
        for entry in index['batches']:
            for value in entry.values():
                if _sha(value['path'])!=value['sha256']:raise ValueError('PDF_SCOPE_REVIEW_EVIDENCE_CHANGED')
            packet,review,receipt=(_read(entry[k]['path']) for k in ('packet','review','receipt'))
            if review['packet_sha256']!=entry['packet']['sha256'] or receipt['packet_sha256']!=entry['packet']['sha256']:
                raise ValueError('PDF_SCOPE_PACKET_BINDING_CHANGED')
            images={str(Path(r['path']).resolve()):r for r in packet['images']}
            reads=receipt['images']
            if len(reads)!=len(images) or not 1<=len(reads)<=4:
                raise ValueError('PDF_SCOPE_IMAGE_READ_POPULATION')
            if {str(Path(r['path']).resolve()) for r in reads}!=set(images):
                raise ValueError('PDF_SCOPE_IMAGE_READ_SET')
            for row in reads:
                expected=images[str(Path(row['path']).resolve())]
                if (_sha(row['path'])!=expected['sha256'] or row['sha256']!=expected['sha256']
                        or not (row.get('actually_viewed') is True or row.get('actual_visual_read') is True)):
                    raise ValueError('PDF_SCOPE_IMAGE_READ_UNVERIFIED')
            source_image=next(r for r in packet['images'] if r['role']=='SOURCE_PAGE')
            binding=packet['source_binding']
            if source_image['sha256']!=binding['actual_image_sha256']:
                raise ValueError('PDF_SCOPE_SOURCE_IMAGE_CHANGED')
            with Image.open(source_image['path']) as image:width,height=image.size
            original=Path(packet['final_markdown_path']).read_bytes()
            if hashlib.sha256(original).hexdigest()!=packet['final_markdown_sha256']:
                raise ValueError('PDF_SCOPE_ORIGINAL_RENDER_CHANGED')
            parent=packet['parent_output_text'].encode('utf-8');start,end=binding['parent_output_byte_range']
            if original[start:end]!=parent:raise ValueError('PDF_SCOPE_PARENT_RENDER_CHANGED')
            targets={r['sample_uid']:r for r in packet['items']}
            findings=review.get('items',review.get('findings',[]))
            if {r['sample_uid'] for r in findings}!=set(targets):
                raise ValueError('PDF_SCOPE_TARGET_POPULATION_CHANGED')
            for finding in findings:
                if finding['scope_verdict']!='CONFIRMED_TARGET_SCOPE':continue
                target=targets[finding['sample_uid']];a,b=target['proposed_output_byte_range']
                crop=next(r for r in packet['images'] if r.get('sample_uid')==finding['sample_uid'])
                if (finding['source_crop_sha']!=crop['sha256'] or finding['output_byte_range']!=[a,b]
                        or finding['output_sha']!=target['proposed_output_sha256']
                        or finding['final_markdown_sha256']!=packet['final_markdown_sha256']
                        or not start<=a<=b<=end
                        or hashlib.sha256(original[a:b]).hexdigest()!=target['proposed_output_sha256']):
                    raise ValueError('PDF_SCOPE_REVIEW_TARGET_BINDING_CHANGED')
                self.records.append({'source':binding,'target_bbox_pdf_pt':target['target_bbox_pdf_pt'],
                    'parent_sha256':hashlib.sha256(parent).hexdigest(),'relative_byte_range':[a-start,b-start],
                    'output_sha256':target['proposed_output_sha256'],'source_image_size':[width,height],
                    'review_evidence':entry,'review_target_id':finding['sample_uid'],
                    'reason':finding['reason'],'uncertainty':finding.get('uncertainty','')})

    def locate(self,projection,markdown,*,page_index,bbox_pdf_pt):
        if _sha(self.path)!=self.index_sha256:raise ValueError('PDF_SCOPE_INDEX_CHANGED')
        if len(bbox_pdf_pt)!=4 or not all(math.isfinite(x) for x in bbox_pdf_pt):
            raise ValueError('PDF_SCOPE_TARGET_GEOMETRY_INVALID')
        data=markdown.encode('utf-8') if isinstance(markdown,str) else markdown
        if hashlib.sha256(data).hexdigest()!=projection['markdown_sha256']:
            raise ValueError('PDF_SCOPE_CURRENT_RENDER_CHANGED')
        matches=[]
        for record in self.records:
            source=record['source']
            if (source['processing_input_sha256']!=projection['processing_input_sha256'] or source['page_index']!=page_index
                    or any(abs(a-b)>1e-6 for a,b in zip(record['target_bbox_pdf_pt'],bbox_pdf_pt,strict=True))):continue
            origin=source['source_crop_transform']['origin_pdf_pt'];scale=source['source_crop_transform']['pdf_points_per_render_pixel']
            width,height=record['source_image_size'];expected_box=[*origin,origin[0]+width*scale,origin[1]+height*scale]
            for row in projection['occurrences']:
                box=row.get('bbox_pdf_pt')
                source_views={row.get('source_image_sha256'),row.get('actual_input_sha256')}
                if (row['page_index']!=page_index or source['actual_image_sha256'] not in source_views
                        or not box or any(abs(a-b)>scale+.01 for a,b in zip(box,expected_box,strict=True))):continue
                start,end=row['output_markdown_byte_range'];parent=data[start:end]
                if hashlib.sha256(parent).hexdigest()!=record['parent_sha256']:continue
                a,b=record['relative_byte_range'];actual=[start+a,start+b]
                if hashlib.sha256(data[actual[0]:actual[1]]).hexdigest()!=record['output_sha256']:
                    raise ValueError('PDF_SCOPE_CURRENT_TARGET_CHANGED')
                matches.append({'render_occurrence_uid':row['render_occurrence_uid'],
                    'output_markdown_byte_range':actual,'output_text_sha256':record['output_sha256'],
                    'review_evidence':record['review_evidence'],'reason':record['reason'],'uncertainty':record['uncertainty']})
        if not matches:return None
        identities=[(r['render_occurrence_uid'],tuple(r['output_markdown_byte_range'])) for r in matches]
        if len(set(identities))!=len(identities):raise ValueError('PDF_SCOPE_DUPLICATE_APPROVAL')
        return {'status':'INDEPENDENTLY_REVIEWED_TARGET_SCOPE_MATCH',
            'render_occurrence_uids':[r['render_occurrence_uid'] for r in matches],
            'target_occurrences':matches,'scope_certified':True,'accuracy_certified':False,
            'character_geometry_certified':False,'no_best_substring_selected':True,
            'index_path':str(self.path),'index_sha256':_sha(self.path)}
