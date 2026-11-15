import numpy as np
import pytest
from bemarkdown.pdf.spatial_assembly import fragment_decision


def mask(points, shape=(32,32)):
    out=np.zeros(shape,dtype=bool)
    for x0,y0,x1,y1 in points:out[y0:y1,x0:x1]=True
    return out


@pytest.mark.parametrize('scale',[1,2,3])
def test_punctuation_and_sparse_edge_residue_are_filtered_at_each_scale(scale):
    comma=mask([(12,22,15,27),(11,26,13,30)])
    scaled=np.repeat(np.repeat(comma,scale,axis=0),scale,axis=1)
    assert fragment_decision(scaled,30*scale)[0]=='ISOLATED_LOW_INFORMATION_MARK'
    sliver=mask([(0,0,2,2),(0,20,2,23)],(40,3))
    assert fragment_decision(np.repeat(np.repeat(sliver,scale,0),scale,1),30*scale)[0]=='SPARSE_CUT_EDGE_SLIVER'


@pytest.mark.parametrize('points',[
    [(10,4,13,28)],  # narrow continuous 1 / l
    [(5,4,20,7),(5,4,8,28),(17,4,20,28),(5,25,20,28)], # zero
    [(5,13,22,16)], # minus
    [(5,8,22,11),(5,19,22,22)], # equals
    [(5,5,8,28),(5,5,22,8),(5,15,22,18),(19,5,22,28)], # whole character
    [(5,5,8,28),(15,25,18,28),(23,5,26,28)], # decimal within a full token
    [(5,10,8,29),(15,2,20,5),(18,2,21,10)], # base with superscript
])
def test_complete_character_and_sign_shapes_survive(points):
    assert fragment_decision(mask(points),30)[0] is None


def test_blank_is_not_an_ocr_request():
    reason,bounds,metrics=fragment_decision(np.zeros((40,90),dtype=bool),30)
    assert reason=='EMPTY' and bounds is None and metrics['ink_pixels']==0


def test_whitespace_bounds_do_not_discard_foreground():
    ink=mask([(10,5,22,29)],(50,80))
    reason,box,_=fragment_decision(ink,30)
    assert reason is None and box==(10,5,22,29)
    a,b,c,d=box
    assert ink[b:d,a:c].sum()==ink.sum()


def test_filter_omits_ocr_slot_but_keeps_both_formula_owners(tmp_path):
    import hashlib
    from PIL import Image,ImageDraw
    from bemarkdown.pdf.local_model_api import api_policy
    from bemarkdown.pdf.visual_serial_pipeline import VisualRegion,build_visual_serial_plan,prepare_text_inputs
    from bemarkdown.pdf.spatial_assembly import prepare_slots,assemble
    im=Image.new('RGB',(140,40),'white');draw=ImageDraw.Draw(im)
    draw.rectangle((5,5,25,29),fill='black');draw.rectangle((75,25,77,29),fill='black')
    def region(name,label,box,image):
        path=tmp_path/(name+'.png');image.save(path)
        return VisualRegion(name,label,box,{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    p=region('parent','text',(0,0,140,40),im)
    f1=region('f1','formula',(30,5,70,30),im.crop((30,5,70,30)))
    f2=region('f2','formula',(80,5,130,30),im.crop((80,5,130,30)))
    values={'f1':'x=1','f2':'y=2'}
    plan=prepare_text_inputs(build_visual_serial_plan([p,f1,f2],policy=api_policy()),values,tmp_path)
    effective,groups,_,_=prepare_slots(plan,values,tmp_path)
    assert len(effective.text_regions)==1
    removed=groups[0]['filtered_fragments']
    assert len(removed)==1 and removed[0]['region_id'] not in [r.region_id for r in effective.text_regions]
    text={effective.text_regions[0].region_id:'条件'}
    assembled,consumed,provenance=assemble(groups,text,values,[])
    assert list(assembled.values())==['条件 $x=1$ $y=2$']
    assert 'f1' in consumed and 'f2' in consumed
    assert provenance[0]['filtered_fragments']==removed
