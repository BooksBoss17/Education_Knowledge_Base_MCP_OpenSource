from dataclasses import replace
import hashlib
from PIL import Image, ImageDraw
from bemarkdown.pdf.spatial_assembly import deduplicate, prepare_slots, assemble, contained_results
from bemarkdown.pdf.visual_serial_pipeline import VisualRegion, build_visual_serial_plan, prepare_text_inputs
from bemarkdown.pdf.local_model_api import api_policy


def region(tmp_path, name, label, box, image=None):
    image = image or Image.new('RGB',(box[2]-box[0],box[3]-box[1]),'white')
    path=tmp_path/(name+'.png');image.save(path)
    return VisualRegion(name,label,box,{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


def test_duplicate_source_does_not_remove_other_occurrence(tmp_path):
    a=region(tmp_path,'a','chart',(0,0,20,20))
    b=replace(a,region_id='b',label='image')
    c=replace(a,region_id='c',bbox=(30,0,50,20))
    kept=deduplicate([a,b,c])
    assert [r.region_id for r in kept]==['a','c']
    assert kept[0].metadata['source_aliases']==['b']


def test_formula_slots_preserve_option_values_and_source_order(tmp_path):
    im=Image.new('RGB',(180,70),'white');d=ImageDraw.Draw(im)
    for y in (10,45):
        d.rectangle((5,y,20,y+12),fill='black')
        d.rectangle((90,y,105,y+12),fill='black')
    parent=region(tmp_path,'text','text',(0,0,180,70),im)
    formulas=[region(tmp_path,str(i),'formula',box) for i,box in enumerate([(25,8,75,25),(110,8,170,25),(25,43,75,60),(110,43,170,60)])]
    values={f.region_id:'x=1' for f in formulas}
    plan=build_visual_serial_plan([parent,*formulas],policy=api_policy())
    plan=prepare_text_inputs(plan,values,tmp_path)
    effective,groups,aliases,reviews=prepare_slots(plan,values,tmp_path)
    assert len(effective.text_regions)==4 and not aliases and not reviews
    texts={r.region_id:v for r,v in zip(effective.text_regions,['A.','B.','C.','D.'])}
    outputs,consumed,_=assemble(groups,texts,values,[])
    assert list(outputs.values())==['A. $x=1$ B. $x=1$\n\nC. $x=1$ D. $x=1$']
    assert len(consumed)==len(set(consumed))==8


def test_failed_formula_remains_visible_to_text(tmp_path):
    im=Image.new('RGB',(100,30),'black')
    p=region(tmp_path,'p','text',(0,0,100,30),im)
    f=region(tmp_path,'f','formula',(20,0,50,30))
    plan=prepare_text_inputs(build_visual_serial_plan([p,f],policy=api_policy()),{'f':''},tmp_path)
    effective,groups,_,_=prepare_slots(plan,{'f':''},tmp_path)
    assert not groups and len(effective.text_regions)==1
    with Image.open(effective.text_regions[0].crop['path']) as image:
        assert image.getpixel((25,10))==(0,0,0)


def test_containment_requires_same_page_and_output_coverage(tmp_path):
    outer=region(tmp_path,'o','formula',(0,0,200,40))
    child=region(tmp_path,'c','formula',(30,0,80,40))
    assert contained_results([outer,child],{'o':'x=1','c':'y=2'})=={}
    assert contained_results([outer,replace(child,page_index=1)],{'o':'x=1','c':'x=1'})=={}


def test_caption_never_crosses_columns_or_chooses_ambiguous_owner(tmp_path):
    from bemarkdown.pdf.spatial_assembly import caption_owners
    a=region(tmp_path,'a','chart',(0,0,100,80))
    b=region(tmp_path,'b','chart',(120,0,220,80))
    c=region(tmp_path,'c','figure_title',(140,85,160,100))
    assert caption_owners([a,b,c],{'c':'B'})=={'b':['c']}
    assert caption_owners([a,b,replace(b,region_id='other'),c],{'c':'B'})=={}


def test_clipped_figure_edge_restored_only_without_competing_region(tmp_path):
    from bemarkdown.pdf.spatial_assembly import complete_figure_edges
    from types import SimpleNamespace
    from io import BytesIO
    image=Image.new('RGB',(100,100),'white');ImageDraw.Draw(image).rectangle((10,45,20,55),fill='black')
    data=BytesIO();image.save(data,format='PNG')
    a=region(tmp_path,'a','chart',(0,0,80,50),image.crop((0,0,80,50)))
    plan=build_visual_serial_plan([a],policy=api_policy())
    result=complete_figure_edges(plan,[SimpleNamespace(image_bytes=data.getvalue())],tmp_path)
    assert result.preserved_regions[0].bbox==(0,0,80,58)
    b=region(tmp_path,'b','text',(5,50,30,70))
    plan=build_visual_serial_plan([a,b],policy=api_policy())
    assert complete_figure_edges(plan,[SimpleNamespace(image_bytes=data.getvalue())],tmp_path).preserved_regions[0].bbox==a.bbox


def test_duplicate_text_uses_ink_not_white_box_margins(tmp_path):
    from bemarkdown.pdf.spatial_assembly import duplicate_slot_text
    im=Image.new('RGB',(100,40),'white');ImageDraw.Draw(im).rectangle((10,10,60,25),fill='black')
    original=region(tmp_path,'original','text',(0,0,100,40),im)
    cut=im.crop((0,8,100,28));padded=Image.new('RGB',(112,32),'white');padded.paste(cut,(6,6))
    slot=replace(region(tmp_path,'slot','text',(0,8,100,28),padded),metadata={'assembly_parent':'p'})
    assert duplicate_slot_text([original,slot],{'original':'题干文字','slot':'题干文字'})[0]['region_id']=='slot'
    moved=replace(slot,bbox=(200,8,300,28))
    assert duplicate_slot_text([original,moved],{'original':'题干文字','slot':'题干文字'})==[]
