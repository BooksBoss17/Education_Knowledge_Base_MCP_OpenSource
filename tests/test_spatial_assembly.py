from dataclasses import replace
import hashlib
from pathlib import Path
import pytest
from PIL import Image, ImageDraw
from bemarkdown.pdf.spatial_assembly import (
    deduplicate, prepare_slots, assemble, contained_results, merge_continuous_text_rows,
)
from bemarkdown.pdf.visual_serial_pipeline import (
    VisualRegion, VisualSerialExecutor, build_visual_serial_plan, prepare_text_inputs,
)
from bemarkdown.pdf.local_model_api import api_policy


def region(tmp_path, name, label, box, image=None):
    image = image or Image.new('RGB',(box[2]-box[0],box[3]-box[1]),'white')
    path=tmp_path/(name+'.png');image.save(path)
    return VisualRegion(name,label,box,{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


def line_merge_plan(tmp_path, kinds=('formula','text','text'), *, label='text', metadata=None,
                    pitch=22, table=False, prepared=True):
    height=10+pitch*(len(kinds)-1)+22
    image=Image.new('RGB',(120,height),'white');draw=ImageDraw.Draw(image)
    formulas=[]
    for index,kind in enumerate(kinds):
        y=10+index*pitch
        if kind=='formula':
            formulas.append(region(tmp_path,f'f{len(formulas)}','formula',(55,y,95,y+12)))
            draw.rectangle((55,y,94,y+11),fill='black')
        else:
            draw.rectangle((5,y,45,y+11),fill='black')
    parent=replace(region(tmp_path,'p',label,(0,0,120,height),image),metadata=metadata or {})
    regions=[parent,*formulas]
    if table:regions.append(region(tmp_path,'table','table',(0,0,8,height)))
    values={f.region_id:f'x={i}' for i,f in enumerate(formulas)}
    plan=build_visual_serial_plan(regions,policy=api_policy(table_enabled=table))
    if prepared:plan=prepare_text_inputs(plan,values,tmp_path/'masked')
    return plan,values


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


@pytest.mark.parametrize('scale', [1, 2])
def test_masked_formula_overhang_uses_unique_source_parent(tmp_path, scale):
    image = Image.new('RGB', (180*scale, 70*scale), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((5*scale, 10*scale, 170*scale, 22*scale), fill='black')
    draw.rectangle((5*scale, 45*scale, 65*scale, 57*scale), fill='black')
    draw.rectangle((110*scale, 45*scale, 170*scale, 57*scale), fill='black')
    parent = replace(region(tmp_path, 'parent', 'text',
                            tuple(value*scale for value in (0, 0, 180, 67)), image),
                     reading_order=0)
    # The formula extends three pixels below the text box at the base scale.
    # Masking is a candidate clue; body coverage and same-line prose establish ownership.
    formula = replace(region(tmp_path, 'formula', 'formula',
                             tuple(value*scale for value in (70, 43, 105, 70))),
                      reading_order=1)
    values = {'formula': 'f=1/T'}
    plan = build_visual_serial_plan([parent, formula], policy=api_policy())
    plan = prepare_text_inputs(plan, values, tmp_path)
    effective, groups, aliases, reviews = prepare_slots(plan, values, tmp_path)
    assert not aliases and not reviews
    assert len(groups) == 1
    rows = groups[0]['rows']
    row = next(row for row in rows if ['formula', 'formula'] in row)
    assert [slot[0] for slot in row] == ['text', 'formula', 'text']
    text_ids = [slot[1] for slot in row if slot[0] == 'text']
    outputs, consumed, _ = assemble(groups, {text_ids[0]: '即', text_ids[1]: '，所以'}, values, [])
    assert outputs[groups[0]['parent'].region_id].endswith('即 $f=1/T$ ，所以')
    assert consumed.count('formula') == 1


def large_overhang_plan(tmp_path, scale=1, *, intervening_text=True):
    image = Image.new('RGB', (180*scale, 50*scale), 'white')
    ImageDraw.Draw(image).rectangle((5*scale, 12*scale, 175*scale, 25*scale), fill='black')
    parent = replace(region(tmp_path, 'previous-question', 'text',
                            tuple(value*scale for value in (0, 0, 180, 50)), image),
                     reading_order=0)
    formula = replace(region(tmp_path, 'next-question-formula', 'formula',
                             tuple(value*scale for value in (70, 49, 105, 140))),
                      reading_order=2 if intervening_text else 1)
    regions = [parent]
    if intervening_text:
        next_text = replace(region(tmp_path, 'next-question-text', 'text',
                                   tuple(value*scale for value in (0, 70, 60, 90))),
                            reading_order=1)
        regions.append(next_text)
    regions.append(formula)
    values = {'next-question-formula': 'x=99'}
    plan = prepare_text_inputs(build_visual_serial_plan(regions, policy=api_policy()), values, tmp_path)
    return plan, values


@pytest.mark.parametrize('scale', [1, 2])
def test_large_overhang_with_next_question_preserves_final_output_order(tmp_path, scale):
    from bemarkdown.pdf.visual_package import _write_visual_package

    plan, values = large_overhang_plan(tmp_path, scale)
    effective, groups, _, _ = prepare_slots(plan, values, tmp_path)
    texts = {r.region_id: ('下一题' if r.region_id == 'next-question-text' else '上一题')
             for r in effective.text_regions}
    assembled, consumed, provenance = assemble(groups, texts, values, [])
    stage_result = {'formula': values, 'text': texts, 'assembled_text': assembled,
                    'assembly_consumed_ids': consumed, 'spatial_assembly': provenance,
                    'stage_trace': [], 'formula_markdown_ids': []}
    output = tmp_path / 'package'
    _write_visual_package(payload={'policy': effective.policy.to_dict(), 'plan': effective.to_dict()},
                          plan=effective, stage_result=stage_result,
                          source=tmp_path / 'previous-question.png', output_dir=output,
                          document_id='geometry', renders=[])
    markdown = (output / 'document.md').read_text(encoding='utf-8')
    assert markdown.index('上一题') < markdown.index('下一题') < markdown.index('x=99')
    assert markdown.count('x=99') == 1
    assert 'next-question-formula' not in consumed


@pytest.mark.parametrize("other_page", [0, 1])
def test_intervening_question_blocks_only_on_formula_page(tmp_path, other_page):
    from bemarkdown.pdf.visual_package import _write_visual_package

    image = Image.new("RGB", (180, 67), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 45, 65, 57), fill="black")
    draw.rectangle((110, 45, 170, 57), fill="black")
    parent = replace(region(tmp_path, "parent", "text", (0, 0, 180, 67), image), reading_order=0)
    question = replace(region(tmp_path, "question", "text", (0, 90, 60, 110)),
                       reading_order=1, page_index=other_page)
    formula = replace(region(tmp_path, "inline", "formula", (70, 43, 105, 70)), reading_order=2)
    following = replace(region(tmp_path, "display", "formula", (70, 125, 105, 145)), reading_order=3)
    values = {"inline": "f=1/T", "display": "x=99"}
    plan = build_visual_serial_plan([parent, question, formula, following], policy=api_policy())
    plan = prepare_text_inputs(plan, values, tmp_path)
    effective, groups, _, _ = prepare_slots(plan, values, tmp_path)
    texts = {}
    for item in effective.text_regions:
        if item.region_id == "question":
            texts[item.region_id] = "下一题"
        else:
            texts[item.region_id] = "所以" if item.bbox[0] >= formula.bbox[2] else "互为倒数"
    assembled, consumed, provenance = assemble(groups, texts, values, [])
    output = tmp_path / "package"
    _write_visual_package(
        payload={"policy": effective.policy.to_dict(), "plan": effective.to_dict()}, plan=effective,
        stage_result={"formula": values, "text": texts, "assembled_text": assembled,
                      "assembly_consumed_ids": consumed, "spatial_assembly": provenance, "stage_trace": []},
        source=tmp_path / "parent.png", output_dir=output, document_id="question-order", renders=[])
    markdown = (output / "document.md").read_text(encoding="utf-8")
    assert markdown.count("f=1/T") == markdown.count("x=99") == 1
    assert markdown.index("f=1/T") < markdown.index("x=99")
    if other_page == 0:
        assert markdown.index("互为倒数") < markdown.index("下一题") < markdown.index("f=1/T")
        assert "inline" not in consumed
    else:
        assert "互为倒数 $f=1/T$ 所以" in markdown
        assert markdown.index("x=99") < markdown.index("下一题")
        assert "inline" in consumed


def test_large_overhang_without_intervening_text_is_not_absorbed(tmp_path):
    plan, values = large_overhang_plan(tmp_path, intervening_text=False)
    _, groups, _, _ = prepare_slots(plan, values, tmp_path)
    assert groups == []


def test_small_overhang_with_ambiguous_reading_order_is_not_absorbed(tmp_path):
    image = Image.new('RGB', (180, 70), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 45, 65, 57), fill='black')
    draw.rectangle((110, 45, 170, 57), fill='black')
    parent = region(tmp_path, 'parent', 'text', (0, 0, 180, 67), image)
    formula = region(tmp_path, 'formula', 'formula', (70, 43, 105, 70))
    values = {'formula': 'x=1'}
    plan = prepare_text_inputs(build_visual_serial_plan([parent, formula], policy=api_policy()), values, tmp_path)
    _, groups, _, _ = prepare_slots(plan, values, tmp_path)
    assert groups == []


def test_overhang_requires_adjacent_text_on_the_same_line(tmp_path):
    image = Image.new('RGB', (180, 110), 'white')
    ImageDraw.Draw(image).rectangle((5, 10, 170, 20), fill='black')
    parent = replace(region(tmp_path, 'parent', 'text', (0, 0, 180, 100), image), reading_order=0)
    formula = replace(region(tmp_path, 'formula', 'formula', (70, 60, 105, 110)), reading_order=1)
    values = {'formula': 'x=1'}
    plan = prepare_text_inputs(build_visual_serial_plan([parent, formula], policy=api_policy()), values, tmp_path)
    _, groups, _, _ = prepare_slots(plan, values, tmp_path)
    assert groups == []


def test_overhang_is_bounded_by_adjacent_text_height(tmp_path):
    image = Image.new('RGB', (180, 120), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 50, 65, 60), fill='black')
    draw.rectangle((110, 50, 170, 60), fill='black')
    parent = replace(region(tmp_path, 'parent', 'text', (0, 0, 180, 100), image), reading_order=0)
    formula = replace(region(tmp_path, 'formula', 'formula', (70, 0, 105, 120)), reading_order=1)
    values = {'formula': 'x=1'}
    plan = prepare_text_inputs(build_visual_serial_plan([parent, formula], policy=api_policy()), values, tmp_path)
    _, groups, _, _ = prepare_slots(plan, values, tmp_path)
    assert groups == []


def test_masked_formula_top_overhang_keeps_prior_unassembled_path(tmp_path):
    image = Image.new('RGB', (180, 70), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 45, 65, 57), fill='black')
    draw.rectangle((110, 45, 170, 57), fill='black')
    parent = region(tmp_path, 'parent', 'text', (0, 3, 180, 70), image)
    formula = region(tmp_path, 'formula', 'formula', (70, 0, 105, 50))
    values = {'formula': 'x=1'}
    plan = prepare_text_inputs(build_visual_serial_plan([parent, formula], policy=api_policy()), values, tmp_path)
    effective, groups, aliases, _ = prepare_slots(plan, values, tmp_path)
    assert len(effective.text_regions) == 1
    assert groups == [] and aliases == {}


def test_formula_shared_by_multiple_masked_columns_remains_unassembled(tmp_path):
    left_image = Image.new('RGB', (100, 40), 'white')
    right_image = Image.new('RGB', (100, 40), 'white')
    ImageDraw.Draw(left_image).rectangle((5, 10, 60, 20), fill='black')
    ImageDraw.Draw(right_image).rectangle((40, 10, 95, 20), fill='black')
    left = region(tmp_path, 'left', 'text', (0, 0, 100, 40), left_image)
    right = region(tmp_path, 'right', 'text', (120, 0, 220, 40), right_image)
    formula = region(tmp_path, 'shared', 'formula', (80, 8, 140, 32))
    values = {'shared': 'x=1'}
    plan = prepare_text_inputs(build_visual_serial_plan([left, formula, right], policy=api_policy()), values, tmp_path)
    effective, groups, aliases, _ = prepare_slots(plan, values, tmp_path)
    assert len(effective.text_regions) == 2
    assert groups == [] and aliases == {}


def test_intentional_repeated_formulas_at_distinct_sources_are_not_suppressed(tmp_path):
    first = region(tmp_path, 'first', 'formula', (0, 0, 50, 20))
    second = region(tmp_path, 'second', 'formula', (100, 0, 150, 20))
    assert contained_results([first, second], {'first': 'x=1', 'second': 'x=1'}) == {}


def test_unmasked_formula_from_next_question_is_not_absorbed(tmp_path):
    image = Image.new('RGB', (100, 20), 'white')
    ImageDraw.Draw(image).rectangle((5, 5, 80, 14), fill='black')
    parent = region(tmp_path, 'question', 'text', (0, 0, 100, 20), image)
    next_formula = region(tmp_path, 'next', 'formula', (20, 35, 80, 50))
    values = {'next': 'x=2'}
    plan = prepare_text_inputs(build_visual_serial_plan([parent, next_formula], policy=api_policy()), values, tmp_path)
    effective, groups, aliases, _ = prepare_slots(plan, values, tmp_path)
    assert len(effective.text_regions) == 1
    assert groups == [] and aliases == {}


def test_formula_companion_prose_is_preserved_without_double_wrapping(tmp_path):
    parent = region(tmp_path, 'parent', 'text', (0, 0, 100, 20))
    groups = [{'parent': parent, 'rows': [[['text', 'before'], ['formula', 'f'], ['text', 'after']]]}]
    outputs, consumed, _ = assemble(groups, {'before': '前', 'after': '后'},
                                    {'f': '文字 $x=1$ 成立'}, ['f'])
    assert outputs['parent'] == '前 文字 $x=1$ 成立 后'
    assert consumed == ['before', 'f', 'after']


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


def test_line_merge_default_off_preserves_original_slots_and_plan(tmp_path):
    raw,values=line_merge_plan(tmp_path,prepared=False)
    prepared=prepare_text_inputs(raw,values,tmp_path)
    original,groups,_,_=prepare_slots(prepared,values,tmp_path)
    paths=[r.crop['path'] for r in original.text_regions]
    contents=[Path(path).read_bytes() for path in paths]
    class Runtime:
        def __init__(self,value):self.value=value
        def recognize_batch(self,crops):return [self.value for _ in crops]
    executor=VisualSerialExecutor(formula_runtime=Runtime('x=1'),text_runtime=Runtime('text'),policy=api_policy())
    result=executor.execute(raw,output_dir=tmp_path)
    assert executor.effective_plan.to_dict()==original.to_dict()
    assert [Path(path).read_bytes() for path in paths]==contents
    assert result['spatial_assembly'][0]['rows']==groups[0]['rows']
    assert result['line_merge']=={'enabled':False,'merged_blocks':0,'source_rows':0}


def test_line_merge_falls_back_before_text_formula_text_row(tmp_path):
    parent = region(tmp_path, 'parent', 'text', (0, 0, 120, 100))
    fragments = []
    for name, box in (('a', (5, 5, 45, 17)), ('b', (5, 27, 45, 39)),
                      ('c', (5, 49, 15, 61)), ('d', (70, 49, 110, 61))):
        padded = Image.new('RGB', (52, 24), 'white')
        padded.paste('black', (6, 6, 46, 18))
        fragments.append(replace(region(tmp_path, name, 'text', box, padded),
                                 metadata={'assembly_parent': 'parent'}))
    formula = region(tmp_path, 'formula', 'formula', (30, 49, 65, 61))
    rows = [[['text', 'a']], [['text', 'b']],
            [['text', 'c'], ['formula', 'formula'], ['text', 'd']]]
    actual_rows, actual_fragments, records = merge_continuous_text_rows(
        parent, rows, fragments, tmp_path / 'merged')
    assert actual_rows == rows
    assert actual_fragments == fragments
    assert records == []


def test_line_merge_preserves_source_provenance_and_new_hash(tmp_path):
    plan,values=line_merge_plan(tmp_path)
    original,_,_,_=prepare_slots(plan,values,tmp_path/'original')
    effective,groups,_,_=prepare_slots(plan,values,tmp_path/'merged',line_merge=True)
    originals={r.region_id:r for r in original.text_regions}
    merged,=effective.text_regions
    record,=groups[0]['line_merges'];parent=groups[0]['parent']
    assert len(originals)==2
    assert record['merged_region_id']==merged.region_id
    assert record['parent_region_id']==parent.region_id and record['page_index']==parent.page_index
    assert record['bbox']==list(merged.bbox)
    assert merged.bbox==(min(r.bbox[0] for r in originals.values()),min(r.bbox[1] for r in originals.values()),
                         max(r.bbox[2] for r in originals.values()),max(r.bbox[3] for r in originals.values()))
    assert merged.metadata['assembly_parent']==parent.region_id
    assert merged.metadata['line_merge_sources']==list(originals)
    for source in record['sources']:
        old=originals[source['region_id']]
        assert source=={'region_id':old.region_id,'bbox':list(old.bbox),'page_index':old.page_index,
                        'parent_region_id':parent.region_id,'source_crop_sha256':old.crop['sha256']}
    digest=hashlib.sha256(Path(merged.crop['path']).read_bytes()).hexdigest()
    assert merged.crop['sha256']==record['input_sha256']==digest
    assert digest not in {r.crop['sha256'] for r in originals.values()}


def test_line_merge_never_crosses_formula_rows_and_assembles_in_order(tmp_path):
    kinds=('formula','text','text','formula','text','text','formula','text')
    plan,values=line_merge_plan(tmp_path,kinds)
    _,original_groups,_,_=prepare_slots(plan,values,tmp_path/'original')
    _,groups,_,_=prepare_slots(plan,values,tmp_path/'merged',line_merge=True)
    old_ids=[row[0][1] for row in original_groups[0]['rows']]
    merges=groups[0]['line_merges']
    assert [[s['region_id'] for s in m['sources']] for m in merges]==[[old_ids[1],old_ids[2]],[old_ids[4],old_ids[5]]]
    merged_ids=[m['merged_region_id'] for m in merges]
    texts={merged_ids[0]:'A B',merged_ids[1]:'C D',old_ids[-1]:'E'}
    outputs,consumed,_=assemble(groups,texts,values,[])
    assert consumed==['f0',merged_ids[0],'f1',merged_ids[1],'f2',old_ids[-1]]
    assert outputs[groups[0]['parent'].region_id]=='$x=0$\n\nA B\n\n$x=1$\n\nC D\n\n$x=2$\n\nE'


@pytest.mark.parametrize('options',[
    {'label':'paragraph_title'},
    {'label':'figure_title'},
    {'pitch':40},
    {'metadata':{'review_reason':'SOURCE_REVIEW'}},
],ids=['title','caption','vertical-gap','review-metadata'])
def test_line_merge_rejects_unsafe_text_rows(tmp_path,options):
    plan,values=line_merge_plan(tmp_path,**options)
    _,groups,_,_=prepare_slots(plan,values,tmp_path/'slots',line_merge=True)
    assert [row[0][0] for row in groups[0]['rows']]==['formula','text','text']
    assert groups[0]['line_merges']==[]


def test_line_merge_rejects_nonoverlapping_columns(tmp_path):
    parent=region(tmp_path,'p','text',(0,0,120,60))
    slots=[]
    for name,box in (('a',(5,5,45,17)),('b',(70,27,110,39))):
        padded=Image.new('RGB',(52,24),'white');padded.paste('black',(6,6,46,18))
        slots.append(replace(region(tmp_path,name,'text',box,padded),metadata={'assembly_parent':'p'}))
    original=[[["text","a"]],[["text","b"]]]
    rows,regions,records=merge_continuous_text_rows(parent,original,slots,tmp_path/'merged')
    assert rows==original and regions==slots and records==[]


def test_line_merge_rejects_table_overlap(tmp_path):
    plan,values=line_merge_plan(tmp_path,table=True)
    effective,groups,_,reviews=prepare_slots(plan,values,tmp_path/'slots',line_merge=True)
    assert groups==[] and len(effective.text_regions)==1
    assert reviews==[{'region_id':effective.text_regions[0].region_id,'reason':'SPATIAL_TABLE_PARENT_REVIEW'}]


def test_line_merge_pixels_come_only_from_retained_slots(tmp_path):
    source=Image.new('RGB',(60,45),'white');draw=ImageDraw.Draw(source)
    draw.rectangle((5,5,45,16),fill='black');draw.rectangle((5,27,45,38),fill='black')
    for x,color in ((10,'red'),(20,'green'),(30,'blue')):  # formula, filtered and foreign ink
        draw.rectangle((x,20,x+2,22),fill=color)
    parent=region(tmp_path,'parent','text',(0,0,60,45),source)
    slots=[]
    for name,box in (('a',(5,5,46,17)),('b',(5,27,46,39))):
        padded=Image.new('RGB',(53,24),'white');padded.paste(source.crop(box),(6,6))
        slots.append(replace(region(tmp_path,name,'text',box,padded),metadata={'assembly_parent':'parent'}))
    _,additions,records=merge_continuous_text_rows(
        parent,[[["text","a"]],[["text","b"]]],slots,tmp_path/'merged')
    assert len(records)==1
    with Image.open(additions[0].crop['path']) as image:
        actual=image.crop((6,6,47,40))
    expected=Image.new('RGB',(41,34),'white')
    expected.paste('black',(0,0,41,12));expected.paste('black',(0,22,41,34))
    assert actual.tobytes()==expected.tobytes()


def test_executor_empty_merged_ocr_keeps_existing_review_and_assembly_behavior(tmp_path):
    raw,_=line_merge_plan(tmp_path,prepared=False)
    class Formula:
        def recognize_batch(self,crops):return ['x=1' for _ in crops]
    class Text:
        spec={'text_line_merge':'on'}
        def recognize_batch(self,crops):
            assert len(crops)==1
            return ['']
    executor=VisualSerialExecutor(formula_runtime=Formula(),text_runtime=Text(),policy=api_policy())
    result=executor.execute(raw,output_dir=tmp_path)
    merged,=executor.effective_plan.text_regions
    parent_id=result['spatial_assembly'][0]['parent']['region_id']
    reviews={(r['region_id'],r['reason']) for r in result['review_items']}
    assert result['line_merge']=={'enabled':True,'merged_blocks':1,'source_rows':2}
    assert result['text']=={merged.region_id:''}
    assert result['assembled_text'][parent_id]=='$x=1$\n\n'
    assert result['assembly_consumed_ids']==['f0',merged.region_id]
    assert (merged.region_id,'EMPTY_OCR') in reviews
    assert (parent_id,'ASSEMBLY_TEXT_FRAGMENT_EMPTY_SOURCE_PRESERVED') in reviews


def test_nested_unit_formatting_rounding_does_not_merge_conflicting_units(tmp_path):
    parent = region(tmp_path, 'combined', 'formula', (0, 1, 200, 40))
    child = region(tmp_path, 'first', 'formula', (2, 0, 80, 39))
    separate = region(tmp_path, 'separate', 'formula', (2, 90, 80, 130))
    values = {'combined': r'm_1=12~\mathrm{kg},m_2=15~\mathrm{kg}',
              'first': r'm_1=12\text{ kg}', 'separate': r'm_1=12\text{ kg}'}
    assert contained_results([parent, child, separate], values) == {'first':'combined'}
    values['first'] = r'm_1=12\mathrm{g}'
    assert contained_results([parent, child, separate], values) == {}


def test_unrelated_score_on_different_line_preserves_independent_blocks(tmp_path):
    from bemarkdown.pdf.visual_package import _join_score_rows
    answer = region(tmp_path, 'answer', 'formula', (0, 0, 150, 36))
    score = region(tmp_path, 'score', 'formula_number', (220, 40, 270, 70))
    blocks = ['$$\nx=12\n$$', '(2分)']
    _join_score_rows(blocks, [answer, score], {'answer':0, 'score':1}, {'answer':'x=12'})
    assert blocks == ['$$\nx=12\n$$', '(2分)']


def test_touching_tall_formulas_on_successive_lines_keep_full_text_rows(tmp_path):
    image = Image.new('RGB',(320,110),'white')
    draw = ImageDraw.Draw(image)
    for top in (15,75):
        draw.rectangle((5,top,300,top+15),fill='black')
    parent = region(tmp_path,'parent','text',(0,0,320,110),image)
    upper = region(tmp_path,'upper','formula',(80,0,130,56))
    lower = region(tmp_path,'lower','formula',(180,53,220,105))
    values = {'upper':'a=1', 'lower':'b=2'}
    plan = build_visual_serial_plan([parent,upper,lower],policy=api_policy())
    plan = prepare_text_inputs(plan,values,tmp_path)
    effective,groups,_,_ = prepare_slots(plan,values,tmp_path)
    rows = groups[0]['rows']
    assert [[rid for kind,rid in row if kind=='formula'] for row in rows]==[['upper'],['lower']]
    assert all(not (t.bbox[1] < 45 and t.bbox[3] > 65) for t in effective.text_regions)


def test_score_after_a_duplicate_lead_in_follows_one_formula(tmp_path):
    from bemarkdown.pdf.visual_package import _join_score_rows
    meta = {'masked_formula_region_ids':['distance']}
    wide = replace(region(tmp_path,'wide','text',(0,10,280,45)),metadata=meta)
    narrow = replace(region(tmp_path,'short','text',(0,9,140,46)),metadata=meta)
    formula = region(tmp_path,'distance','formula',(65,10,135,43))
    blocks = ['得： (1分)','得：','$$\nd=2r\n$$']
    _join_score_rows(blocks,[wide,narrow,formula],{'wide':0,'short':1,'distance':2},{'distance':'d=2r'})
    assert [b for b in blocks if b] == ['得： $d=2r$ (1分)']
