from pathlib import Path
from io import BytesIO
import json
import pytest
from PIL import Image
from lxml import etree
from bemarkdown.native_api import word_picture_input
from bemarkdown.next_runtime import convert_document
from bemarkdown.package_io import validate_package

def owner(crop='', transform='', extent=''):
    return etree.fromstring(f"""<w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"
        xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><pic:pic>
        <pic:blipFill><a:srcRect {crop}/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
        <pic:spPr><a:xfrm {transform}>{extent}</a:xfrm><a:prstGeom prst="rect"/></pic:spPr></pic:pic></w:drawing>""")

@pytest.mark.parametrize('transform',['','flipH="1"','flipV="true"','rot="5400000"','flipH="1" rot="5400000"'])
def test_hidden_half_never_enters_visible_crop(tmp_path,transform):
    image=Image.new('RGB',(40,20),'red')
    image.paste('blue',(20,0,40,10));image.paste('green',(20,10,40,20))
    source=tmp_path/'original.png';image.save(source)
    result,receipt=word_picture_input(source,[owner('l="50000"',transform)],tmp_path,0)
    crop=Image.open(result)
    assert (255,0,0) not in list(crop.get_flattened_data())
    reference=image.crop((20,0,40,20))
    if 'flipH' in transform:reference=reference.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if 'flipV' in transform:reference=reference.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if 'rot' in transform:reference=reference.transpose(Image.Transpose.ROTATE_270)
    assert crop.tobytes()==reference.tobytes()
    assert Image.open(source).tobytes()==image.tobytes()

def test_padding_display_ratio_and_distinct_occurrences(tmp_path):
    p=tmp_path/'source.png';Image.new('RGB',(40,20),'blue').save(p)
    a,_=word_picture_input(p,[owner('l="-50000"')],tmp_path,0)
    assert Image.open(a).getpixel((0,0))==(255,255,255)
    b,_=word_picture_input(p,[owner('l="50000"',extent='<a:ext cx="1" cy="2"/>')],tmp_path,1)
    assert Image.open(b).size==(20,40)
    assert a!=b

def test_invalid_crop_rejected(tmp_path):
    p=tmp_path/'source.png';Image.new('RGB',(40,20),'blue').save(p)
    with pytest.raises(ValueError,match='EMPTY'):
        word_picture_input(p,[owner('l="50000" r="50000"')],tmp_path,0)


def test_group_selects_matching_media_and_applies_group_scale(tmp_path):
    from copy import deepcopy
    group=etree.fromstring("""<w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
        xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
        xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
        <wpg:wgp><wpg:cNvGrpSpPr/><wpg:grpSpPr><a:xfrm><a:ext cx="200" cy="100"/><a:chExt cx="100" cy="100"/></a:xfrm></wpg:grpSpPr></wpg:wgp></w:drawing>""")
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','pic':'http://schemas.openxmlformats.org/drawingml/2006/picture'}
    for rid,x in [('r1',0),('r2',60)]:
        pic=owner('l="50000"',extent=f'<a:off x="{x}" y="0"/><a:ext cx="20" cy="20"/>')[0]
        blip=etree.Element('{'+ns['a']+'}blip')
        blip.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed',rid)
        pic.find('pic:blipFill',ns).insert(0,blip)
        group[0].append(pic)
    path=tmp_path/'source.png';pixels=Image.new('RGB',(40,20),'red');pixels.paste('blue',(20,0,40,20));pixels.save(path)
    result,_=word_picture_input(path,[group],tmp_path,0,'r2')
    visible=Image.open(result)
    assert visible.size==(40,20) and visible.getpixel((0,0))==(0,0,255)
    # An overlapping sibling requires compositing, never reuse the hidden pixels.
    group[0][-1].find('pic:spPr/a:xfrm/a:off',ns).set('x','10')
    with pytest.raises(ValueError,match='OVERLAPPING'):
        word_picture_input(path,[group],tmp_path,1,'r2')

def test_group_composite_contains_both_visible_pictures(tmp_path):
    from zipfile import ZipFile
    from bemarkdown.native_api import word_group_input
    group=etree.fromstring("""<w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup"
        xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
        <wpg:wgp><wpg:cNvGrpSpPr/><wpg:grpSpPr><a:xfrm><a:ext cx="100" cy="20"/><a:chExt cx="100" cy="20"/><a:chOff x="0" y="0"/></a:xfrm></wpg:grpSpPr></wpg:wgp></w:drawing>""")
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','pic':'http://schemas.openxmlformats.org/drawingml/2006/picture'}
    payloads=[]
    for i,color in enumerate(('blue','green')):
        pic=owner('l="50000"',extent=f'<a:off x="{i*60}" y="0"/><a:ext cx="20" cy="20"/>')[0]
        blip=etree.Element('{'+ns['a']+'}blip')
        blip.set('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed',f'r{i}')
        pic.find('pic:blipFill',ns).insert(0,blip);group[0].append(pic)
        image=Image.new('RGB',(40,20),'red');image.paste(color,(20,0,40,20))
        buf=BytesIO();image.save(buf,format='PNG');payloads.append(buf.getvalue())
    doc=tmp_path/'source.docx'
    with ZipFile(doc,'w') as z:
        z.writestr('word/_rels/document.xml.rels','<Relationships>'+''.join(f'<Relationship Id="r{i}" Target="media/{i}.png"/>' for i in range(2))+'</Relationships>')
        for i,data in enumerate(payloads):z.writestr(f'word/media/{i}.png',data)
    path,receipt=word_group_input(doc,'word/document.xml',group,tmp_path,0)
    image=Image.open(path)
    assert image.getpixel((10,10))==(0,0,255)
    assert image.getpixel((70,10))==(0,128,0)
    assert (255,0,0) not in list(image.get_flattened_data())
    assert len(receipt['members'])==2


def test_alphabet_array_review_checks_image_and_keeps_visible_letters(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    from bemarkdown.pdf.visual_content_guard import alphabet_array_without_glyphs
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    array = r'$$\begin{array}{c}' + r'\quad'.join(r'\text{' + c + '}' for c in alphabet) + r'\end{array}$$'
    path = tmp_path/'crop.png'
    image = Image.new('L', (700, 64), 'white')
    ImageDraw.Draw(image).line((5, 48, 695, 48), fill=0, width=2)
    image.save(path)
    assert alphabet_array_without_glyphs(array, {'path':path})
    assert not alphabet_array_without_glyphs('A B C', {'path':path})
    draw = ImageDraw.Draw(image)
    for i, letter in enumerate(alphabet):
        draw.text((i*25, 10),letter,font=ImageFont.load_default(),fill=0)
    image.save(path)
    assert not alphabet_array_without_glyphs(array, {'path':path})
