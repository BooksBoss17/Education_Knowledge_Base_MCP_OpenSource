"""Source-only routing signals. A risk flag is not a proposed correction."""
from pathlib import Path
import hashlib
import math
from PIL import Image, ImageChops, ImageDraw

VERSION = 'visible-line-coverage-risk-v1'


def source_coverage_risks(image, evidence):
    if hashlib.sha256(Path(image['path']).read_bytes()).hexdigest() != image['sha256'] or evidence.get('input_sha256') != image['sha256']:
        raise ValueError('SOURCE_COVERAGE_ACTUAL_INPUT_CHANGED')
    lines=evidence.get('pp_lines', [])
    if not lines:
        return {'version':VERSION,'input_sha256':image['sha256'],'reasons':[],'geometry_available':False}
    with Image.open(image['path']) as original:
        if original.width*original.height>32_000_000:
            return {'version':VERSION,'input_sha256':image['sha256'],'reasons':['SOURCE_COVERAGE_PIXEL_WINDOW_UNMEASURED'],'geometry_available':False}
        gray=original.convert('L')
    mask=Image.new('L',gray.size);draw=ImageDraw.Draw(mask)
    for line in lines:
        box=line.get('bbox_local_px')
        if not isinstance(box,(list,tuple)) or len(box)!=4 or any(
                not isinstance(v,(int,float)) or not math.isfinite(v) for v in box):
            return {'version':VERSION,'input_sha256':image['sha256'],'reasons':[],'geometry_available':False}
        left,top,right,bottom=box
        if not (0<=left<right<=gray.width and 0<=top<bottom<=gray.height):
            return {'version':VERSION,'input_sha256':image['sha256'],'reasons':['SOURCE_COVERAGE_LINE_OUTSIDE_INPUT'],'geometry_available':False}
        draw.rectangle((max(0,math.floor(left)-2),max(0,math.floor(top)-2),
            min(gray.width-1,math.ceil(right)+2),min(gray.height-1,math.ceil(bottom)+2)),fill=255)
    ink=gray.point(lambda p:255 if p<200 else 0)
    residual=ImageChops.subtract(ink,mask)
    bounds=residual.getbbox()
    return {'version':VERSION,'geometry_available':True,
        'reasons':['SOURCE_VISIBLE_INK_OUTSIDE_TEXT_LINES'] if bounds else [],
        'uncovered_ink_bbox_px':list(bounds) if bounds else None,
        'input_sha256':image['sha256'],'not_transcription_correctness_proof':True}
