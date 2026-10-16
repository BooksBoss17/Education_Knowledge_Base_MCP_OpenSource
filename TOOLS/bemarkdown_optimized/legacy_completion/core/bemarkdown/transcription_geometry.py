"""Conservative same-line request grouping with explicit original target boxes.

Render one contiguous source-page rectangle, never a collage. Grouping reduces
image reads, not node count: every target remains an independently bound task.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

GEOMETRY_POLICY='same-source-line-bounded-v2-small-region-readable'


@dataclass(frozen=True)
class GeometryPolicy:
    max_targets: int = 8
    max_gap_in_heights: float = 2.0
    max_width_pt: float = 560.0
    padding_pt: float = 2.0
    render_scale: float = 3.0
    small_region_scale: float = 6.0
    small_region_max_height_pt: float = 32.0
    max_pixels: int = 1048576


def _box(region):
    b=region['bbox']
    if len(b)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in b) or b[0]>=b[2] or b[1]>=b[3]:
        raise ValueError('SOURCE_REGION_BBOX_INVALID')
    return tuple(float(x) for x in b)


def _union(a,b):return (min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))


def group_source_regions(regions,policy=None):
    policy=policy or GeometryPolicy();values=list(regions)
    if len({r['id'] for r in values})!=len(values):raise ValueError('DUPLICATE_SOURCE_REGION_ID')
    if policy.max_targets<1:raise ValueError('INVALID_GROUP_LIMIT')
    groups=[]
    for r in sorted(values,key=lambda r:(r['page_index'],_box(r)[1],_box(r)[0],r['id'])):
        box=_box(r);placed=False
        for group in reversed(groups):
            if group['page_index']!=r['page_index'] or len(group['regions'])>=policy.max_targets:continue
            previous=group['bbox'];h=min(previous[3]-previous[1],box[3]-box[1])
            overlap=min(previous[3],box[3])-max(previous[1],box[1])
            gap=max(0,max(previous[0],box[0])-min(previous[2],box[2]))
            union=_union(previous,box)
            # Subscript-height overlap can be small. Require a shared baseline
            # envelope as well as a bounded gap and height to avoid next lines.
            same_line=(overlap>=.35*h or abs((previous[1]+previous[3]-box[1]-box[3])/2)<=.5*max(previous[3]-previous[1],box[3]-box[1]))
            height_ok=union[3]-union[1]<=1.8*max(previous[3]-previous[1],box[3]-box[1])
            if same_line and height_ok and gap<=policy.max_gap_in_heights*h and union[2]-union[0]<=policy.max_width_pt:
                group['regions'].append(r);group['bbox']=union;placed=True;break
        if not placed:groups.append({'page_index':r['page_index'],'bbox':box,'regions':[r]})
    return groups


def render_source_groups(source,regions,output_root,*,policy=None,grouped=True):
    import pymupdf
    policy=policy or GeometryPolicy();source=Path(source);output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    values=list(regions)
    groups=group_source_regions(values,policy) if grouped else [dict(page_index=r['page_index'],bbox=_box(r),regions=[r]) for r in values]
    bindings=[]
    with pymupdf.open(source) as pdf:
        for index,g in enumerate(groups):
            page=pdf[g['page_index']];b=g['bbox'];p=policy.padding_pt
            clip=pymupdf.Rect(b[0]-p,b[1]-p,b[2]+p,b[3]+p)&page.rect
            if clip.is_empty:raise ValueError('SOURCE_RENDER_EMPTY')
            scale=policy.small_region_scale if clip.height<=policy.small_region_max_height_pt else policy.render_scale
            if math.ceil(clip.width*scale)*math.ceil(clip.height*scale)>policy.max_pixels:
                # No downscaling to hide input budget issues. Split the group.
                if len(g['regions'])==1:raise ValueError('SOURCE_REGION_PIXEL_BUDGET')
                bindings.extend(render_source_groups(source,g['regions'],output_root/f'split-{index}',policy=policy,grouped=False));continue
            image=page.get_pixmap(matrix=pymupdf.Matrix(scale,scale),clip=clip,alpha=False)
            path=output_root/f'g{index:04d}.png';image.save(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            # Pixmap origins include render-grid rounding, not simply clip.x0*s.
            for r in g['regions']:
                target=_box(r)
                pixels=[target[0]*scale-image.x,target[1]*scale-image.y,
                        target[2]*scale-image.x,target[3]*scale-image.y]
                pixels=[max(0,min(float(image.width if i%2==0 else image.height),v)) for i,v in enumerate(pixels)]
                bindings.append({'id':r['id'],'page_index':g['page_index'],'image_path':str(path),'input_sha256':digest,
                    'source_bbox':list(target),'rendered_source_bbox':list(clip),'target_bbox_pixels':pixels,
                    'image_size':[image.width,image.height],'render_scale':scale,'geometry_policy':GEOMETRY_POLICY,'group_ids':[x['id'] for x in g['regions']]})
    assert {b['id'] for b in bindings}=={r['id'] for r in values}
    return bindings
