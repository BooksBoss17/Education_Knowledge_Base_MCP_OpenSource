"""Table ownership and bounded HTML validation; no inference dependencies."""
from __future__ import annotations
import re


def contains(outer, inner):
    a,b,c,d=outer.bbox; x,y,z,w=inner.bbox
    return outer.page_index == inner.page_index and a <= x and b <= y and c >= z and d >= w


def table_ownership(regions):
    tables=[]
    for row in sorted((r for r in regions if r.normalized_label=='table'),
                      key=lambda r: (-(r.bbox[2]-r.bbox[0])*(r.bbox[3]-r.bbox[1]),r.reading_order,r.region_id)):
        if not any(contains(t,row) for t in tables):tables.append(row)
    owned={r.region_id:t.region_id for r in regions for t in tables
           if r.region_id!=t.region_id and contains(t,r)}
    return tuple(sorted(tables,key=lambda r:(r.page_index,r.reading_order,r.region_id))),owned


def normalize_table(value):
    """Accept one inert, structurally bounded HTML grid; keep raw on failure."""
    from lxml import etree
    if not isinstance(value,str) or not value.strip():return '', 'EMPTY_TABLE_OUTPUT'
    value=re.sub(r'^```(?:html)?\s*|\s*```$','',value.strip(),flags=re.I).strip()
    # XML parser rejects implicit repairs and external entities. Require explicit
    # HTML closing tags so a token-truncated table is never treated as complete.
    try:
        parser=etree.XMLParser(resolve_entities=False,no_network=True,recover=False)
        table=etree.fromstring(value.encode('utf-8'),parser)
        if table.tag!='table' or table.getroottree().docinfo.doctype:raise ValueError()
        # OCR commonly emits this inert presentation hint despite the prompt.
        # Remove it without altering any cell or accepting arbitrary attributes.
        for hint in ('border','cellspacing','cellpadding'):
            if table.get(hint) is not None:
                if not re.fullmatch(r'\d{1,2}',table.get(hint)):raise ValueError()
                del table.attrib[hint]
        allowed={'table','thead','tbody','tfoot','tr','th','td','br','sup','sub'}
        for node in table.iter():
            if node.tag not in allowed:raise ValueError()
            if set(node.attrib)-({'rowspan','colspan'} if node.tag in {'td','th'} else set()):raise ValueError()
            for span in node.attrib.values():
                if not span.isdigit() or not 1 <= int(span) <= 1000:raise ValueError()
            parent=node.getparent()
            if node.tag=='tr' and parent.tag not in {'table','thead','tbody','tfoot'}:raise ValueError()
            if node.tag in {'td','th'} and parent.tag!='tr':raise ValueError()
        rows=table.findall('.//tr')
        if not rows or any(not any(c.tag in {'td','th'} for c in row) for row in rows):raise ValueError()
        if not ''.join(table.itertext()).strip():raise ValueError()
        # Reject overlapping spans and incomplete rectangular grids.
        occupied=set(); widths=[]
        for y,row in enumerate(rows):
            x=0
            for cell in row:
                if cell.tag not in {'td','th'}:raise ValueError()
                while (y,x) in occupied:x+=1
                height=int(cell.get('rowspan','1'));width=int(cell.get('colspan','1'))
                if y+height>len(rows) or x+width>1000:raise ValueError()
                for yy in range(y,y+height):
                    for xx in range(x,x+width):
                        if (yy,xx) in occupied:raise ValueError()
                        occupied.add((yy,xx))
                x+=width
            widths.append(max((xx+1 for yy,xx in occupied if yy==y),default=0))
        if len(set(widths))!=1 or len(occupied)!=len(rows)*widths[0]:raise ValueError()
        return etree.tostring(table,encoding='unicode',method='html'),None
    except (etree.LxmlError,ValueError,TypeError,AttributeError):
        return '', 'TABLE_STRUCTURE_INVALID_SOURCE_PRESERVED'
