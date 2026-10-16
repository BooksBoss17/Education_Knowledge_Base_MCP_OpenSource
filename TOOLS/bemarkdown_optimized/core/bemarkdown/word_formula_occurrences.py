"""Read-only native formula emission tracing over the existing Word serializer.

Markers exist only in a shadow rendering. Removing them must reproduce the
entire original Markdown byte-for-byte; otherwise no ranges are certified.
"""
from dataclasses import fields
import hashlib
import re

from . import ir
from .serializer import MarkdownSerializer

VERSION = 'native-word-formula-emission-v1'
CLASSES = {'text': ir.TextNode, 'break': ir.BreakNode, 'formula': ir.FormulaNode,
           'image': ir.ImageNode, 'image_content': ir.ImageContentNode,
           'hyperlink': ir.HyperlinkNode, 'paragraph': ir.ParagraphNode,
           'heading': ir.HeadingNode, 'list_item': ir.ListItemNode, 'table': ir.TableNode}


def load_native_document(value):
    def node(raw):
        cls = CLASSES.get(raw.get('kind'))
        if cls is None:
            raise ValueError('NATIVE_WORD_UNKNOWN_NODE_KIND')
        names = {f.name for f in fields(cls) if f.init}
        if set(raw) - names - {'kind'}:
            raise ValueError('NATIVE_WORD_UNKNOWN_NODE_FIELDS')
        args = {k: v for k, v in raw.items() if k in names}
        if 'children' in args:
            args['children'] = [node(x) for x in args['children']]
        if 'assets' in args:
            args['assets'] = {k: node(x) for k, x in args['assets'].items()}
        if 'rows' in args:
            args['rows'] = [[ir.TableCell(blocks=[node(b) for b in cell['blocks']],
                column=cell['column'], colspan=cell.get('colspan',1), vmerge=cell.get('vmerge'))
                for cell in row] for row in args['rows']]
        return cls(**args)
    return ir.DocumentIR(blocks=[node(b) for b in value['blocks']],
        source_part=value.get('source_part','word/document.xml'), metadata=value.get('metadata',{}))


def trace_native_formula_occurrences(document, *, selected_formula_ids=None):
    original = MarkdownSerializer().serialize(document)
    prefix = 'BMTRACE' + hashlib.sha256(original.encode('utf-8')).hexdigest()[:16] + 'X'
    if prefix in original:
        raise ValueError('NATIVE_WORD_SHADOW_MARKER_COLLISION')
    emitted = []
    class Shadow(MarkdownSerializer):
        def _inline(self, node):
            value = super()._inline(node)
            if isinstance(node, ir.FormulaNode) and (selected_formula_ids is None or node.formula_id in selected_formula_ids):
                ordinal = len(emitted)
                emitted.append(node)
                return prefix + f'BEGIN{ordinal}Z' + value + prefix + f'END{ordinal}Z'
            return value
    shadow = Shadow().serialize(document)
    pattern = re.compile(re.escape(prefix) + r'(BEGIN|END)([0-9]+)Z')
    chunks = []
    positions = {}
    cursor = 0
    byte_offset = 0
    for match in pattern.finditer(shadow):
        chunk = shadow[cursor:match.start()]
        chunks.append(chunk)
        byte_offset += len(chunk.encode('utf-8'))
        key = int(match.group(2))
        positions.setdefault(key, {})[match.group(1)] = byte_offset
        cursor = match.end()
    chunks.append(shadow[cursor:])
    restored = ''.join(chunks)
    if restored != original:
        return {'version': VERSION, 'status': 'SHADOW_SERIALIZATION_CHANGED_WHITESPACE_UNCERTIFIED',
                'byte_equivalent': False, 'occurrences': [],
                'markdown_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest()}
    data = original.encode('utf-8')
    records = []
    for ordinal, source in enumerate(emitted):
        pos = positions.get(ordinal, {})
        if set(pos) != {'BEGIN','END'} or pos['BEGIN'] > pos['END']:
            raise ValueError('NATIVE_WORD_SHADOW_MARKER_PAIR')
        a, b = pos['BEGIN'], pos['END']
        records.append({'formula_id': source.formula_id, 'source_type': source.source_type,
            'source_part': source.source_part, 'source_locator': source.source_locator,
            'render_emission_ordinal': ordinal, 'display_mode': source.display_mode,
            'output_utf8_byte_range': [a,b], 'output_text_sha256': hashlib.sha256(data[a:b]).hexdigest(),
            'source_occurrence_uid': hashlib.sha256((source.source_part + ':' + source.source_locator + ':' + str(ordinal)).encode('utf-8')).hexdigest(),
            'actual_rendered_fragment': data[a:b].decode('utf-8')})
    return {'version': VERSION, 'status': 'SOURCE_FORMULA_EMISSION_RANGES_FROZEN',
        'byte_equivalent': True, 'markdown_sha256': hashlib.sha256(data).hexdigest(),
        'occurrences': records, 'word_pagination_certified': False,
        'no_formula_text_search_used': True, 'gold_read': False}
