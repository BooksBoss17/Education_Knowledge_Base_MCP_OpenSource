"""Source ranges for actual Markdown images, checked against CommonMark parsing."""
from collections import Counter
from dataclasses import dataclass
import re
from markdown_it import MarkdownIt

@dataclass(frozen=True)
class ImageReference:
    source: str
    begin: int
    finish: int
    destination: str
    title: str | None
    destination_begin: int
    destination_finish: int
    def start(self): return self.begin
    def end(self): return self.finish
    def group(self, index=0):
        if index != 0: raise IndexError(index)
        return self.source[self.begin:self.finish]
    def __getitem__(self, index): return self.group(index)

def image_references(source):
    parser = MarkdownIt('commonmark').enable('table')
    tokens = parser.parse(source); lines = [0]
    for match in re.finditer('\n', source): lines.append(match.end())
    lines.append(len(source))
    blocked = []
    expected = Counter()
    def visit(token):
        if token.type == 'image': expected[(token.attrGet('src'), token.attrGet('title'))] += 1
        for child in token.children or []: visit(child)
    for token in tokens:
        if token.type in ('fence', 'code_block') and token.map:
            blocked.append((lines[token.map[0]], lines[min(token.map[1], len(lines)-1)]))
        visit(token)
    found = []; i = 0
    while i < len(source):
        bound = next((b for a,b in blocked if a <= i < b), None)
        if bound is not None: i = bound; continue
        if source[i] == '\\': i += 2; continue
        if source[i] == '`':
            n = len(source[i:]) - len(source[i:].lstrip('`')); closing = re.search(r'(?<!`)'+re.escape('`'*n)+r'(?!`)', source[i+n:])
            i = i+n+closing.end() if closing else i+n
            continue
        if source[i:i+2] != '![': i += 1; continue
        start = i; j = i+2; depth = 1
        while j < len(source) and depth:
            if source[j] == '\\': j += 2; continue
            if source[j] == '[': depth += 1
            if source[j] == ']': depth -= 1
            j += 1
        if depth or j >= len(source) or source[j] != '(': i += 2; continue
        k = j+1; depth = 1; quote = None; angle = False
        while k < len(source) and depth:
            c = source[k]
            if c == '\\': k += 2; continue
            if quote:
                if c == quote: quote = None
            elif angle:
                if c == '>': angle = False
            elif c == '<': angle = True
            elif c in ('"', "'") and k > j+1 and source[k-1].isspace(): quote = c
            elif c == '(': depth += 1
            elif c == ')': depth -= 1
            k += 1
        if depth: i += 2; continue
        parsed = parser.parseInline(source[start:k])
        children = [t for t in parsed[0].children or [] if t.type == 'image'] if parsed else []
        if len(children) == 1:
            token = children[0]
            dest_begin = j + 1
            while dest_begin < k and source[dest_begin].isspace():
                dest_begin += 1
            dest_finish = dest_begin
            if source[dest_begin] == '<':
                dest_begin += 1
                dest_finish = dest_begin
                while dest_finish < k and source[dest_finish] != '>':
                    dest_finish += 2 if source[dest_finish] == '\\' else 1
            else:
                nested = 0
                while dest_finish < k:
                    char = source[dest_finish]
                    if char == '\\':
                        dest_finish += 2
                        continue
                    if char == '(':
                        nested += 1
                    elif char == ')':
                        if nested == 0:
                            break
                        nested -= 1
                    elif char.isspace() and nested == 0:
                        break
                    dest_finish += 1
            found.append(ImageReference(source,start,k,token.attrGet('src'),token.attrGet('title'),
                                        dest_begin,dest_finish))
            i = k
        else: i += 2
    actual = Counter((m.destination,m.title) for m in found)
    if actual != expected: raise ValueError('MARKDOWN_IMAGE_SOURCE_RANGES_NOT_FULLY_SUPPORTED')
    return found

def replace_image_references(source, destination, replacement):
    matches = [m for m in image_references(source) if m.destination == destination]
    for m in reversed(matches):
        value = replacement(m) if callable(replacement) else replacement
        source = source[:m.begin]+value+source[m.finish:]
    return source

def renamed_image_reference(match, destination):
    """Change the actual destination only; retain alt text and quoted title."""
    # Runtime asset names are relative portable paths. Reject arbitrary URI or
    # escaped user text here instead of synthesizing a new Markdown expression.
    import re
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', destination):
        raise ValueError('RENAMED_ASSET_PATH_NOT_PORTABLE')
    return (match.source[match.begin:match.destination_begin] + destination
            + match.source[match.destination_finish:match.finish])

def rename_image_destinations(source, destinations):
    """Rename in one pass so a new filename cannot become a later old target."""
    for match in reversed(image_references(source)):
        if match.destination in destinations:
            value = renamed_image_reference(match, destinations[match.destination])
            source = source[:match.begin] + value + source[match.finish:]
    return source
