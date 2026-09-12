"""Optional source-risk screen for extra formatted recognition, never for text OCR.

This screen only selects auxiliary work. Its coverage must be measured against
complete documents and independent references before enabling it by default.
"""
import math
import re
import unicodedata


def formatted_context_risk(layout, image):
    def verdict(required, reason):
        return {'required': required, 'reason': reason}

    text = layout.text
    if re.search(r'[\u0370-\u03ff]', text):
        return verdict(True, 'GREEK_TEXT')
    if re.search(r'[\u00b2\u00b3\u00b9\u2070-\u209f_^]', text):
        return verdict(True, 'SCRIPT_TEXT')
    # An answer-option prefix is not a mathematical superscript signal.
    body = re.sub(r'^\s*[A-D][.、)]\s*', '', text)
    if re.search(r'\d\s*[A-Za-z]|[A-Za-z]\s*[\d。.]', body):
        return verdict(True, 'LETTER_DIGIT_OR_SCRIPT_PUNCTUATION')
    if (image.size != (layout.width, layout.height) or len(layout.centers) != len(text)
            or any(not math.isfinite(x) for x in layout.centers)
            or any(a >= b for a, b in zip(layout.centers, layout.centers[1:]))):
        return verdict(True, 'SOURCE_GEOMETRY_UNCERTAIN')
    import numpy as np
    import cv2

    ink = np.asarray(image.convert('L')) < 180
    height = layout.height
    # Use complete 2-D ink components. Stacked nucleus scripts share columns
    # but are separate glyphs; column projection would hide both as tall ink.
    # Complete components also avoid cutting a neighboring Chinese stroke.
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    bands = []
    for x, y, width, component_height, area in stats[1:]:
        if area >= 3:
            bands.append((int(x), int(x+width), int(y), int(y+component_height)))
    for index, char in enumerate(text):
        if not re.fullmatch('[A-Za-z]', char):
            continue
        center = layout.centers[index]
        left = (layout.centers[index-1] + center)/2 if index else 0
        right = (layout.centers[index+1] + center)/2 if index+1 < len(text) else layout.width
        low = max(0, int(left-height*0.25))
        high = min(layout.width, int(math.ceil(right+height*0.25)))
        for start, end, top, bottom in bands:
            if end <= low or start >= high:
                continue
            midpoint = (start+end)/2
            nearest = min(range(len(text)), key=lambda i: abs(layout.centers[i]-midpoint))
            if (top >= height*0.5 and unicodedata.category(text[nearest]).startswith('P')
                    and abs(layout.centers[nearest]-midpoint) <= height*0.4):
                continue
            if (abs(midpoint-center) > height*0.2 and end-start <= height*0.75
                    and 2 <= bottom-top <= height*0.7
                    and (bottom <= height*0.72 or top >= height*0.35)):
                return {**verdict(True, 'POSSIBLE_SCRIPT_PIXELS'),
                        'signal': {'character_index': index, 'bbox': [start, top, end, bottom],
                                   'nearest_character_index': nearest}}
    return verdict(False, 'NO_STRUCTURED_MATH_SIGNAL')
