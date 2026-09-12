"""Validated CTC geometry for source-bound local mathematics recognition.

PaddleX's text predictor sorts width ratios for word positions while retaining
image order. Sorting each submitted batch avoids that mismatch; outputs are
then restored to caller order. No image resizing or text rewriting happens here.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class CharacterLayout:
    text: str
    width: int
    height: int
    centers: tuple[float, ...]

    @classmethod
    def from_recognition(cls, value, source_size):
        try:
            text, info = value
            feature_length, groups, columns, states = info
            width, height = source_size
            characters = [char for group in groups for char in group]
            positions = [float(position) for group in columns for position in group]
        except (TypeError, ValueError) as exc:
            raise ValueError("CHARACTER_LAYOUT_INVALID") from exc
        if not isinstance(text, str) or any(not isinstance(c, str) or len(c) != 1 for c in characters):
            raise ValueError("CHARACTER_LAYOUT_INVALID")
        if "".join(characters) != text:
            raise ValueError("CHARACTER_TEXT_MISMATCH")
        if len(groups) != len(columns) or len(groups) != len(states) or len(positions) != len(text):
            raise ValueError("CHARACTER_POSITION_CARDINALITY_MISMATCH")
        if any(len(g) != len(c) for g, c in zip(groups, columns, strict=True)):
            raise ValueError("CHARACTER_POSITION_CARDINALITY_MISMATCH")
        if not isinstance(feature_length, (int, float)) or not math.isfinite(feature_length) or feature_length <= 0:
            raise ValueError("FEATURE_MAP_INVALID")
        if width <= 0 or height <= 0:
            raise ValueError("SOURCE_SIZE_INVALID")
        if any(not math.isfinite(p) or p < 0 or p >= feature_length for p in positions):
            raise ValueError("CHARACTER_OUTSIDE_FEATURE_MAP")
        if any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError("CHARACTER_POSITION_ORDER_INVALID")
        return cls(text, int(width), int(height), tuple(p * width / feature_length for p in positions))

    def crop_box(self, image, start: int, end: int, *, prefix_sup=False, prefix_sub=False):
        """Snap character boundaries to source whitespace, or leave unresolved.

        A source padding margin can recruit the previous Chinese character's
        strokes and change an isotope into a negative number. Caller may add
        white canvas after extraction; this function never recruits neighbor ink.
        """
        import numpy as np

        if image.size != (self.width, self.height):
            raise ValueError("CHARACTER_LAYOUT_IMAGE_SIZE_MISMATCH")
        if not 0 <= start < end <= len(self.text):
            raise ValueError("CHARACTER_SPAN_INVALID")
        gray = np.asarray(image.convert("L"))
        empty_columns = np.flatnonzero((gray < 180).sum(axis=0) == 0)

        def gap(left_center, right_center):
            allowed = empty_columns[(empty_columns > left_center) & (empty_columns < right_center)]
            if not len(allowed):
                return None
            midpoint = (left_center + right_center) / 2
            return int(min(allowed, key=lambda x: (abs(float(x)-midpoint), int(x))))

        left = 0 if start == 0 else gap(self.centers[start-1], self.centers[start])
        right = self.width if end == len(self.text) else gap(self.centers[end-1], self.centers[end])
        if left is None or right is None or left >= right:
            return None
        if start and (prefix_sup or prefix_sub):
            # CTC centers are quantized by batch padding. A thin, unrecognized
            # raised digit can sit immediately before the chosen whitespace.
            # Recover only a compact, vertically displaced source component;
            # never cross the preceding character center or recruit tall ink.
            ink = gray < 180
            occupied = ink.any(axis=0)
            floor = max(0, int(math.floor(self.centers[start-1])) + 1)
            while left > floor:
                columns = np.flatnonzero(occupied[floor:left]) + floor
                if not len(columns):
                    break
                last = int(columns[-1])
                if left - last - 1 > self.height * 0.25:
                    break
                first = last
                while first > 0 and occupied[first-1]:
                    first -= 1
                ys = np.flatnonzero(ink[:, first:last+1].any(axis=1))
                top, bottom = int(ys[0]), int(ys[-1]) + 1
                is_script = ((prefix_sup and top <= self.height * 0.35 and bottom <= self.height * 0.72)
                             or (prefix_sub and top >= self.height * 0.35 and bottom >= self.height * 0.65))
                if (first <= floor or bottom-top > self.height * 0.7
                        or last-first+1 > self.height * 0.75 or not is_script):
                    break
                left = first - 1
        return (left, 0, right, self.height)


def predict_positioned_batch(model: Any, paths: Sequence[Path], *, batch_size: int):
    """Ask the existing recognizer for positions without reordering its callers."""
    from PIL import Image

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("CHARACTER_POSITION_BATCH_SIZE_INVALID")
    paths = [Path(path) for path in paths]
    results = []
    for offset in range(0, len(paths), batch_size):
        group = paths[offset:offset+batch_size]
        ratios = []
        for index, path in enumerate(group):
            with Image.open(path) as image:
                ratios.append((image.width / image.height, index, path))
        ordered = sorted(ratios)
        predicted = list(model.predict([str(p) for _, _, p in ordered], batch_size=batch_size, return_word_box=True))
        if len(predicted) != len(ordered):
            raise ValueError("CHARACTER_PREDICTION_CARDINALITY_MISMATCH")
        restored = [None] * len(group)
        for (_, index, source), value in zip(ordered, predicted, strict=True):
            if Path(value["input_path"]).resolve() != source.resolve():
                raise ValueError("CHARACTER_PREDICTION_SOURCE_PATH_MISMATCH")
            restored[index] = value
        results.extend(restored)
    return results
