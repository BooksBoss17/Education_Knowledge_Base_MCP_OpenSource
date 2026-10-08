"""One bounded protocol repair for invalid/unproved records, retaining every completed valid response."""

from probe_zero99_toc_semantic import PROMPT

PROMPT_V2=PROMPT.replace('D对应DECORATIVE_SCENE/INDEPENDENT_FOOTER_MARK/MEANINGLESS_FRAGMENT',
    'D对应DECORATIVE_SCENE/DECORATIVE_BACKGROUND/INDEPENDENT_FOOTER_MARK/MEANINGLESS_FRAGMENT')+'''
契约补充：DECORATIVE_BACKGROUND表示无独立内容的渐变、纹理、边框等背景。INDEPENDENT_FOOTER_MARK只用于图片主体为单个页脚页序；大块背景或场景中附带一个页码不等于整张图就是页脚标记。多行页码数列、目录条目文字及其点状引线都是目录内容，不能称为独立页脚。
visible_text只需最短的有代表性片段，不必完整转录长文字；不得从MD补写图里没有的字。D必须提供存在的原文行号，且来源确实是目录；否则应选择U。'''
