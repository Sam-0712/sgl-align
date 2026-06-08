"""sgl.sgltext -- sentence segmentation."""

import re

_PAT = re.compile(
    r'([\u3002\uff01\uff1f.!?\u2026]+'
    r'[\u201c\u201d\u2018\u2019\"\'\u300a\u300b\u3008\u3009\u300c\u300d]*|\n+)'
)


def splitsents(text: str, minlen: int = 1,
               filterpunct: bool = True) -> list[str]:
    """Split text into sentences (Chinese + English punctuation)."""
    if not text or not text.strip():
        return []
    text = text.strip()
    parts = _PAT.split(text)
    sents = []
    i = 0
    while i < len(parts) - 1:
        c = (parts[i] + (parts[i + 1] or '')).strip()
        if c: sents.append(c)
        i += 2
    if len(parts) % 2 == 0 and parts[-1].strip():
        sents.append(parts[-1].strip())
    result = []
    for s in sents:
        s = re.sub(r'\s+', ' ', s.strip())
        if len(s) <= minlen: continue
        if filterpunct and not re.search(r'[\w\u4e00-\u9fff]', s): continue
        result.append(s)
    return result if result else [text]


def countsents(text: str) -> int:
    return len(splitsents(text))
