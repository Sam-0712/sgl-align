"""
sgl.sglword -- word-level alignment (jieba segmentation + NW).

    >>> from sgl.sglword import word_align
    >>> r = word_align("夏日的余温尚未散尽", "夏天的余热尚未散尽")
    >>> for b in r.blocks:
    ...     print(b.block_type, b.source, "->", b.target)
    modify 夏日 -> 夏天
    equal 的 -> 的
    modify 余温 -> 余热
    equal 尚未 散尽 -> 尚未 散尽
"""

import re
from typing import Callable, Optional

from sgl.sglsim import hybrid_similarity
from sgl.sglalign import Aligner
from sgl.sgldiff import CharLevelAligner
from sgl._types import AlignedPair, CharDiff, WordDiffBlock, WordDiffResult


# segmenter type: str -> list[str]
Segmenter = Callable[[str], list[str]]

# Lazy-load jieba so it is optional.
_jieba = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        try:
            import jieba as _j
        except ImportError:
            raise ImportError(
                "jieba is required for word alignment.  "
                "Install with: pip install jieba")
        _j.setLogLevel(20)  # suppress jieba debug output
        _jieba = _j
    return _jieba


# ========== Built-in segmenter ==========

# Characters that should always be standalone tokens.
_SEP_PUNCT_RE = re.compile(
    r'[\u3000-\u303f'   # CJK punctuation (，。！？；：「」『』)
    r'\uff00-\uffef'    # full-width forms
    r'.,!?;:\"\'\(\)\[\]\{\}]+'
)


def _split_punct(words: list[str]) -> list[str]:
    """Split punctuation attached to word boundaries into separate tokens.

    e.g. ['执念，'] -> ['执念', '，'], ['「中'] -> ['「', '中']
    """
    result: list[str] = []
    for w in words:
        # Split at punctuation boundaries while keeping punctuation.
        parts = re.split(f'({_SEP_PUNCT_RE.pattern})', w)
        for p in parts:
            if p:
                result.append(p)
    return result


def _default_segment(text: str) -> list[str]:
    """Segment mixed Chinese/English text using jieba; English parts stay as whole words."""
    jieba = _get_jieba()
    words = jieba.lcut(text)
    result: list[str] = []
    buf = ""
    for w in words:
        if w.isspace():
            if buf:
                result.append(buf); buf = ""
            continue
        if re.fullmatch(r'[a-zA-Z0-9]+', w):
            buf += w
        else:
            if buf:
                result.append(buf); buf = ""
            result.append(w)
    if buf:
        result.append(buf)
    return _split_punct(result)


# ========== Word alignment entry ==========

def word_align(
    src: str,
    tgt: str,
    *,
    segmenter: Optional[Segmenter] = None,
    gap_open: float = -1.5,
    gap_extend: float = -0.2,
    mm_th: float = 0.5,
    detect_moves: bool = False,
) -> WordDiffResult:
    """Align two texts at the **word** level.

    Parameters
    ----------
    src / tgt: source and target strings.
    segmenter: optional custom segmenter ``str -> list[str]``.
    gap_open, gap_extend, mm_th: passed to the internal ``Aligner``.
    detect_moves: bool, default False
        If True, detect moved blocks (identical content at different positions).

    Returns
    -------
    WordDiffResult containing a list of ``WordDiffBlock`` objects.
    """
    seg = segmenter or _default_segment
    sw = seg(src)
    tw = seg(tgt)

    if not sw and not tw:
        return WordDiffResult(blocks=[])
    if not sw:
        return WordDiffResult(blocks=[
            WordDiffBlock("insert", None, tgt, [], tw, _char_diff(None, tgt))])
    if not tw:
        return WordDiffResult(blocks=[
            WordDiffBlock("delete", src, None, sw, [], _char_diff(src, None))])

    aligner = Aligner(gap_open=gap_open, gap_extend=gap_extend,
                      mm_th=mm_th, linear=False)
    result = aligner.align(sw, tw, hybrid_similarity)
    blocks = _merge_pairs(result.pairs)

    if detect_moves:
        blocks = _detect_word_moves(blocks)

    return WordDiffResult(blocks=blocks)


def _detect_word_moves(blocks: list[WordDiffBlock]) -> list[WordDiffBlock]:
    """Detect moved word blocks: matching delete + insert pairs -> move.

    Only works at word-level where jieba segmentation is used.
    Content must be **exactly equal** (==) to qualify as a move.
    Supports both delete->insert and insert->delete order.
    """
    result: list[WordDiffBlock] = []
    deletes: list[tuple[int, WordDiffBlock]] = []
    inserts: list[tuple[int, WordDiffBlock]] = []

    for block in blocks:
        if block.block_type == "delete":
            # Try to match this delete with a pending insert
            matched = False
            if inserts:
                for idx, ins_block in inserts:
                    if (ins_block.target is not None and
                        block.source is not None and
                        ins_block.target == block.source):
                        result[idx] = WordDiffBlock(
                            "move", block.source, ins_block.target,
                            block.source_words, ins_block.target_words,
                            ins_block.char_diffs,
                        )
                        inserts.remove((idx, ins_block))
                        matched = True
                        break
            if not matched:
                deletes.append((len(result), block))
                result.append(block)
        elif block.block_type == "insert":
            # Try to match this insert with a pending delete
            matched = False
            if deletes:
                for idx, del_block in deletes:
                    if (del_block.source is not None and
                        block.target is not None and
                        del_block.source == block.target):
                        result[idx] = WordDiffBlock(
                            "move", del_block.source, block.target,
                            del_block.source_words, block.target_words,
                            block.char_diffs,
                        )
                        deletes.remove((idx, del_block))
                        matched = True
                        break
            if not matched:
                inserts.append((len(result), block))
                result.append(block)
        else:
            result.append(block)

    return result


# ========== _merge_pairs refactored (T04) ==========


def _classify_pair_type(p: AlignedPair) -> str:
    """Classify a single AlignedPair into 'equal', 'modify', 'delete', 'insert', or 'gap'.

    Returns
    -------
    str describing the pair type.
    """
    if p.state == "match" and not p.is_gap:
        if p.source == p.target:
            return "equal"
        return "modify"
    if p.is_gap:
        if p.source and not p.target:
            return "delete"
        if p.target and not p.source:
            return "insert"
        return "gap"
    return "gap"


def _build_modify_block(pairs: list[AlignedPair], start: int, end: int) -> WordDiffBlock:
    """Build a merged modify block from pairs[start:end].

    Collects all source and target tokens from a contiguous gap region
    that contains both deletes and inserts (alternating).
    """
    src_raw: list[str] = []
    tgt_raw: list[str] = []
    src_words: list[str] = []
    tgt_words: list[str] = []

    for k in range(start, end):
        cp = pairs[k]
        if cp.source:
            src_raw.append(cp.source)
            src_words.append(cp.source)
        if cp.target:
            tgt_raw.append(cp.target)
            tgt_words.append(cp.target)

    src_text = "".join(src_raw)
    tgt_text = "".join(tgt_raw)
    return WordDiffBlock(
        "modify", src_text, tgt_text,
        src_words, tgt_words,
        _char_diff(src_text, tgt_text),
    )


def _build_gap_block(p: AlignedPair) -> WordDiffBlock:
    """Build a single delete or insert block from a gap pair.

    Returns
    -------
    WordDiffBlock with block_type "delete" or "insert".
    """
    if p.source and not p.target:
        sw = [p.source]
        return WordDiffBlock(
            "delete", p.source, None, sw, [],
            _char_diff(p.source, None),
        )
    # insert
    tw = [p.target]
    return WordDiffBlock(
        "insert", None, p.target, [], tw,
        _char_diff(None, p.target),
    )


def _merge_pairs(pairs: list[AlignedPair]) -> list[WordDiffBlock]:
    """Group aligned pairs into word-diff blocks — one block per token.

    - Identical words  → individual 'equal' blocks
    - Matched but different words → individual 'modify' blocks
    - Pure consecutive deletes  → individual 'delete' blocks
    - Pure consecutive inserts  → individual 'insert' blocks
    - Alternating deletes + inserts → one 'modify' block (merged)
    """
    blocks: list[WordDiffBlock] = []
    i, n = 0, len(pairs)

    while i < n:
        p = pairs[i]

        if _classify_pair_type(p) in ("equal", "modify"):
            # Direct single-pair equal or modify block
            if p.source == p.target:
                blocks.append(WordDiffBlock(
                    "equal", p.source, p.target,
                    [p.source], [p.target], None))
            elif p.source and p.target:
                blocks.append(WordDiffBlock(
                    "modify", p.source, p.target,
                    [p.source], [p.target],
                    _char_diff(p.source, p.target)))
            i += 1
            continue

        if p.is_gap:
            # Gap pair: peek ahead to determine if alternating or pure
            has_del = bool(p.source)
            has_ins = bool(p.target)
            j = i + 1

            while j < n:
                cp = pairs[j]
                cp_type = _classify_pair_type(cp)
                if cp_type in ("equal", "modify"):
                    break
                if cp.is_gap:
                    if cp.source:
                        has_del = True
                    if cp.target:
                        has_ins = True
                j += 1

            if has_del and has_ins:
                # Alternating deletes + inserts -> merged modify block
                blocks.append(_build_modify_block(pairs, i, j))
                i = j
            else:
                # Pure type: individual blocks per pair
                while i < j:
                    cp = pairs[i]
                    cp_type = _classify_pair_type(cp)
                    if cp_type in ("equal", "modify"):
                        break
                    if cp.is_gap:
                        blocks.append(_build_gap_block(cp))
                    i += 1
        else:
            # Mismatch pair: convert to individual delete + insert blocks
            if p.source:
                blocks.append(WordDiffBlock(
                    "delete", p.source, None, [p.source], [],
                    _char_diff(p.source, None)))
            if p.target:
                blocks.append(WordDiffBlock(
                    "insert", None, p.target, [], [p.target],
                    _char_diff(None, p.target)))
            i += 1

    return blocks


def _char_diff(src: Optional[str], tgt: Optional[str]) -> list[CharDiff]:
    """Build character-level diff for a modify/delete/insert block."""
    if src is None:
        return [CharDiff(None, c, "insert") for c in (tgt or "")]
    if tgt is None:
        return [CharDiff(c, None, "delete") for c in src]
    return CharLevelAligner().align_chars(src, tgt)
