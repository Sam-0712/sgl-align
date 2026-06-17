"""core.sgldiff -- character-level diff alignment."""

from __future__ import annotations

from typing import Any, Callable, Optional

from core.sglalign import Aligner
from core._types import AlignedPair, CharDiff, DiffBlock, RichDiffResult

SimilarityFunc = Callable[[Any, Any], float]


def _char_sim(c1: str, c2: str) -> float:
    return 1.0 if c1 == c2 else 0.0


class CharLevelAligner:
    """Character-level diff alignment with stricter gap penalties."""

    def __init__(self):
        self.aligner = Aligner(gap_open=-0.8, gap_extend=-0.8, mm_th=0.1, linear=False)

    def align_chars(self, text1: str, text2: str) -> list[CharDiff]:
        """Align two strings at character level with block-merging logic.

        Consecutive non-equal character pairs are collected together, then
        emitted as all deletes followed by all inserts (matching the reference
        char_aligner.py behaviour).  This prevents fragmented ``[del][ins][del][ins]``
        and produces clean ``[del...del][ins...ins]`` blocks.
        """
        if not text1:
            return [CharDiff(None, c, "insert") for c in text2]
        if not text2:
            return [CharDiff(c, None, "delete") for c in text1]

        raw = self.aligner.align(list(text1), list(text2), _char_sim).pairs
        refined: list[CharDiff] = []
        i, n = 0, len(raw)

        while i < n:
            p = raw[i]
            # equal character – pass through directly
            if p.source and p.target and p.source == p.target:
                refined.append(CharDiff(p.source, p.target, "equal"))
                i += 1
                continue

            # difference cluster: collect consecutive non-equal pairs
            orig, rew = [], []
            while i < n:
                cp = raw[i]
                if cp.source and cp.target and cp.source == cp.target:
                    break
                if cp.source:
                    orig.append(cp.source)
                if cp.target:
                    rew.append(cp.target)
                i += 1
            # emit all deletes first, then all inserts
            for c in orig:
                refined.append(CharDiff(c, None, "delete"))
            for c in rew:
                refined.append(CharDiff(None, c, "insert"))

        return refined


def richdiff(
    alignment_result: list[AlignedPair],
    similarity_func: Optional[SimilarityFunc] = None,
    anchor_threshold: float = 0.80,
    **kwargs: Any,
) -> RichDiffResult:
    """Generate rich diff with block-merging logic (matching reference char_aligner.py).

    Consecutive non-anchor pairs are collected and merged into a single
    ``modify`` block, followed by a character-level diff of the joined text.
    This prevents fragmented delete/insert blocks and produces cleaner output.

    Parameters
    ----------
    alignment_result : list[AlignedPair]
    similarity_func : SimilarityFunc, optional
    anchor_threshold : float, default 0.80
    **kwargs
        Absorb legacy parameters (e.g. ``detect_moves``) for backward compatibility.
    """
    ca = CharLevelAligner()

    def anchor(s1: Optional[str], s2: Optional[str]) -> bool:
        if not s1 or not s2:
            return False
        return s1 == s2 or (similarity_func is not None and similarity_func(s1, s2) >= anchor_threshold)

    blocks: list[DiffBlock] = []
    i, n = 0, len(alignment_result)

    while i < n:
        p = alignment_result[i]

        # 1. Anchor matched pair
        if anchor(p.source, p.target):
            if p.source is not None and p.target is not None:
                if p.source == p.target:
                    blocks.append(DiffBlock(
                        "equal", p.source, p.target,
                        [CharDiff(c, c, "equal") for c in p.source],
                    ))
                else:
                    blocks.append(DiffBlock(
                        "modify", p.source, p.target,
                        ca.align_chars(p.source, p.target),
                    ))
            i += 1
            continue

        # 2. Non-anchor cluster: collect consecutive non-anchor pairs and merge
        cluster_orig: list[str] = []
        cluster_rew: list[str] = []

        while i < n:
            cp = alignment_result[i]
            if anchor(cp.source, cp.target):
                break
            if cp.source:
                cluster_orig.append(cp.source)
            if cp.target:
                cluster_rew.append(cp.target)
            i += 1

        # 3. Emit merged block
        full_orig = "".join(cluster_orig)
        full_rew = "".join(cluster_rew)

        if full_orig and full_rew:
            # Complex modification block – char-level diff on the combined text
            blocks.append(DiffBlock(
                "modify", full_orig, full_rew,
                ca.align_chars(full_orig, full_rew),
            ))
        elif full_orig:
            blocks.append(DiffBlock(
                "delete", full_orig, None,
                [CharDiff(c, None, "delete") for c in full_orig],
            ))
        elif full_rew:
            blocks.append(DiffBlock(
                "insert", None, full_rew,
                [CharDiff(None, c, "insert") for c in full_rew],
            ))

    return RichDiffResult(blocks=blocks)


def chardiff(text1: str, text2: str) -> list[CharDiff]:
    return CharLevelAligner().align_chars(text1, text2)
