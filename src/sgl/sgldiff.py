"""sgl.sgldiff -- character-level diff alignment."""

from __future__ import annotations

from typing import Any, Callable, Optional

from sgl.sglalign import Aligner
from sgl._types import AlignedPair, CharDiff, DiffBlock, RichDiffResult

SimilarityFunc = Callable[[Any, Any], float]


def _char_sim(c1: str, c2: str) -> float:
    return 1.0 if c1 == c2 else 0.0


class CharLevelAligner:
    """Character-level diff alignment with stricter gap penalties."""

    def __init__(self):
        self.aligner = Aligner(gap_open=-0.8, gap_extend=-0.8, mm_th=0.1, linear=False)

    def align_chars(self, text1: str, text2: str) -> list[CharDiff]:
        if not text1:
            return [CharDiff(None, c, "insert") for c in text2]
        if not text2:
            return [CharDiff(c, None, "delete") for c in text1]

        raw = self.aligner.align(list(text1), list(text2), _char_sim).pairs
        refined: list[CharDiff] = []
        i, n = 0, len(raw)

        while i < n:
            p = raw[i]
            # 相等字符直接通过
            if p.source and p.target and p.source == p.target:
                refined.append(CharDiff(p.source, p.target, "equal"))
                i += 1
                continue

            # 差异块：连续不等字符先删后增
            orig, rew = [], []
            while i < n:
                cp = raw[i]
                if cp.source and cp.target and cp.source == cp.target:
                    break
                if cp.source: orig.append(cp.source)
                if cp.target: rew.append(cp.target)
                i += 1
            for c in orig:
                refined.append(CharDiff(c, None, "delete"))
            for c in rew:
                refined.append(CharDiff(None, c, "insert"))

        return refined


def richdiff(
    alignment_result: list[AlignedPair],
    similarity_func: Optional[SimilarityFunc] = None,
    anchor_threshold: float = 0.80,
    detect_moves: bool = False,
) -> RichDiffResult:
    """Generate rich diff with optional move detection.

    Parameters
    ----------
    alignment_result : list[AlignedPair]
    similarity_func : SimilarityFunc, optional
    anchor_threshold : float, default 0.80
    detect_moves : bool, default False
        If True, perform cross-block move detection on delete+insert pairs.
        Blocks with identical content at different positions are marked as
        "move" instead of "delete" + "insert".
        Only meaningful when used with word-level alignment results.
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
        if anchor(p.source, p.target):
            if p.source == p.target:
                blocks.append(DiffBlock("equal", p.source, p.target, [CharDiff(c, c, "equal") for c in p.source]))
            else:
                blocks.append(DiffBlock("modify", p.source, p.target, ca.align_chars(p.source or "", p.target or "")))
            i += 1
            continue

        orig, rew = [], []
        while i < n:
            cp = alignment_result[i]
            if anchor(cp.source, cp.target):
                break
            if cp.source: orig.append(cp.source)
            if cp.target: rew.append(cp.target)
            i += 1

        fo, fr = "".join(orig), "".join(rew)
        if fo and fr:
            blocks.append(DiffBlock("modify", fo, fr, ca.align_chars(fo, fr)))
        elif fo:
            blocks.append(DiffBlock("delete", fo, None, [CharDiff(c, None, "delete") for c in fo]))
        elif fr:
            blocks.append(DiffBlock("insert", None, fr, [CharDiff(None, c, "insert") for c in fr]))

    if detect_moves:
        blocks = _detect_moves(blocks)

    return RichDiffResult(blocks=blocks)


def _detect_moves(blocks: list[DiffBlock]) -> list[DiffBlock]:
    """Detect cross-block movement: matching delete + insert pairs -> move.

    策略：
    1. 遍历 blocks，找出 delete 块和 insert 块
    2. 对 delete 块的内容与 insert 块的内容做**完全匹配**（==）
    3. 如果内容完全相同的 delete/insert 对存在，标记为 move
    4. move 块保留 source_text 和 target_text 不变，
       但 block_type = "move"，chars 中的每个 CharDiff 也标记为 "move"
    5. 匹配过的块不再参与后续匹配
    6. 支持 delete->insert 和 insert->delete 两种顺序

    Returns
    -------
    list[DiffBlock] with delete/insert pairs replaced by move blocks.
    """
    result: list[DiffBlock] = []
    deletes: list[tuple[int, DiffBlock]] = []  # (index_in_result, block)
    inserts: list[tuple[int, DiffBlock]] = []  # (index_in_result, block)

    for block in blocks:
        if block.block_type == "delete":
            # Try to match this delete with a pending insert
            matched = False
            if inserts:
                for idx, ins_block in inserts:
                    if (ins_block.target_text is not None and
                        block.source_text is not None and
                        ins_block.target_text == block.source_text):
                        # Match found -> replace both with a move block
                        result[idx] = DiffBlock(
                            "move", block.source_text, ins_block.target_text,
                            [CharDiff(c, c, "move") for c in (block.source_text or "")],
                            source_idx=block.source_idx,
                            target_idx=ins_block.target_idx,
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
                    if (del_block.source_text is not None and
                        block.target_text is not None and
                        del_block.source_text == block.target_text):
                        # Match found -> replace both with a move block
                        result[idx] = DiffBlock(
                            "move", del_block.source_text, block.target_text,
                            [CharDiff(c, c, "move") for c in (del_block.source_text or "")],
                            source_idx=del_block.source_idx,
                            target_idx=block.target_idx,
                        )
                        deletes.remove((idx, del_block))
                        matched = True
                        break
            if not matched:
                inserts.append((len(result), block))
                result.append(block)
        elif block.block_type == "modify" and block.source_text == block.target_text:
            # Merged gap pairs with identical content: split for move detection
            content = block.source_text
            del_b = DiffBlock("delete", content, None,
                [CharDiff(c, None, "delete") for c in content],
                source_idx=block.source_idx)
            ins_b = DiffBlock("insert", None, content,
                [CharDiff(None, c, "insert") for c in content],
                target_idx=block.target_idx)

            # Process delete
            matched_del = False
            if inserts:
                for idx, ins_block in inserts:
                    if ins_block.target_text == content:
                        result[idx] = DiffBlock(
                            "move", content, ins_block.target_text,
                            [CharDiff(c, c, "move") for c in content])
                        inserts.remove((idx, ins_block))
                        matched_del = True
                        break
            if not matched_del:
                deletes.append((len(result), del_b))
                result.append(del_b)

            # Process insert
            matched_ins = False
            if deletes:
                for idx, del_block in deletes:
                    if del_block.source_text == content:
                        result[idx] = DiffBlock(
                            "move", del_block.source_text, content,
                            [CharDiff(c, c, "move") for c in content])
                        deletes.remove((idx, del_block))
                        matched_ins = True
                        break
            if not matched_ins:
                inserts.append((len(result), ins_b))
                result.append(ins_b)
        else:
            result.append(block)

    return result


def chardiff(text1: str, text2: str) -> list[CharDiff]:
    return CharLevelAligner().align_chars(text1, text2)
