"""core._types -- structured output types."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Optional


def _to_native(val: object) -> int | float | list:
    """Convert numpy types to native Python types."""
    import numpy as np
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (int, float, list)):
        return val
    if isinstance(val, str):
        return val
    return val


def _deep_native(obj: object, round4: bool = True) -> object:
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _deep_native(v, round4) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_native(v, round4) for v in obj]
    val = _to_native(obj)
    if round4 and isinstance(val, float) and math.isfinite(val):
        return round(val, 4)
    return val


@dataclass
class SimilarityResult:
    score: float
    dice_score: float
    lcs_score: float
    roc_score: float
    complexity_a: float
    complexity_b: float
    n_gram_order: int
    n_gram_weights: dict[int, float]
    logic_reward: float
    len_penalty: float
    dispersion_consistency: float
    struct_ratio: float
    confidence: float

    def to_dict(self) -> dict[str, object]:
        d = _deep_native(asdict(self))
        assert isinstance(d, dict)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> dict[str, object]:
        return {
            "score": round(self.score, 4),
            "dice": round(self.dice_score, 4),
            "lcs": round(self.lcs_score, 4),
            "roc": round(self.roc_score, 4),
            "reward": round(self.logic_reward, 4),
        }


@dataclass
class ShuffleGroup:
    """A group of reordered sentence pairs."""
    source_indices: list[int]
    target_indices: list[int]


@dataclass
class AlignedPair:
    source: Optional[str]
    target: Optional[str]
    similarity: float = 0.0
    is_gap: bool = False
    state: str = "match"
    is_shuffled: bool = False


@dataclass
class AlignmentResult:
    pairs: list[AlignedPair]
    score: float
    n_source: int
    n_target: int
    source_seqs: list[str]
    target_seqs: list[str]
    dp_matrix: Optional[list[list[float]]] = None
    matrix_m: Optional[list[list[float]]] = None
    matrix_x: Optional[list[list[float]]] = None
    matrix_y: Optional[list[list[float]]] = None
    backtrace_path: Optional[list[tuple[int, int, str]]] = None
    fb_matrix: Optional[list[list[float]]] = None
    shuffle_groups: Optional[list[ShuffleGroup]] = None
    gap_open: float = -1.5
    gap_extend: float = -0.2
    mismatch_threshold: float = 0.2

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "n_source": self.n_source,
            "n_target": self.n_target,
            "score": round(self.score, 4),
            "pairs": [
                {"source": p.source, "target": p.target,
                 "similarity": round(p.similarity, 4), "is_gap": p.is_gap,
                 "state": p.state, "is_shuffled": p.is_shuffled}
                for p in self.pairs
            ],
            "backtrace_path": self.backtrace_path,
            "gap_open": round(self.gap_open, 4),
            "gap_extend": round(self.gap_extend, 4),
            "mismatch_threshold": round(self.mismatch_threshold, 4),
        }
        for k in ("dp_matrix", "matrix_m", "matrix_x", "matrix_y", "fb_matrix"):
            v = getattr(self, k)
            if v is not None:
                d[k] = _deep_native(v)
        if self.shuffle_groups:
            d["shuffle_groups"] = [
                {"source_indices": g.source_indices, "target_indices": g.target_indices}
                for g in self.shuffle_groups
            ]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> dict[str, object]:
        counts: dict[str, int] = {"match": 0, "mismatch": 0, "delete": 0, "insert": 0}
        shuffled = 0
        for p in self.pairs:
            counts[p.state] += 1
            if p.is_shuffled:
                shuffled += 1
        r: dict[str, object] = {"score": round(self.score, 4),
             f"{self.n_source}x{self.n_target}": "align", **counts}
        if shuffled:
            r["shuffled"] = shuffled
        return r


@dataclass
class CharDiff:
    """Character-level diff entry.

    - ``diff_type`` : ``"equal"`` | ``"delete"`` | ``"insert"``.
    """
    char_src: Optional[str]
    char_tgt: Optional[str]
    diff_type: str


@dataclass
class DiffBlock:
    """A contiguous block of character-level diff.

    - ``block_type``: 'equal', 'modify', 'delete', or 'insert'.
    - ``source_idx`` / ``target_idx``: optional sentence indices for move detection.
    """
    block_type: str
    source_text: Optional[str]
    target_text: Optional[str]
    chars: list[CharDiff]
    source_idx: Optional[int] = None
    target_idx: Optional[int] = None


@dataclass
class RichDiffResult:
    blocks: list[DiffBlock]
    has_char_matrix: bool = False

    def to_dict(self) -> dict[str, object]:
        d = _deep_native(asdict(self))
        assert isinstance(d, dict)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {"equal": 0, "modify": 0, "delete": 0, "insert": 0}
        for b in self.blocks:
            counts[b.block_type] += 1
        return counts

    def to_diff_text(self) -> str:
        lines: list[str] = []
        for b in self.blocks:
            if b.block_type == "equal":
                lines.append(f"  {b.source_text}")
            elif b.block_type == "modify":
                lines.append(f"- {b.source_text}")
                lines.append(f"+ {b.target_text}")
            elif b.block_type == "delete":
                lines.append(f"- {b.source_text}")
            elif b.block_type == "insert":
                lines.append(f"+ {b.target_text}")
        return "\n".join(lines)


# ========== Word-level alignment types ==========

@dataclass
class WordDiffBlock:
    """A contiguous block of word-level diff.

    - ``block_type``: 'equal', 'modify', 'delete', 'insert', or 'move'.
    - ``source`` / ``target``: joined source / target strings (None for pure insert / delete).
    - ``source_words`` / ``target_words``: individual word lists.
    - ``char_diffs``: character-level diffs for modify blocks (list of CharDiff or None).
    """
    block_type: str
    source: Optional[str]
    target: Optional[str]
    source_words: list[str]
    target_words: list[str]
    char_diffs: Optional[list[CharDiff]] = None

    @property
    def is_gap(self) -> bool:
        return self.block_type in ("delete", "insert")


@dataclass
class WordDiffResult:
    blocks: list[WordDiffBlock]

    def to_dict(self) -> dict[str, object]:
        d = _deep_native(asdict(self))
        assert isinstance(d, dict)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {"equal": 0, "modify": 0, "delete": 0, "insert": 0, "move": 0}
        for b in self.blocks:
            counts[b.block_type] += 1
        return counts

    def to_diff_lines(self) -> list[str]:
        lines: list[str] = []
        for b in self.blocks:
            if b.block_type == "equal":
                lines.append(f"  {b.source}")
            elif b.block_type == "modify":
                lines.append(f"- {b.source}")
                lines.append(f"+ {b.target}")
                if b.char_diffs:
                    src_chars = "".join(c.char_src or "" for c in b.char_diffs)
                    tgt_chars = "".join(c.char_tgt or "" for c in b.char_diffs)
                    lines.append(f"  [{src_chars} → {tgt_chars}]")
            elif b.block_type == "delete":
                lines.append(f"- {b.source}")
            elif b.block_type == "insert":
                lines.append(f"+ {b.target}")
            elif b.block_type == "move":
                lines.append(f"~ {b.source}")
        return lines

    def to_diff_text(self) -> str:
        return "\n".join(self.to_diff_lines())


# ========== CMI (Change Magnitude Index) types ==========

@dataclass
class CMIResult:
    """Change Magnitude Index at multiple granularities.

    CMI ∈ [0, 1].  0 = identical texts, 1 = maximum possible edit distance.

    Three granularities:
    - ``cmi_document`` — single scalar for the whole alignment.
    - ``cmi_paragraphs`` — per-paragraph CMI (requires paragraph mapping).
    - ``cmi_sentences`` — per-aligned-pair CMI.

    ``direction`` is ``"a_to_b"`` for single-direction CMI or ``"merged"``
    for the arithmetic-mean of both directions.
    """
    # ── document ──
    cmi_document: float          # full-document CMI  [0, 1]
    # ── paragraph ──
    cmi_paragraphs: list[float]   # per-paragraph CMI (empty if no para map)
    n_paragraphs: int
    # ── sentence ──
    cmi_sentences: list[float]    # per-aligned-pair CMI
    n_pairs: int
    # ── metadata ──
    direction: str                # "a_to_b" | "merged"
    # ── cost breakdown ──
    total_edit_cost: float        # Σ actual edit costs
    total_max_cost: float         # Σ max possible costs
    match_cost_sum: float         # Σ costs from match/mismatch pairs
    gap_cost_sum: float           # Σ costs from gap pairs
    n_match_pairs: int
    n_gap_pairs: int
    # ── alignment parameters (for provenance) ──
    gap_open: float
    gap_extend: float

    def to_dict(self) -> dict[str, object]:
        d = _deep_native(asdict(self))
        assert isinstance(d, dict)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
