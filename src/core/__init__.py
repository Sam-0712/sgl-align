"""
core -- text similarity & alignment toolkit

Sub-modules:
    sglsim   hybrid similarity (hybrid_similarity)
    sglalign sentence / character-level NW alignment
    sgldiff  character-level diff
    sgltext  paragraph + sentence segmentation
    sglword  word-level alignment (jieba)
    sglcmi   Change Magnitude Index computation
"""

from core._types import (
    SimilarityResult, AlignedPair, AlignmentResult,
    ShuffleGroup, CharDiff, DiffBlock, RichDiffResult,
    WordDiffBlock, WordDiffResult, CMIResult,
)

from core import sglsim, sglalign, sgldiff, sgltext, sglword, sglcmi
from core.sglsim import hybrid_similarity, hybriddetail
from core.sglalign import Aligner
from core.sgltext import splitsents, splitparas, countparas
from core.sglcmi import compute_cmi, merged_cmi, build_para_sent_map

__all__ = [
    "sglsim", "sglalign", "sgldiff", "sgltext", "sglword", "sglcmi",
    "SimilarityResult", "AlignedPair", "AlignmentResult",
    "ShuffleGroup", "CharDiff", "DiffBlock", "RichDiffResult",
    "WordDiffBlock", "WordDiffResult", "CMIResult",
    "hybrid_similarity", "hybriddetail", "Aligner",
    "splitsents", "splitparas", "countparas",
    "compute_cmi", "merged_cmi", "build_para_sent_map",
]
