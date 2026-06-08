"""
core -- text similarity & alignment toolkit

Sub-modules:
    sglsim   hybrid similarity (hybrid_similarity)
    sglalign sentence / character-level NW alignment
    sgldiff  character-level diff
    sgltext  sentence segmentation
    sglword  word-level alignment (jieba)
"""

from core._types import (
    SimilarityResult, AlignedPair, AlignmentResult,
    ShuffleGroup, CharDiff, DiffBlock, RichDiffResult,
    WordDiffBlock, WordDiffResult
)

from core import sglsim, sglalign, sgldiff, sgltext, sglword
from core.sglsim import hybrid_similarity, hybriddetail
from core.sglalign import Aligner
from core.sgltext import splitsents

__all__ = [
    "sglsim", "sglalign", "sgldiff", "sgltext", "sglword",
    "SimilarityResult", "AlignedPair", "AlignmentResult",
    "ShuffleGroup", "CharDiff", "DiffBlock", "RichDiffResult",
    "WordDiffBlock", "WordDiffResult",
    "hybrid_similarity", "hybriddetail", "Aligner", "splitsents",
]
