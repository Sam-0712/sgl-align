"""
sgl -- text similarity & alignment toolkit

Sub-modules:
    sglsim   hybrid similarity (hybrid_similarity)
    sglalign sentence / character-level NW alignment
    sgldiff  character-level diff
    sgltext  sentence segmentation
    sglword  word-level alignment (jieba)
"""

from sgl._types import (
    SimilarityResult, AlignedPair, AlignmentResult,
    ShuffleGroup, CharDiff, DiffBlock, RichDiffResult,
    WordDiffBlock, WordDiffResult
)

from sgl import sglsim, sglalign, sgldiff, sgltext, sglword
from sgl.sglsim import hybrid_similarity, hybriddetail
from sgl.sglalign import Aligner
from sgl.sgltext import splitsents

__all__ = [
    "sglsim", "sglalign", "sgldiff", "sgltext", "sglword",
    "SimilarityResult", "AlignedPair", "AlignmentResult",
    "ShuffleGroup", "CharDiff", "DiffBlock", "RichDiffResult",
    "WordDiffBlock", "WordDiffResult",
    "hybrid_similarity", "hybriddetail", "Aligner", "splitsents",
]
