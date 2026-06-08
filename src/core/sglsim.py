"""core.sglsim -- hybrid similarity engine (V5)."""

from functools import lru_cache
import math
import numpy as np
from core._types import SimilarityResult

# Default char-set overlap threshold for fast pre-filter.
_DEF_CHAR_TH = 0.1


def _char_ov(x: str, y: str) -> float:
    """Quick char-set overlap: |sx & sy| / min(|sx|, |sy|)."""
    if not x or not y:
        return 0.0
    sx, sy = set(x), set(y)
    com = len(sx & sy)
    den = min(len(sx), len(sy))
    return com / den if den > 0 else 0.0


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def _complexity(s: str) -> float:
    return len(set(s)) / len(s) if len(s) > 0 else 0.0


def _dice_pos(x: str, y: str, n: int) -> tuple[float, list[int], list[int]]:
    gx = [x[i:i + n] for i in range(len(x) - n + 1)] if len(x) >= n else []
    gy = [y[i:i + n] for i in range(len(y) - n + 1)] if len(y) >= n else []
    cx, cy = {}, {}
    for i, g in enumerate(gx): cx.setdefault(g, []).append(i)
    for i, g in enumerate(gy): cy.setdefault(g, []).append(i)
    ov = 0; px, py = [], []
    for g in cx:
        if g in cy:
            c = min(len(cx[g]), len(cy[g]))
            ov += c
            px.extend(cx[g][:c])
            py.extend(cy[g][:c])
    return _safe_div(2.0 * ov, len(gx) + len(gy)), px, py


def _lis_len(arr: list[int]) -> int:
    tails: list[int] = []
    for num in arr:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < num: lo = mid + 1
            else: hi = mid
        if lo == len(tails): tails.append(num)
        else: tails[lo] = num
    return len(tails)


def _roc(x: str, y: str) -> float:
    yp: dict[str, list[int]] = {}
    for i, ch in enumerate(y): yp.setdefault(ch, []).append(i)
    ix: list[int] = []
    ptr = {ch: 0 for ch in yp}
    com = 0
    for ch in x:
        if ch in yp and ptr[ch] < len(yp[ch]):
            ix.append(yp[ch][ptr[ch]])
            ptr[ch] += 1; com += 1
    return (0.0 if com == 0 else _lis_len(ix) / com)


def _lcs_ratio(x: str, y: str) -> float:
    n, m = len(x), len(y)
    if n * m == 0: return 0.0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        xi = x[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if xi == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return (2.0 * prev[m]) / (n + m)


def _dispersion(positions: list[int], text_len: int) -> float:
    if len(positions) <= 1: return 0.5
    std = float(np.std(positions, ddof=0))
    return min(std / (text_len / 2.0 + 1e-9), 1.0)


def _weights(text_len: int, order: int) -> list[float]:
    if order <= 1: return [1.0]
    w = [1.0 / (k * math.log(k + 1)) for k in range(1, order + 1)]
    t = sum(w)
    return [v / t for v in w] if t > 0 else [1.0 / order] * order


@lru_cache(maxsize=4096)
def _cached(x: str, y: str) -> tuple[float, ...]:
    return _compute(x, y)


def _compute(x: str, y: str) -> tuple:
    if not x or not y:
        return (0.0,) * 12

    n_min = min(len(x), len(y))
    n_max = max(len(x), len(y))
    alphabet = len(set(x) | set(y))
    cplx = (_complexity(x) + _complexity(y)) / 2.0
    order = max(1, n_min.bit_length() - 1)
    w = _weights(n_min, order)

    dice_sum = 0.0
    all_px: list[int] = []
    all_py: list[int] = []
    match_cnt = 0
    log_len = (math.log(len(x) + 1) + math.log(len(y) + 1)) / 2.0 + 1.0

    for i, k in enumerate(range(1, order + 1)):
        ds, px, py = _dice_pos(x, y, k)
        dice_sum += w[i] * ds
        if k == 1: match_cnt = len(px)
        if k <= log_len:
            all_px.extend(px); all_py.extend(py)

    lcs = _lcs_ratio(x, y)
    roc = _roc(x, y)

    ln_min = math.log(n_min + 1)
    ln_ctx = math.log(n_max + alphabet + 1)
    sr = _safe_div(ln_min, ln_ctx) * cplx
    base = dice_sum * (1.0 - sr) + lcs * sr

    if all_px and all_py:
        dx = _dispersion(all_px, len(x))
        dy = _dispersion(all_py, len(y))
        dc = 1.0 - abs(dx - dy)
    else:
        dc = 0.0

    er = 1.0 / (alphabet + 1.0)
    cf = _safe_div(match_cnt / n_min - er, 1.0 - er)
    dw = max(0.0, cf) * cplx
    lr = roc ** (1.0 / (cplx + 1e-9))
    gm = _safe_div(ln_min, math.log(n_max + 1))
    lp = (n_min / n_max) ** gm

    final = lp * lr * (base + dc * dw) / (1.0 + dw)
    return (final, dice_sum, lcs, roc, cplx, cplx,
            order, sr, lr, lp, dc, cf)


def hybrid_similarity(x: str, y: str,
                      char_th: float = _DEF_CHAR_TH) -> float:
    # Fast char-set pre-filter at entry point.
    if char_th > 0.0 and _char_ov(x, y) < char_th:
        return 0.0
    return round(float(_cached(x, y)[0]), 4)


def hybriddetail(x: str, y: str,
                 char_th: float = _DEF_CHAR_TH) -> SimilarityResult:
    # Fast char-set pre-filter at entry point.
    if char_th > 0.0 and _char_ov(x, y) < char_th:
        return SimilarityResult(
            score=0.0, dice_score=0.0, lcs_score=0.0, roc_score=0.0,
            complexity_a=0.0, complexity_b=0.0, n_gram_order=0,
            n_gram_weights={}, logic_reward=0.0, len_penalty=0.0,
            dispersion_consistency=0.0, struct_ratio=0.0, confidence=0.0,
        )
    r = _compute(x, y)
    cpx = _complexity(x); cpy = _complexity(y)
    order = r[6]
    w = _weights(min(len(x), len(y)), order)
    return SimilarityResult(
        score=round(r[0], 4), dice_score=round(r[1], 4),
        lcs_score=round(r[2], 4), roc_score=round(r[3], 4),
        complexity_a=round(cpx, 4), complexity_b=round(cpy, 4),
        n_gram_order=order,
        n_gram_weights={k + 1: round(v, 4) for k, v in enumerate(w)},
        logic_reward=round(r[8], 4), len_penalty=round(r[9], 4),
        dispersion_consistency=round(r[10], 4),
        struct_ratio=round(r[7], 4), confidence=round(r[11], 4),
    )


def cacheclear() -> None:
    _cached.cache_clear()


def cacheinfo():
    return _cached.cache_info()
