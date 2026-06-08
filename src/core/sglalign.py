"""
core.sglalign -- Needleman-Wunsch global alignment (NumPy + affine gap + Myers-Miller).

Core API:
  align(seq1, seq2, sim) -- default Myers-Miller (O(min(N,M)) space)
  align(seq1, seq2, sim, linear=False) -- full matrix NW (O(N*M) space)
  alignfb(seq1, seq2, sim) -- forward-backward (full matrix only)
  with_matrices(seq1, seq2, sim) -- full matrix for viz / debug
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable
import math
import numpy as np

from core._types import AlignedPair, AlignmentResult, ShuffleGroup

Sim = Callable[[Any, Any], float]
_STATES = {'M': 0, 'X': 1, 'Y': 2}


class Aligner:
    """Needleman-Wunsch global aligner with affine gap penalties.

    Parameters
    ----------
    gap_open : float, default -1.5
    gap_extend : float, default -0.2
    mm_th : float, default 0.2
        Mismatch threshold -- similarity below this is treated as a mismatch.
    linear : bool, default True
        Use Myers-Miller linear-space algorithm.
    n_workers : int | None, default None
        Number of ProcessPoolExecutor workers (None = serial).
    """

    def __init__(self, gap_open=-1.5, gap_extend=-0.2,
                 mm_th=0.2, linear: bool = True,
                 n_workers: int | None = None):
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.mm_th = mm_th
        self.linear = linear
        self.n_workers = n_workers

    # ========== Entry points ==========

    def align(self, seq1: list[Any], seq2: list[Any], sim: Sim, linear: bool | None = None) -> AlignmentResult:
        """NW global alignment."""
        if (linear is None and self.linear) or linear:
            return self._linear(seq1, seq2, sim)
        return self._full(seq1, seq2, sim)

    def alignfb(self, seq1: list[Any], seq2: list[Any], sim: Sim) -> AlignmentResult:
        """Forward-backward alignment"""
        result = self._full(seq1, seq2, sim)
        result.fb_matrix = self._fb(seq1, seq2, sim)
        return result

    def with_matrices(self, seq1: list[Any], seq2: list[Any], sim: Sim) -> AlignmentResult:
        """Full-matrix alignment returning dp_matrix / M/X/Y"""
        return self._full(seq1, seq2, sim)

    # ========== Full-matrix NW ==========

    def _full(self, seq1: list[Any], seq2: list[Any],
              sim: Sim) -> AlignmentResult:
        n, m = len(seq1), len(seq2)
        g_o, g_e, th = self.gap_open, self.gap_extend, self.mm_th
        M, X, Y = self._nw_fill(np.full, n, m, -np.inf)
        M[0, 0] = 0.0
        X[1:, 0] = g_o + np.arange(n, dtype=np.float64) * g_e
        Y[0, 1:] = g_o + np.arange(m, dtype=np.float64) * g_e
        self._nw_rows(seq1, seq2, M, X, Y, sim)
        combined = np.maximum.reduce([M, X, Y])
        score = float(combined[n, m])
        pairs, path = self._bt(seq1, seq2, M, X, Y, sim)
        return AlignmentResult(
            pairs=pairs, score=score, n_source=n, n_target=m,
            source_seqs=list(seq1), target_seqs=list(seq2),
            dp_matrix=combined.tolist(), matrix_m=M.tolist(),
            matrix_x=X.tolist(), matrix_y=Y.tolist(),
            backtrace_path=path,
            gap_open=g_o, gap_extend=g_e, mismatch_threshold=th,
        )

    @staticmethod
    def _nw_fill(fn, n, m, val):
        """Allocate three (n+1, m+1) float64 matrices."""
        return (fn((n + 1, m + 1), val, dtype=np.float64),
                fn((n + 1, m + 1), val, dtype=np.float64),
                fn((n + 1, m + 1), val, dtype=np.float64))

    def _nw_rows(self, seq1, seq2, M, X, Y, sim):
        """Fill NW matrices row by row (M, X, Y already initialized)."""
        g_o, g_e, th = self.gap_open, self.gap_extend, self.mm_th
        n, m = len(seq1), len(seq2)
        for i in range(1, n + 1):
            si = seq1[i - 1]
            sr = np.array([sim(si, sj) for sj in seq2], dtype=np.float64)
            sc = np.where(sr >= th, sr, g_o * 1.5)
            dm = np.maximum(np.maximum(M[i - 1, :-1], X[i - 1, :-1]), Y[i - 1, :-1])
            M[i, 1:] = dm + sc
            X[i, 1:] = np.maximum(M[i - 1, 1:] + g_o, X[i - 1, 1:] + g_e)
            yrow = np.empty(m + 1, dtype=np.float64)
            yrow[0] = Y[i, 0]
            mo = M[i, :-1] + g_o
            for j in range(1, m + 1):
                yrow[j] = max(mo[j - 1], yrow[j - 1] + g_e)
            Y[i, :] = yrow

    # ========== Forward-backward ==========

    def _fb(self, seq1, seq2, sim):
        n, m = len(seq1), len(seq2)
        g_o, g_e, th = self.gap_open, self.gap_extend, self.mm_th
        fwd = self._fwd_cmb(seq1, seq2, sim)
        total = fwd[n, m]
        BM, BX, BY = self._nw_fill(np.full, n, m, -np.inf)
        BM[n, m] = BX[n, m] = BY[n, m] = 0.0
        for i in range(n - 1, -1, -1):
            BX[i, m] = g_o + (n - 1 - i) * g_e
        for j in range(m - 1, -1, -1):
            BY[n, j] = g_o + (m - 1 - j) * g_e
        for i in range(n - 1, -1, -1):
            si = seq1[i]
            sr = np.array([sim(si, sj) for sj in seq2], dtype=np.float64)
            sc = np.where(sr >= th, sr, g_o * 1.5)
            dm = np.maximum(np.maximum(BM[i + 1, 1:], BX[i + 1, 1:]), BY[i + 1, 1:])
            BM[i, :m] = dm + sc
            BX[i, :m] = np.maximum(BM[i + 1, :m] + g_o, BX[i + 1, :m] + g_e)
            bmo = BM[i, 1:] + g_o
            byrow = np.empty(m + 1, dtype=np.float64)
            byrow[m] = BY[i, m]
            for j in range(m - 1, -1, -1):
                byrow[j] = max(bmo[j], byrow[j + 1] + g_e)
            BY[i, :] = byrow
        return (fwd + np.maximum.reduce([BM, BX, BY]) - total).tolist()

    def _fwd_cmb(self, seq1, seq2, sim):
        n, m = len(seq1), len(seq2)
        M, X, Y = self._nw_fill(np.full, n, m, -np.inf)
        M[0, 0] = 0.0
        X[1:, 0] = self.gap_open + np.arange(n, dtype=np.float64) * self.gap_extend
        Y[0, 1:] = self.gap_open + np.arange(m, dtype=np.float64) * self.gap_extend
        self._nw_rows(seq1, seq2, M, X, Y, sim)
        return np.maximum.reduce([M, X, Y])

    # ========== Reorder-aware alignment ==========

    def reorderalign(self, seq1: list[Any], seq2: list[Any], sim: Sim, threshold: float = 0.3) -> AlignmentResult:
        n, m = len(seq1), len(seq2)
        sm = self._sim_mat(seq1, seq2, sim)
        matches = self._greedy(sm, threshold)
        matches.sort(key=lambda x: x[0])
        tseq = [p[1] for p in matches]
        groups = self._shuffle_grp(matches, tseq)
        shuffled = {i for g in groups for i in g.source_indices}
        mmap = {p[0]: (p[1], p[2]) for p in matches}
        mtgt = {p[1] for p in matches}
        pairs: list[AlignedPair] = []
        si = tj = 0
        while si < n or tj < m:
            if si < n and si in mmap:
                j, s = mmap[si]
                while tj < j:
                    if tj not in mtgt:
                        pairs.append(AlignedPair(None, seq2[tj], 0.0, True, "insert"))
                    tj += 1
                state = "match" if s >= self.mm_th else "mismatch"
                pairs.append(AlignedPair(seq1[si], seq2[j], s, False, state,
                                         is_shuffled=si in shuffled))
                si += 1
                tj = max(tj, j + 1)
            elif si < n:
                pairs.append(AlignedPair(seq1[si], None, 0.0, True, "delete"))
                si += 1
            else:
                if tj not in mtgt:
                    pairs.append(AlignedPair(None, seq2[tj], 0.0, True, "insert"))
                tj += 1
        return AlignmentResult(
            pairs=pairs, score=sum(p.similarity for p in pairs if not p.is_gap),
            n_source=n, n_target=m, source_seqs=list(seq1), target_seqs=list(seq2),
            shuffle_groups=groups or None,
            gap_open=self.gap_open, gap_extend=self.gap_extend,
            mismatch_threshold=self.mm_th,
        )

    def _sim_mat(self, seq1, seq2, sim) -> np.ndarray:
        """Compute similarity matrix -- supports ProcessPoolExecutor parallelization."""
        n, m = len(seq1), len(seq2)
        mat = np.zeros((n, m))

        if self.n_workers is not None and self.n_workers >= 2:
            # Parallel mode: split seq2 into chunks per worker
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {}
                chunk_size = max(1, m // self.n_workers)
                for i, s1 in enumerate(seq1):
                    for chunk_start in range(0, m, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, m)
                        chunk = seq2[chunk_start:chunk_end]
                        future = executor.submit(_sim_chunk, s1, chunk, sim, chunk_start)
                        futures[(i, chunk_start)] = future
                for (i, chunk_start), future in futures.items():
                    chunk_end = min(chunk_start + chunk_size, m)
                    mat[i, chunk_start:chunk_end] = future.result()
        else:
            # Serial mode (original logic)
            for i, s1 in enumerate(seq1):
                mat[i] = [sim(s1, s2) for s2 in seq2]
        return mat

    @staticmethod
    def _greedy(sim_mat: np.ndarray, th: float) -> list[tuple[int, int, float]]:
        n, m = sim_mat.shape
        items = [(i, j, float(sim_mat[i, j])) for i in range(n) for j in range(m) if sim_mat[i, j] >= th]
        items.sort(key=lambda x: -x[2])
        matches: list[tuple[int, int, float]] = []
        used_s, used_t = set(), set()
        for i, j, s in items:
            if i not in used_s and j not in used_t:
                matches.append((i, j, s))
                used_s.add(i); used_t.add(j)
        return matches

    @staticmethod
    def _shuffle_grp(matches: list[tuple[int, int, float]], tseq: list[int]) -> list[ShuffleGroup]:
        if len(matches) < 2:
            return []
        exp = sorted(tseq)
        bad = [k for k in range(len(matches)) if tseq[k] != exp[k]]
        if len(bad) < 2:
            return []
        grps: list[list[int]] = [[bad[0]]]
        for k in bad[1:]:
            if k == grps[-1][-1] + 1:
                grps[-1].append(k)
            else:
                grps.append([k])
        return [ShuffleGroup([matches[k][0] for k in g], [matches[k][1] for k in g])
                for g in grps if len(g) >= 2]

    # ========== Backtrace (full matrix) ==========

    def _bt(self, seq1, seq2, M, X, Y, sim):
        n, m = len(seq1), len(seq2)
        cmb = np.maximum.reduce([M, X, Y])
        pairs: list[AlignedPair] = []
        path: list[tuple[int, int, str]] = []
        i, j = n, m
        curr = int(np.argmax([float(cmb[i, j]),
                              float(X[i, j]) if np.isclose(cmb[i, j], X[i, j]) else -np.inf,
                              float(Y[i, j]) if np.isclose(cmb[i, j], Y[i, j]) else -np.inf]))
        path.append((i, j, "MXY"[curr]))
        while i > 0 or j > 0:
            if curr == 0 and i > 0 and j > 0:
                s = sim(seq1[i - 1], seq2[j - 1])
                sc = s if s >= self.mm_th else self.gap_open * 1.5
                prev = float(M[i, j]) - sc
                pairs.append(AlignedPair(
                    source=seq1[i - 1], target=seq2[j - 1],
                    similarity=s, is_gap=False,
                    state="match" if s >= self.mm_th else "mismatch"))
                cand = [(0, float(M[i - 1, j - 1])),
                        (1, float(X[i - 1, j - 1])),
                        (2, float(Y[i - 1, j - 1]))]
                curr = max(cand, key=lambda t: t[1] if np.isclose(t[1], prev) else -np.inf)[0]
                i -= 1; j -= 1
            elif curr == 1 and i > 0:
                pairs.append(AlignedPair(source=seq1[i - 1], target=None, is_gap=True, state="delete"))
                prev = float(X[i, j])
                curr = 0 if np.isclose(prev, float(M[i - 1, j]) + self.gap_open) else 1
                i -= 1
            elif curr == 2 and j > 0:
                pairs.append(AlignedPair(source=None, target=seq2[j - 1], is_gap=True, state="insert"))
                prev = float(Y[i, j])
                curr = 0 if np.isclose(prev, float(M[i, j - 1]) + self.gap_open) else 2
                j -= 1
            else:
                break
            path.append((i, j, "MXY"[curr]))
        pairs.reverse(); path.reverse()
        return pairs, path

    # ========== Myers-Miller linear-space alignment (O(min(N,M)) space) ==========

    def _linear(self, seq1: list[Any], seq2: list[Any],
                sim: Sim) -> AlignmentResult:
        n, m = len(seq1), len(seq2)
        if n == 0:
            return self._mk(seq1, seq2, [AlignedPair(None, s, 0.0, True, "insert") for s in seq2], 0.0)
        if m == 0:
            return self._mk(seq1, seq2, [AlignedPair(s, None, 0.0, True, "delete") for s in seq1], 0.0)
        top = self._fwd_vecs(seq1, seq2, sim)
        exit_ = ['M','X','Y'][int(np.argmax([top.M[-1], top.X[-1], top.Y[-1]]))]
        pairs = self._mm(seq1, seq2, 0, n, 0, m, sim, enter=None, exit_=exit_)
        score = float(np.maximum.reduce([top.M, top.X, top.Y])[-1])
        return self._mk(seq1, seq2, pairs, score)

    def _mm(self, seq1, seq2, i1, i2, j1, j2, sim, enter: str | None, exit_: str | None):
        """Myers-Miller recursive divide-and-conquer."""
        n = i2 - i1
        m = j2 - j1
        if n == 0:
            return [AlignedPair(None, seq2[j], 0.0, True, "insert") for j in range(j1, j2)]
        if m == 0:
            return [AlignedPair(seq1[i], None, 0.0, True, "delete") for i in range(i1, i2)]
        mid = i1 + n // 2
        if mid == i1 or mid == i2:
            return self._submx(seq1, seq2, i1, i2, j1, j2, sim, enter, exit_)
        fwd = self._fwd_vecs(seq1[i1:mid], seq2[j1:j2], sim, enter)
        bwd = self._fwd_vecs(list(reversed(seq1[mid:i2])),
                             list(reversed(seq2[j1:j2])), sim)
        rM = bwd.M[::-1].copy(); rX = bwd.X[::-1].copy(); rY = bwd.Y[::-1].copy()
        best_j, best_s, best = j1, 'M', -np.inf
        for col in range(m + 1):
            for s, v in [('M', fwd.M[col] + rM[col]),
                         ('X', fwd.X[col] + rX[col]),
                         ('Y', fwd.Y[col] + rY[col])]:
                if v > best:
                    best = v; best_j = j1 + col; best_s = s
        left = self._mm(seq1, seq2, i1, mid, j1, best_j, sim, enter, best_s)
        right = self._mm(seq1, seq2, mid, i2, best_j, j2, sim, best_s, exit_)
        return left + right

    def _submx(self, seq1, seq2, i1, i2, j1, j2, sim, enter, exit_):
        """Exact small-matrix DP with forced exit state."""
        n, m = i2 - i1, j2 - j1
        g_o, g_e, th = self.gap_open, self.gap_extend, self.mm_th
        sub1, sub2 = seq1[i1:i2], seq2[j1:j2]
        M, X, Y = self._nw_fill(np.full, n, m, -np.inf)
        if enter == 'X':
            X[0, 0] = 0.0; Y[0, 1:] = g_o + np.arange(m, dtype=np.float64) * g_e
        elif enter == 'Y':
            Y[0, 0] = 0.0; Y[0, 1:] = g_e * np.arange(m, dtype=np.float64)
        else:
            M[0, 0] = 0.0; Y[0, 1:] = g_o + np.arange(m, dtype=np.float64) * g_e
        self._nw_rows(sub1, sub2, M, X, Y, sim)
        curr = _STATES[exit_]
        ii, jj = n, m
        pairs: list[AlignedPair] = []
        while ii > 0 or jj > 0:
            if curr == 0 and ii > 0 and jj > 0:
                s = sim(sub1[ii - 1], sub2[jj - 1])
                sc = s if s >= th else g_o * 1.5
                prev = float(M[ii, jj]) - sc
                pairs.append(AlignedPair(
                    source=sub1[ii - 1], target=sub2[jj - 1],
                    similarity=s, is_gap=False,
                    state="match" if s >= th else "mismatch"))
                cand = [(0, float(M[ii - 1, jj - 1])),
                        (1, float(X[ii - 1, jj - 1])),
                        (2, float(Y[ii - 1, jj - 1]))]
                curr = max(cand, key=lambda t: t[1] if np.isclose(t[1], prev) else -np.inf)[0]
                ii -= 1; jj -= 1
            elif curr == 1 and ii > 0:
                pairs.append(AlignedPair(source=sub1[ii - 1], target=None, is_gap=True, state="delete"))
                prev = float(X[ii, jj])
                curr = 0 if np.isclose(prev, float(M[ii - 1, jj]) + g_o) else 1
                ii -= 1
            elif curr == 2 and jj > 0:
                pairs.append(AlignedPair(source=None, target=sub2[jj - 1], is_gap=True, state="insert"))
                prev = float(Y[ii, jj])
                curr = 0 if np.isclose(prev, float(M[ii, jj - 1]) + g_o) else 2
                jj -= 1
            else:
                break
        pairs.reverse()
        return pairs

    def _fwd_vecs(self, seq1, seq2, sim, enter: str | None = None):
        """O(m) forward DP; returns (_Vectors) at the last row.

        ``enter``: None = standard, 'X' = continuing deletion, 'Y' = continuing insertion.
        """
        g_o, g_e, th = self.gap_open, self.gap_extend, self.mm_th
        m = len(seq2)
        M = np.full(m + 1, -np.inf, dtype=np.float64)
        X = np.full(m + 1, -np.inf, dtype=np.float64)
        Y = np.full(m + 1, -np.inf, dtype=np.float64)
        if enter == 'X':
            X[0] = 0.0; Y[1:] = g_o + np.arange(m, dtype=np.float64) * g_e
        elif enter == 'Y':
            Y[0] = 0.0; Y[1:] = g_e * np.arange(m, dtype=np.float64)
        else:
            M[0] = 0.0; Y[1:] = g_o + np.arange(m, dtype=np.float64) * g_e
        for si in seq1:
            sr = np.array([sim(si, sj) for sj in seq2], dtype=np.float64)
            sc = np.where(sr >= th, sr, g_o * 1.5)
            dm = np.maximum(np.maximum(M[:-1], X[:-1]), Y[:-1])
            nM = np.empty(m + 1, dtype=np.float64)
            nM[0] = -np.inf; nM[1:] = dm + sc
            nX = np.empty(m + 1, dtype=np.float64)
            nX[0] = M[0] + g_o if M[0] > -np.inf else X[0] + g_e
            nX[1:] = np.maximum(M[1:] + g_o, X[1:] + g_e)
            nY = np.empty(m + 1, dtype=np.float64)
            nY[0] = -np.inf
            mo = nM[:-1] + g_o
            for j in range(1, m + 1):
                nY[j] = max(mo[j - 1], nY[j - 1] + g_e)
            M, X, Y = nM, nX, nY
        return _Vectors(M, X, Y)

    def _mk(self, seq1, seq2, pairs, score) -> AlignmentResult:
        return AlignmentResult(
            pairs=pairs, score=score,
            n_source=len(seq1), n_target=len(seq2),
            source_seqs=list(seq1), target_seqs=list(seq2),
            gap_open=self.gap_open, gap_extend=self.gap_extend,
            mismatch_threshold=self.mm_th,
        )


class _Vectors:
    """Triplet of M, X, Y DP vectors."""
    __slots__ = ('M', 'X', 'Y')
    def __init__(self, M, X, Y):
        self.M = M; self.X = X; self.Y = Y


def _sim_chunk(si: Any, chunk: list[Any], sim: Sim, offset: int = 0) -> np.ndarray:
    """Parallel worker: compute similarity of si against a chunk."""
    return np.array([sim(si, sj) for sj in chunk], dtype=np.float64)
