/**
 * sgl_js  V5 -- hybrid similarity + Myers-Miller linear-space alignment + Float64Array
 */

// ========== Util ==========
const safeDiv = (a, b) => b !== 0 ? a / b : 0;
const complexity = s => s.length > 0 ? new Set(s).size / s.length : 0;
const _DEF_CHAR_TH = 0.1;

function _charOv(x, y) {
    if (!x || !y) return 0.0;
    const sx = new Set(x), sy = new Set(y);
    let com = 0;
    if (sx.size <= sy.size) {
        for (const ch of sx) { if (sy.has(ch)) com++; }
    } else {
        for (const ch of sy) { if (sx.has(ch)) com++; }
    }
    const den = Math.min(sx.size, sy.size);
    return den > 0 ? com / den : 0.0;
}

// ========== sgltext -- sentence segmentation ==========
const SENT_RE = /([\u3002\uff01\uff1f.!?\u2026]+[\u201c\u201d\u2018\u2019\"\'\u300a\u300b\u3008\u3009\u300c\u300d]*|\n+)/g;

function splitSents(text, minlen = 1, filterpunct = true) {
    if (!text || !text.trim()) return [];
    text = text.trim();
    const parts = text.split(SENT_RE);
    const sents = [];
    for (let i = 0; i < parts.length - 1; i += 2) {
        const c = (parts[i] + (parts[i + 1] || '')).trim();
        if (c) sents.push(c);
    }
    if (parts.length % 2 === 1 && parts[parts.length - 1].trim()) {
        sents.push(parts[parts.length - 1].trim());
    }
    const result = [];
    for (const s of sents) {
        const cleaned = s.replace(/\s+/g, ' ').trim();
        if (cleaned.length <= minlen) continue;
        if (filterpunct && !/[\w\u4e00-\u9fff]/.test(cleaned)) continue;
        result.push(cleaned);
    }
    return result.length > 0 ? result : [text];
}
const countSents = text => splitSents(text).length;

// ========== sglsim -- hybrid similarity ==========
const _simCache = new Map();

function _dicePos(x, y, n) {
    const gx = [], gy = [];
    for (let i = 0; i <= x.length - n; i++) gx.push(x.substring(i, i + n));
    for (let i = 0; i <= y.length - n; i++) gy.push(y.substring(i, i + n));
    const cx = {}, cy = {};
    gx.forEach((g, i) => { if (!cx[g]) cx[g] = []; cx[g].push(i); });
    gy.forEach((g, i) => { if (!cy[g]) cy[g] = []; cy[g].push(i); });
    let ov = 0; const px = [], py = [];
    for (const g in cx) {
        if (cy[g]) {
            const c = Math.min(cx[g].length, cy[g].length);
            ov += c;
            px.push(...cx[g].slice(0, c));
            py.push(...cy[g].slice(0, c));
        }
    }
    return [safeDiv(2.0 * ov, gx.length + gy.length), px, py];
}

const _lisLen = arr => {
    const tails = [];
    for (const num of arr) {
        let lo = 0, hi = tails.length;
        while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (tails[mid] < num) lo = mid + 1; else hi = mid;
        }
        if (lo === tails.length) tails.push(num); else tails[lo] = num;
    }
    return tails.length;
};

function _roc(x, y) {
    const yp = {};
    for (let i = 0; i < y.length; i++) {
        if (!yp[y[i]]) yp[y[i]] = []; yp[y[i]].push(i);
    }
    const ix = []; const ptr = {}; let com = 0;
    for (const ch of x) {
        if (yp[ch] && (ptr[ch] || 0) < yp[ch].length) {
            ix.push(yp[ch][ptr[ch] || 0]);
            ptr[ch] = (ptr[ch] || 0) + 1; com++;
        }
    }
    return com === 0 ? 0.0 : _lisLen(ix) / com;
}

function _lcsRatio(x, y) {
    const n = x.length, m = y.length;
    if (n * m === 0) return 0.0;
    let prev = new Float64Array(m + 1);
    for (let i = 1; i <= n; i++) {
        const cur = new Float64Array(m + 1);
        const xi = x[i - 1];
        for (let j = 1; j <= m; j++) {
            cur[j] = xi === y[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
        }
        prev = cur;
    }
    return (2.0 * prev[m]) / (n + m);
}

const _dispersion = (positions, tlen) => {
    if (positions.length <= 1) return 0.5;
    const mean = positions.reduce((a, b) => a + b, 0) / positions.length;
    const vr = positions.reduce((s, p) => s + Math.pow(p - mean, 2), 0) / positions.length;
    return Math.min(Math.sqrt(vr) / (tlen / 2.0 + 1e-9), 1.0);
};

const _weights = (tlen, order) => {
    if (order <= 1) return [1.0];
    const w = [];
    for (let k = 1; k <= order; k++) w.push(1.0 / (k * Math.log(k + 1)));
    const t = w.reduce((a, b) => a + b, 0);
    return t > 0 ? w.map(v => v / t) : new Array(order).fill(1.0 / order);
};

function _compute(x, y, charTh = _DEF_CHAR_TH) {
    if (!x || !y) return { fin: 0.0, dSum: 0.0, lcs: 0.0, roc: 0.0, cpxX: 0.0, cpxY: 0.0, order: 1, sr: 0.0, lr: 0.0, lp: 0.0, dc: 0.0, cf: 0.0 };
    // Fast char-set pre-filter
    if (charTh > 0.0 && _charOv(x, y) < charTh) {
        return { fin: 0.0, dSum: 0.0, lcs: 0.0, roc: 0.0, cpxX: 0.0, cpxY: 0.0, order: 1, sr: 0.0, lr: 0.0, lp: 0.0, dc: 0.0, cf: 0.0 };
    }
    const nMin = Math.min(x.length, y.length);
    const nMax = Math.max(x.length, y.length);
    const alphabet = new Set(x + y).size;
    const cplx = (complexity(x) + complexity(y)) / 2.0;
    const order = Math.max(1, nMin.toString(2).length - 1);
    const w = _weights(nMin, order);
    let dSum = 0.0; const allPx = [], allPy = []; let mCnt = 0;
    const logLen = (Math.log(x.length + 1) + Math.log(y.length + 1)) / 2.0 + 1.0;
    for (let i = 0; i < order; i++) {
        const k = i + 1;
        const [ds, px, py] = _dicePos(x, y, k);
        dSum += w[i] * ds;
        if (k === 1) mCnt = px.length;
        if (k <= logLen) { allPx.push(...px); allPy.push(...py); }
    }
    const lcs = _lcsRatio(x, y);
    const roc = _roc(x, y);
    const lnMin = Math.log(nMin + 1);
    const lnCtx = Math.log(nMax + alphabet + 1);
    const sr = safeDiv(lnMin, lnCtx) * cplx;
    const base = dSum * (1.0 - sr) + lcs * sr;
    let dc = 0.0;
    if (allPx.length > 0 && allPy.length > 0) {
        const dx = _dispersion(allPx, x.length);
        const dy = _dispersion(allPy, y.length);
        dc = 1.0 - Math.abs(dx - dy);
    }
    const er = 1.0 / (alphabet + 1.0);
    const cf = safeDiv(mCnt / nMin - er, 1.0 - er);
    const dw = Math.max(0.0, cf) * cplx;
    const lr = Math.pow(roc, 1.0 / (cplx + 1e-9));
    const gm = safeDiv(lnMin, Math.log(nMax + 1));
    const lp = Math.pow(nMin / nMax, gm);
    const fin = lp * lr * (base + dc * dw) / (1.0 + dw);
    return { fin, dSum, lcs, roc, cpxX: complexity(x), cpxY: complexity(y), order, sr, lr, lp, dc, cf };
}

function hybridSimilarity(x, y, charTh = _DEF_CHAR_TH) {
    const key = `sgl_${x.length}_${y.length}_${x}_${y}_${charTh}`;
    if (_simCache.has(key)) { _simCache._hits = (_simCache._hits || 0) + 1; return _simCache.get(key); }
    _simCache._misses = (_simCache._misses || 0) + 1;
    const r = _compute(x, y, charTh);
    const score = parseFloat(r.fin.toFixed(4));
    _simCache.set(key, score);
    return score;
}

function hybridDetail(x, y, charTh = _DEF_CHAR_TH) {
    const r = _compute(x, y, charTh);
    const w = _weights(Math.min(x.length, y.length), r.order);
    const obj = {
        score: parseFloat(r.fin.toFixed(4)), diceScore: parseFloat(r.dSum.toFixed(4)),
        lcsScore: parseFloat(r.lcs.toFixed(4)), rocScore: parseFloat(r.roc.toFixed(4)),
        complexityA: parseFloat(r.cpxX.toFixed(4)), complexityB: parseFloat(r.cpxY.toFixed(4)),
        nGramOrder: r.order,
        nGramWeights: Object.fromEntries(w.map((v, i) => [i + 1, parseFloat(v.toFixed(4))])),
        logicReward: parseFloat(r.lr.toFixed(4)), lenPenalty: parseFloat(r.lp.toFixed(4)),
        dispersionConsistency: parseFloat(r.dc.toFixed(4)),
        structRatio: parseFloat(r.sr.toFixed(4)), confidence: parseFloat(r.cf.toFixed(4))
    };
    obj.toDict = () => ({ score: obj.score, dice_score: obj.diceScore, lcs_score: obj.lcsScore, roc_score: obj.rocScore, complexity_a: obj.complexityA, complexity_b: obj.complexityB, n_gram_order: obj.nGramOrder, n_gram_weights: obj.nGramWeights, logic_reward: obj.logicReward, len_penalty: obj.lenPenalty, dispersion_consistency: obj.dispersionConsistency, struct_ratio: obj.structRatio, confidence: obj.confidence });
    obj.toJson = (indent = 2) => JSON.stringify(obj.toDict(), null, indent);
    obj.summary = () => ({ score: obj.score, dice: obj.diceScore, lcs: obj.lcsScore, roc: obj.rocScore, reward: obj.logicReward });
    return obj;
}

const cacheClear = () => _simCache.clear();
const cacheInfo = () => ({ currsize: _simCache.size, hits: _simCache._hits || 0, misses: _simCache._misses || 0 });

// ========== sglalign -- NW alignment (full matrix + Myers-Miller + Float64Array) ==========
const _F64_INF = (len) => { const a = new Float64Array(len); a.fill(-Infinity); return a; };

class Aligner {
    constructor(gapOpen = -1.5, gapExtend = -0.2, mmTh = 0.2, linear = true) {
        this.gapOpen = gapOpen; this.gapExtend = gapExtend; this.mmTh = mmTh; this.linear = linear;
    }

    align(seq1, seq2, sim, linearOverride = null) {
        const use = linearOverride !== null ? linearOverride : this.linear;
        return use ? this._linear(seq1, seq2, sim) : this._full(seq1, seq2, sim);
    }

    // ========== Full matrix NW ==========
    _full(seq1, seq2, sim) {
        const n = seq1.length, m = seq2.length;
        const gO = this.gapOpen, gE = this.gapExtend, th = this.mmTh;
        const M = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        const X = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        const Y = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        M[0][0] = 0.0;
        for (let i = 1; i <= n; i++) X[i][0] = gO + (i - 1) * gE;
        for (let j = 1; j <= m; j++) Y[0][j] = gO + (j - 1) * gE;
        for (let i = 1; i <= n; i++) {
            const si = seq1[i - 1];
            const sr = seq2.map(sj => sim(si, sj));
            const sc = sr.map(s => s >= th ? s : gO * 1.5);
            for (let j = 1; j <= m; j++) {
                M[i][j] = Math.max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + sc[j - 1];
            }
            for (let j = 1; j <= m; j++) {
                X[i][j] = Math.max(M[i - 1][j] + gO, X[i - 1][j] + gE);
            }
            const mo = M[i].slice(0, m).map(v => v + gO);
            const yr = [Y[i][0]];
            for (let j = 1; j <= m; j++) yr.push(Math.max(mo[j - 1], yr[j - 1] + gE));
            Y[i] = yr;
        }
        const cmb = Array(n + 1).fill(null).map((_, i) => Array(m + 1).fill(null).map((_, j) => Math.max(M[i][j], X[i][j], Y[i][j])));
        const score = cmb[n][m];
        const [pairs, path] = this._bt(seq1, seq2, M, X, Y, sim);
        return this._result(pairs, score, seq1, seq2, cmb, M, X, Y, path);
    }

    // ========== Myers-Miller linear-space ==========
    _linear(seq1, seq2, sim) {
        const n = seq1.length, m = seq2.length;
        if (n === 0) return this._mk(seq1, seq2, seq2.map(s => ({ source: null, target: s, similarity: 0.0, isGap: true, state: 'insert', isShuffled: false })), 0.0);
        if (m === 0) return this._mk(seq1, seq2, seq1.map(s => ({ source: s, target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false })), 0.0);
        const top = this._fwdVecs(seq1, seq2, sim, null);
        const exit_ = ['M','X','Y'][[top.M[m], top.X[m], top.Y[m]].indexOf(Math.max(top.M[m], top.X[m], top.Y[m]))];
        const pairs = this._mm(seq1, seq2, 0, n, 0, m, sim, null, exit_);
        const score = Math.max(top.M[m], top.X[m], top.Y[m]);
        return this._mk(seq1, seq2, pairs, score);
    }

    _mm(seq1, seq2, i1, i2, j1, j2, sim, enter, exit_) {
        const n = i2 - i1, m = j2 - j1;
        if (n === 0) { const ps = []; for (let j = j1; j < j2; j++) ps.push({ source: null, target: seq2[j], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false }); return ps; }
        if (m === 0) { const ps = []; for (let i = i1; i < i2; i++) ps.push({ source: seq1[i], target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false }); return ps; }
        if (n <= 4 || m <= 4) return this._submx(seq1, seq2, i1, i2, j1, j2, sim, enter, exit_);
        const mid = i1 + Math.floor(n / 2);
        const lSeq = seq1.slice(i1, mid), sSeq2 = seq2.slice(j1, j2);
        const fwd = this._fwdVecs(lSeq, sSeq2, sim, enter);
        const rSeq = seq1.slice(mid, i2).reverse(), rSeq2 = sSeq2.slice().reverse();
        const bwd = this._fwdVecs(rSeq, rSeq2, sim, null);
        const rM = new Float64Array(m + 1), rX = new Float64Array(m + 1), rY = new Float64Array(m + 1);
        for (let k = 0; k <= m; k++) { rM[k] = bwd.M[m - k]; rX[k] = bwd.X[m - k]; rY[k] = bwd.Y[m - k]; }
        let bestJ = j1, bestS = 'M', best = -Infinity;
        for (let j = j1; j <= j2; j++) {
            const col = j - j1;
            for (const [s, v] of [['M', fwd.M[col] + rM[col]], ['X', fwd.X[col] + rX[col]], ['Y', fwd.Y[col] + rY[col]]]) {
                if (v > best) { best = v; bestJ = j; bestS = s; }
            }
        }
        const left = this._mm(seq1, seq2, i1, mid, j1, bestJ, sim, enter, bestS);
        const right = this._mm(seq1, seq2, mid, i2, bestJ, j2, sim, bestS, exit_);
        return left.concat(right);
    }

    _submx(seq1, seq2, i1, i2, j1, j2, sim, enter, exit_) {
        const n = i2 - i1, m = j2 - j1;
        const gO = this.gapOpen, gE = this.gapExtend, th = this.mmTh;
        const sub1 = seq1.slice(i1, i2), sub2 = seq2.slice(j1, j2);
        const M = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        const X = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        const Y = Array(n + 1).fill(null).map(() => Array(m + 1).fill(-Infinity));
        if (enter === 'X') { X[0][0] = 0.0; for (let j = 1; j <= m; j++) Y[0][j] = gO + (j - 1) * gE; }
        else if (enter === 'Y') { Y[0][0] = 0.0; for (let j = 1; j <= m; j++) Y[0][j] = j * gE; }
        else { M[0][0] = 0.0; for (let j = 1; j <= m; j++) Y[0][j] = gO + (j - 1) * gE; }
        for (let i = 1; i <= n; i++) {
            const si = sub1[i - 1];
            const sr = sub2.map(sj => sim(si, sj));
            const sc = sr.map(s => s >= th ? s : gO * 1.5);
            for (let j = 1; j <= m; j++) M[i][j] = Math.max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]) + sc[j - 1];
            for (let j = 1; j <= m; j++) X[i][j] = Math.max(M[i - 1][j] + gO, X[i - 1][j] + gE);
            const mo = M[i].slice(0, m).map(v => v + gO);
            const yr = [-Infinity];
            for (let j = 1; j <= m; j++) yr.push(Math.max(mo[j - 1], yr[j - 1] + gE));
            Y[i] = yr;
        }
        let curr = {M:0, X:1, Y:2}[exit_], ii = n, jj = m;
        const pairs = [];
        while (ii > 0 || jj > 0) {
            if (curr === 0 && ii > 0 && jj > 0) {
                const s = sim(sub1[ii - 1], sub2[jj - 1]);
                const sc = s >= th ? s : gO * 1.5;
                const prev = M[ii][jj] - sc;
                pairs.unshift({ source: sub1[ii - 1], target: sub2[jj - 1], similarity: s, isGap: false, state: s >= th ? 'match' : 'mismatch', isShuffled: false });
                const vals = [[0, M[ii - 1][jj - 1]], [1, X[ii - 1][jj - 1]], [2, Y[ii - 1][jj - 1]]];
                curr = vals.reduce((a, b) => Math.abs(b[1] - prev) < 1e-10 && b[1] > a[1] ? b : a, [0, -Infinity])[0];
                ii--; jj--;
            } else if (curr === 1 && ii > 0) {
                pairs.unshift({ source: sub1[ii - 1], target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false });
                curr = Math.abs(X[ii][jj] - (M[ii - 1][jj] + gO)) < 1e-10 ? 0 : 1;
                ii--;
            } else if (curr === 2 && jj > 0) {
                pairs.unshift({ source: null, target: sub2[jj - 1], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false });
                curr = Math.abs(Y[ii][jj] - (M[ii][jj - 1] + gO)) < 1e-10 ? 0 : 2;
                jj--;
            } else break;
        }
        return pairs;
    }

    _fwdVecs(seq1, seq2, sim, enter) {
        const gO = this.gapOpen, gE = this.gapExtend, th = this.mmTh, m = seq2.length;
        let M = _F64_INF(m + 1), X = _F64_INF(m + 1), Y = _F64_INF(m + 1);
        if (enter === 'X') { X[0] = 0.0; for (let j = 1; j <= m; j++) Y[j] = gO + (j - 1) * gE; }
        else if (enter === 'Y') { Y[0] = 0.0; for (let j = 1; j <= m; j++) Y[j] = j * gE; }
        else { M[0] = 0.0; for (let j = 1; j <= m; j++) Y[j] = gO + (j - 1) * gE; }
        for (let ii = 0; ii < seq1.length; ii++) {
            const si = seq1[ii];
            const sr = new Float64Array(m); for (let j = 0; j < m; j++) sr[j] = sim(si, seq2[j]);
            const nM = _F64_INF(m + 1), nX = _F64_INF(m + 1), nY = _F64_INF(m + 1);
            for (let j = 1; j <= m; j++) nM[j] = Math.max(M[j - 1], X[j - 1], Y[j - 1]) + (sr[j - 1] >= th ? sr[j - 1] : gO * 1.5);
            nX[0] = M[0] > -1e300 ? M[0] + gO : X[0] + gE;
            for (let j = 1; j <= m; j++) nX[j] = Math.max(M[j] + gO, X[j] + gE);
            const mo = nM.slice(0, m).map(v => v + gO);
            nY[0] = -Infinity;
            for (let j = 1; j <= m; j++) nY[j] = Math.max(mo[j - 1], nY[j - 1] + gE);
            M = nM; X = nX; Y = nY;
        }
        return { M, X, Y };
    }

    _mk(seq1, seq2, pairs, score) {
        const r = { pairs, score, nSource: seq1.length, nTarget: seq2.length, sourceSeqs: [...seq1], targetSeqs: [...seq2], dpMatrix: null, matrixM: null, matrixX: null, matrixY: null, backtracePath: null, fbMatrix: null, shuffleGroups: null, gapOpen: this.gapOpen, gapExtend: this.gapExtend, mismatchThreshold: this.mmTh };
        r.toDict = () => ({ pairs: r.pairs.map(p => ({ source: p.source, target: p.target, similarity: p.similarity, isGap: p.isGap, state: p.state, isShuffled: p.isShuffled })), score: r.score, n_source: r.nSource, n_target: r.nTarget, source_seqs: r.sourceSeqs, target_seqs: r.targetSeqs, gap_open: r.gapOpen, gap_extend: r.gapExtend, mismatch_threshold: r.mismatchThreshold });
        r.toJson = (indent = 2) => JSON.stringify(r.toDict(), null, indent);
        r.summary = () => ({ score: parseFloat(r.score.toFixed(4)), n_pairs: r.pairs.length, matches: r.pairs.filter(p => p.state === 'match').length });
        return r;
    }

    _result(pairs, score, seq1, seq2, cmb, M, X, Y, path) {
        const r = { pairs, score, nSource: seq1.length, nTarget: seq2.length, sourceSeqs: [...seq1], targetSeqs: [...seq2], dpMatrix: cmb, matrixM: M, matrixX: X, matrixY: Y, backtracePath: path, fbMatrix: null, shuffleGroups: null, gapOpen: this.gapOpen, gapExtend: this.gapExtend, mismatchThreshold: this.mmTh };
        r.toDict = () => ({ pairs: r.pairs.map(p => ({ source: p.source, target: p.target, similarity: p.similarity, isGap: p.isGap, state: p.state, isShuffled: p.isShuffled })), score: r.score, n_source: r.nSource, n_target: r.nTarget, source_seqs: r.sourceSeqs, target_seqs: r.targetSeqs, dp_matrix: r.dpMatrix, matrix_m: r.matrixM, matrix_x: r.matrixX, matrix_y: r.matrixY, backtrace_path: r.backtracePath, gap_open: r.gapOpen, gap_extend: r.gapExtend, mismatch_threshold: r.mismatchThreshold });
        r.toJson = (indent = 2) => JSON.stringify(r.toDict(), null, indent);
        r.summary = () => ({ score: parseFloat(r.score.toFixed(4)), n_pairs: r.pairs.length, matches: r.pairs.filter(p => p.state === 'match').length });
        return r;
    }

    _bt(seq1, seq2, M, X, Y, simFunc) {
        const n = seq1.length, m = seq2.length, pairs = [], path = [];
        let i = n, j = m;
        while (i > 0 || j > 0) {
            if (i === 0) { pairs.unshift({ source: null, target: seq2[j - 1], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false }); path.unshift([i, j, 'Y']); j--; }
            else if (j === 0) { pairs.unshift({ source: seq1[i - 1], target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false }); path.unshift([i, j, 'X']); i--; }
            else {
                const cmb = Math.max(M[i][j], X[i][j], Y[i][j]);
                if (Math.abs(M[i][j] - cmb) < 1e-10) {
                    const s = simFunc(seq1[i - 1], seq2[j - 1]);
                    pairs.unshift({ source: seq1[i - 1], target: seq2[j - 1], similarity: s, isGap: false, state: s >= this.mmTh ? 'match' : 'mismatch', isShuffled: false });
                    path.unshift([i, j, 'M']); i--; j--;
                } else if (Math.abs(X[i][j] - cmb) < 1e-10) {
                    pairs.unshift({ source: seq1[i - 1], target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false });
                    path.unshift([i, j, 'X']); i--;
                } else {
                    pairs.unshift({ source: null, target: seq2[j - 1], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false });
                    path.unshift([i, j, 'Y']); j--;
                }
            }
        }
        path.unshift([0, 0, 'M']);
        return [pairs, path];
    }

    _fb(seq1, seq2, sim) {
        const n = seq1.length, m = seq2.length;
        const fb = Array(n + 1).fill(null).map(() => Array(m + 1).fill(0.0));
        for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) fb[i][j] = fb[i - 1][j - 1] + sim(seq1[i - 1], seq2[j - 1]);
        return fb;
    }

    alignfb(seq1, seq2, sim) { const r = this._full(seq1, seq2, sim); r.fbMatrix = this._fb(seq1, seq2, sim); return r; }
    withMatrices(seq1, seq2, sim) { return this._full(seq1, seq2, sim); }

    reorderAlign(seq1, seq2, sim, threshold = 0.3) {
        const n = seq1.length, m = seq2.length;
        const sm = this._simMat(seq1, seq2, sim);
        const matches = this._greedy(sm, threshold);
        matches.sort((a, b) => a[0] - b[0]);
        const tseq = matches.map(p => p[1]);
        const groups = this._shuffleGrp(matches, tseq);
        const shuffled = new Set(groups.flatMap(g => g.sourceIndices));
        const mmap = new Map(matches.map(p => [p[0], [p[1], p[2]]]));
        const mtgt = new Set(matches.map(p => p[1]));
        const pairs = []; let si = 0, tj = 0;
        while (si < n || tj < m) {
            if (si < n && mmap.has(si)) {
                const [j, s] = mmap.get(si);
                while (tj < j) { if (!mtgt.has(tj)) pairs.push({ source: null, target: seq2[tj], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false }); tj++; }
                pairs.push({ source: seq1[si], target: seq2[j], similarity: s, isGap: false, state: s >= this.mmTh ? 'match' : 'mismatch', isShuffled: shuffled.has(si) });
                si++; tj = Math.max(tj, j + 1);
            } else if (si < n) { pairs.push({ source: seq1[si], target: null, similarity: 0.0, isGap: true, state: 'delete', isShuffled: false }); si++; }
            else { if (!mtgt.has(tj)) pairs.push({ source: null, target: seq2[tj], similarity: 0.0, isGap: true, state: 'insert', isShuffled: false }); tj++; }
        }
        const score = pairs.filter(p => !p.isGap).reduce((sum, p) => sum + p.similarity, 0);
        const r = { pairs, score, nSource: n, nTarget: m, sourceSeqs: [...seq1], targetSeqs: [...seq2], dpMatrix: null, matrixM: null, matrixX: null, matrixY: null, backtracePath: null, fbMatrix: null, shuffleGroups: groups, gapOpen: this.gapOpen, gapExtend: this.gapExtend, mismatchThreshold: this.mmTh };
        r.toDict = () => ({ pairs: r.pairs.map(p => ({ source: p.source, target: p.target, similarity: p.similarity, isGap: p.isGap, state: p.state, isShuffled: p.isShuffled })), score: r.score, n_source: r.nSource, n_target: r.nTarget, source_seqs: r.sourceSeqs, target_seqs: r.targetSeqs, shuffle_groups: r.shuffleGroups.map(g => ({ source_indices: g.sourceIndices, target_indices: g.targetIndices })), gap_open: r.gapOpen, gap_extend: r.gapExtend, mismatch_threshold: r.mismatchThreshold });
        r.toJson = (indent = 2) => JSON.stringify(r.toDict(), null, indent);
        r.summary = () => ({ score: parseFloat(r.score.toFixed(4)), n_pairs: r.pairs.length, shuffle_groups: r.shuffleGroups.length });
        return r;
    }

    _simMat(seq1, seq2, sim) { const n = seq1.length, m = seq2.length; const mat = Array(n).fill(null).map(() => Array(m).fill(0)); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) mat[i][j] = sim(seq1[i], seq2[j]); return mat; }
    _greedy(sm, th) { const n = sm.length, m = sm[0].length, flat = []; for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) if (sm[i][j] >= th) flat.push([i, j, sm[i][j]]); flat.sort((a, b) => b[2] - a[2]); const ms = [], us = new Set(), ut = new Set(); for (const [i, j, s] of flat) { if (!us.has(i) && !ut.has(j)) { ms.push([i, j, s]); us.add(i); ut.add(j); } } return ms; }
    _shuffleGrp(matches, tseq) { if (matches.length < 2) return []; const exp = [...tseq].sort((a, b) => a - b); const bad = []; for (let k = 0; k < matches.length; k++) if (tseq[k] !== exp[k]) bad.push(k); if (bad.length < 2) return []; const grps = [[bad[0]]]; for (let i = 1; i < bad.length; i++) { if (bad[i] === grps[grps.length - 1][grps[grps.length - 1].length - 1] + 1) grps[grps.length - 1].push(bad[i]); else grps.push([bad[i]]); } return grps.filter(g => g.length >= 2).map(g => ({ sourceIndices: g.map(k => matches[k][0]), targetIndices: g.map(k => matches[k][1]) })); }
}

// ========== sgldiff -- character-level diff ==========
const _CHAR_SIM = (c1, c2) => c1 === c2 ? 1.0 : 0.0;

class CharLevelAligner {
    constructor() { this.aligner = new Aligner(-0.8, -0.8, 0.1, false); }
    alignChars(text1, text2) {
        if (!text1) return text2.split('').map(c => ({ source: null, target: c, diffType: 'insert' }));
        if (!text2) return text1.split('').map(c => ({ source: c, target: null, diffType: 'delete' }));
        const raw = this.aligner.align(text1.split(''), text2.split(''), _CHAR_SIM).pairs;
        const refined = []; let i = 0, n = raw.length;
        while (i < n) {
            const p = raw[i];
            if (p.source && p.target && p.source === p.target) { refined.push({ source: p.source, target: p.target, diffType: 'equal' }); i++; continue; }
            const orig = [], rew = [];
            while (i < n) { const cp = raw[i]; if (cp.source && cp.target && cp.source === cp.target) break; if (cp.source) orig.push(cp.source); if (cp.target) rew.push(cp.target); i++; }
            for (const c of orig) refined.push({ source: c, target: null, diffType: 'delete' });
            for (const c of rew) refined.push({ source: null, target: c, diffType: 'insert' });
        }
        return refined;
    }
}

const ANCHOR = 0.80;

function richDiff(aln, simFunc = null, anchorTh = ANCHOR) {
    const ca = new CharLevelAligner();
    const anchor = (s1, s2) => !s1 || !s2 ? false : s1 === s2 || (simFunc !== null && simFunc(s1, s2) >= anchorTh);
    const blocks = []; let i = 0, n = aln.length;
    while (i < n) {
        const p = aln[i];
        if (anchor(p.source, p.target)) {
            if (p.source === p.target) blocks.push({ blockType: 'equal', source: p.source, target: p.target, charDiffs: p.source.split('').map(c => ({ source: c, target: c, diffType: 'equal' })) });
            else blocks.push({ blockType: 'modify', source: p.source, target: p.target, charDiffs: ca.alignChars(p.source, p.target) });
            i++; continue;
        }
        const orig = [], rew = [];
        while (i < n) { const cp = aln[i]; if (anchor(cp.source, cp.target)) break; if (cp.source) orig.push(cp.source); if (cp.target) rew.push(cp.target); i++; }
        const fo = orig.join(''), fr = rew.join('');
        if (fo && fr) blocks.push({ blockType: 'modify', source: fo, target: fr, charDiffs: ca.alignChars(fo, fr) });
        else if (fo) blocks.push({ blockType: 'delete', source: fo, target: null, charDiffs: fo.split('').map(c => ({ source: c, target: null, diffType: 'delete' })) });
        else if (fr) blocks.push({ blockType: 'insert', source: null, target: fr, charDiffs: fr.split('').map(c => ({ source: null, target: c, diffType: 'insert' })) });
    }
    const result = { blocks,
        summary() { const s = {}; for (const b of this.blocks) s[b.blockType] = (s[b.blockType] || 0) + 1; return s; },
        toDiffText() { const lines = []; for (const b of this.blocks) { if (b.blockType === 'equal') lines.push(`  ${b.source}`); else if (b.blockType === 'delete') lines.push(`- ${b.source}`); else if (b.blockType === 'insert') lines.push(`+ ${b.target}`); else if (b.blockType === 'modify') { lines.push(`- ${b.source}`); lines.push(`+ ${b.target}`); } } return lines.join('\n'); }
    };
    return result;
}

const charDiff = (t1, t2) => new CharLevelAligner().alignChars(t1, t2);

// ========== Export ==========
const sgl = { hybridSimilarity, hybridDetail, cacheClear, cacheInfo, Aligner, richDiff, charDiff, splitSents, countSents };

if (typeof module !== 'undefined' && module.exports) module.exports = sgl;
if (typeof window !== 'undefined') window.sgl = sgl;
