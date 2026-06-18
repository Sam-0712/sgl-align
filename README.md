# Structure-Gated Lexical Similarity

A text similarity and alignment toolkit for Chinese text comparison, originally designed for AI-assisted essay correction. The library operates entirely in pure Python with NumPy acceleration and provides four major algorithmic modules: hybrid similarity scoring, global sequence alignment, character-level diff, and word-level diff.

# Hybrid Similarity

The similarity function $\mathcal{F}(x, y)$ is the computational core of the entire toolkit. Rather than relying on a single metric, it synthesises five complementary signals and dynamically weights them according to the structural properties of the two input strings. The goal is to distinguish genuine paraphrases and revisions from superficially overlapping but semantically unrelated text—a problem that naive character overlap or edit distance cannot solve.

Before any expensive computation, a fast gate checks whether the two strings share enough characters to warrant further analysis:

$$
\text{ov}(x, y) = \frac{|\text{set}(x) \cap \text{set}(y)|}{\min(|\text{set}(x)|, |\text{set}(y)|)}
$$

If $\text{ov}(x, y)$ falls below a configurable threshold, the function returns zero immediately. This eliminates the vast majority of unrelated sentence pairs in $O(|x| + |y|)$ time without any matrix allocation.

The primary lexical signal is a weighted Dice coefficient computed over character n-grams of orders $1$ through $K$, where the maximum order $K$ is determined adaptively:

$$
K = \max(1, \lfloor \log_2(\min(|x|, |y|)) \rfloor - 1)
$$

For each order $k \in \{1, \dots, K\}$, the Dice coefficient is

$$
D_k(x, y) = \frac{2 \cdot |G_k(x) \cap G_k(y)|}{|G_k(x)| + |G_k(y)|}
$$

where $G_k(\cdot)$ is the multiset of character k-grams. The intersection uses minimum multiplicity—if a gram appears 3 times in $x$ and 2 times in $y$, it contributes 2 to the overlap count.

Higher-order n-grams carry more structural information but are sparser and more brittle. The weights follow an inverse Zipfian decay:

$$
w_k = \frac{1}{k \cdot \ln(k + 1)}, \qquad
\bar{w}_k = \frac{w_k}{\sum_{i = 1}^{K} w_i}
$$

This gives 1-grams roughly 3 times the weight of 4-grams, ensuring that character-level tolerance (typos) dominates while higher-order constraints prevent drift. The weighted Dice sum is:

$$
S_{\text{dice}} = \sum_{k = 1}^{K} \bar{w}_k \cdot D_k(x, y)
$$

The `LCS` ratio measures global sequential similarity independent of n-gram locality:

$$
\text{LCS}(x, y) = \frac{2 \cdot |\text{LCS}(x, y)|}{|x| + |y|}
$$

The implementation uses the standard $O(|x| \cdot |y|)$ dynamic programming algorithm with a space optimisation that keeps only two rows at a time, reducing memory to $O(\min(|x|, |y|))$.

`ROC` detects whether common characters preserve their relative ordering. Given two strings $x$ and $y$, for each character $c$ we record all its occurrence positions in $y$, then scan $x$ left-to-right to produce a sequence of $y$-indices. The `ROC` score is

$$
\text{ROC}(x, y) = \frac{\text{LIS}(\text{indices})}{|\text{indices}|}
$$

where `LIS` is the length of the Longest Increasing Subsequence, computed in $O(n \log n)$ via patience sorting. A `ROC` near 1.0 means shared characters appear in the same order; a low `ROC` signals reordering or semantic disruption. Matching n-gram positions reveal *where* edits are concentrated. For each matched n-gram (from the $k=1$ Dice pass), its starting position is recorded. The dispersion of these positions is measured by their standard deviation normalised against half the text length:

$$
\delta(\text{positions}, L) = \min\left(\frac{\sigma(\text{positions})}{L / 2}, 1\right)
$$

where $\sigma$ is the population standard deviation. Localized edits (e.g. a single character replacement) produce tightly clustered matches with low $\delta$. Structural reorganisation scatters matches across the string, yielding high $\delta$. The dispersion *consistency* between $x$ and $y$ is $\Delta = 1 - |\delta_x - \delta_y|$, a pair where both strings have similarly localised or similarly scattered edits receives $\Delta \approx 1$.

Rather than applying a fixed interpolation weight between Dice and `LCS`, the algorithm derives a *structural ratio* $r_s$ from the texts' intrinsic properties:

$$
r_s = \frac{\ln(\min(|x|, |y|) + 1)}{\ln(\max(|x|, |y|) + |\Sigma| + 1)} \cdot \frac{C(x) + C(y)}{2}
$$

where $C(s) = |\text{set}(s)| / |s|$ is the character complexity (ratio of unique to total characters) and $|\Sigma| = |\text{set}(x) \cup \text{set}(y)|$ is the joint alphabet size.

- **Small $r_s$**: short texts with low complexity → Dice dominates (lexical similarity is sufficient).
- **Large $r_s$**: long texts with rich vocabulary → `LCS` gains weight (sequential structure matters).

The raw match count from 1-gram overlap is calibrated against a random baseline. Given $n_{\min} = \min(|x|, |y|)$ and the alphabet size $|\Sigma|$:

$$
c_f = \frac{m_1 / n_{\min} - \varepsilon_r}{1 - \varepsilon_r}, \qquad
\varepsilon_r = \frac{1}{|\Sigma| + 1}
$$

where $m_1$ is the number of 1-gram matches and $\varepsilon_r$ is the expected match rate under random draws from the joint alphabet. A negative $c_f$ is clamped to zero. This factor is then weighted by the mean complexity to produce the dispersion reward:

$$
w_{\text{disp}} = \max(0, c_f) \cdot \frac{C(x) + C(y)}{2}
$$

Two final modifiers adjust the score for plausibility and size imbalance:

$$
\ell_r = \text{ROC}^{\; 1 / (C_{\text{mean}} + \varepsilon)},
\qquad
\ell_p = \left(\frac{|x|_{\min}}{|x|_{\max}}\right)^{\frac{\ln(|x|_{\min} + 1)}{\ln(|x|_{\max} + 1)}}
$$

The logic reward $\ell_r$ penalises strings whose shared characters are disordered. The exponent $1 / C_{\text{mean}}$ modulates this penalty: complex texts (many unique characters) have weaker ordering constraints. The length penalty $\ell_p$ is a sublinear damping factor. When one string is much shorter, the exponent shrinks the penalty (e.g., a 5:20 ratio yields $\ell_p \approx 0.56$ rather than 0.25), acknowledging that short fragments can legitimately match longer ones.

The base score blends Dice and `LCS` via the structural ratio:

$$
S_{\text{base}} = S_{\text{dice}} \cdot (1 - r_s) + \text{LCS} \cdot r_s
$$

The complete hybrid similarity is

$$
\mathcal{F}(x, y) = \ell_p \cdot \ell_r \cdot \frac{S_{\text{base}} + \Delta \cdot w_{\text{disp}}}{1 + w_{\text{disp}}}
$$

This formulation ensures that dispersion consistency ($\Delta$) can boost the score only when the confidence factor $c_f$ is high (i.e., when there is genuine character-level overlap beyond random chance). The denominator $1 + w_{\text{disp}}$ normalizes the reward so it can increase but not dominate the base score.

# Needleman-Wunsch Global Alignment

The `Aligner` class implements Needleman-Wunsch global sequence alignment with affine gap penalties, designed to operate on sentence sequences using an arbitrary element-wise similarity function. It solves the problem: given two ordered sequences $S = \{s_1, \dots, s_n\}$ and $T = \{t_1, \dots, t_m\}$, find the optimal alignment that maximizes the total score while penalising insertions and deletions. A linear gap penalty charges the same cost for every inserted or deleted element, which fragments contiguous structural edits into many small gaps. Affine penalties distinguish two costs:

- **Gap opening**: the cost of *starting* a gap.
- **Gap extension**: the cost of *continuing* an existing gap.

  A deletion of $k$ consecutive elements costs $g_o + k \cdot g_e$ rather than $k \cdot g_o$. Since $|g_e| \ll |g_o|$, this strongly prefers contiguous gaps over fragmented ones, reflecting the linguistic intuition that a deleted sentence is a single structural edit. This requires splitting the DP into three interleaved matrices:

| Matrix    | Semantics                                            |
| --------- | ---------------------------------------------------- |
| $M[i, j]$ | Best score ending with $s_i$ *aligned* to $t_j$      |
| $X[i, j]$ | Best score ending with $s_i$ *deleted* (gap in $T$)  |
| $Y[i, j]$ | Best score ending with $t_j$ *inserted* (gap in $S$) |

Let $\sigma(s_i, t_j)$ be the similarity function. A mismatch threshold $\tau$ filters unreliable matches: pairs with $\sigma < \tau$ are forced into gap states:

$$
\text{score}(i, j) = \begin{cases}
\sigma(s_i, t_j) & \text{if } \sigma(s_i, t_j) \geq \tau \\
g_o \times 1.5 & \text{otherwise}
\end{cases}
$$

The three recurrences are

$$
\begin{aligned}
M [i, j] &= \max\{M [i-1, j-1], X [i-1, j-1], Y [i-1, j-1]\} + \text{score}(i, j) \\
X [i, j] &= \max\{M [i-1, j] + g_o,\; X [i-1, j] + g_e\} \\
Y [i, j] &= \max\{M [i, j-1] + g_o,\; Y [i, j-1] + g_e\}
\end{aligned}
$$

with base cases $M[0, 0] = 0$, $X[i, 0] = g_o + (i-1) \cdot g_e$, $Y[0, j] = g_o + (j-1) \cdot g_e$. The combined value at each cell is $C[i, j] = \max\{M[i, j], X[i, j], Y[i, j]\}$ and the final alignment score is $C[n, m]$. The $Y$ recurrence is not fully vectorisable in NumPy because each column depends on the previous column *and* the $M$ value at the same column. The implementation vectorises $M$ and $X$ updates via NumPy operations, then fills $Y$ with a fast inner loop in pure Python:

```python
yrow[j] = max(M[i, j-1] + g_o, yrow[j-1] + g_e)
```

This hybrid approach preserves near-C performance for the dominant $M$ and $X$ computations while correctly handling $Y$'s column dependency.

The full-matrix approach requires $O(nm)$ memory, which is prohibitive for long texts. The default aligner uses the Myers-Miller divide-and-conquer algorithm that reduces space to $O(m)$ while preserving the exact optimal alignment. A single row (or column) of DP is maintained using one-dimensional vectors $\mathbf{M}, \mathbf{X}, \mathbf{Y}$, each of length $m+1$. After processing all $n$ rows of $S$, these vectors hold the scores at the final alignment frontier. Given a subproblem $S[i_1:i_2]$ vs $T[j_1:j_2]$:

1. Split $S$ at the midpoint $i_{\text{mid}} = i_1 + \lfloor (i_2 - i_1) / 2 \rfloor$.
2. Compute forward vectors for $S[i_1:i_{\text{mid}}]$ vs $T[j_1:j_2]$.
3. Compute backward vectors for $\text{reverse}(S[i_{\text{mid}}:i_2])$ vs $\text{reverse}(T[j_1:j_2])$.
4. Find the column $j^\ast$ that maximizes $\text{fwd}[j] + \text{rev}[j]$ and the optimal state $s^\ast \in \{M, X, Y\}$.
5. Recurse on the left half $[i_1, i_{\text{mid}}] \times [j_1, j^\ast]$ and right half $[i_{\text{mid}}, i_2] \times [j^\ast, j_2]$ with the appropriate entering/exiting states.

When the subproblem becomes trivially small ($n = 1$ or $m = 1$), a full $O(nm)$ exact submatrix DP resolves the remaining cells with the correct state constraints.

For visualization and diagnostics, `alignfb()` computes the forward-backward matrix:

$$
\text{FB}[i, j] = C_{\text{fwd}}[i, j] + C_{\text{rev}}[i, j] - C_{\text{total}}
$$

Cells with $\text{FB}[i, j] \approx 0$ lie on the optimal alignment path; cells with negative values are off-path. This is computed by running the full forward DP and a separate backward DP from $(n, m)$, then combining the results.

When paragraphs are restructured, a global sequence alignment may produce meaningless results because the order constraint is too rigid. `reorderalign()` relaxes this by detecting and flagging shuffled segments:

1. **Greedy matching**: Build a similarity matrix $n \times m$, collect all pairs with $\sigma \geq \text{threshold}$, and greedily select the highest-scoring one-to-one matches.
2. **Shuffle detection**: Sort matches by source index. If the corresponding target indices are not monotonically increasing, detect contiguous runs of out-of-order indices as shuffle groups.
3. **Alignment construction**: Walk through source indices. Matched indices produce aligned pairs (flagged `is_shuffled=True` if in a shuffle group). Unmatched indices become deletions or insertions.

The output preserves the original source ordering (for readability) while annotating which segments were reordered.

# Character-Level Diff

The character-level module operates on the *output* of sentence-level alignment. It takes aligned sentence pairs and produces fine-grained character-by-character diffs, then merges them into structurally meaningful blocks.Individual characters are aligned using the same Needleman-Wunsch algorithm, but with a binary similarity function:

$$
\sigma(c_1, c_2) = \begin{cases}
1.0 & \text{if } c_1 = c_2 \\
0.0 & \text{otherwise}
\end{cases}
$$

A naive character alignment would interleave delete and insert operations: transforming "ABC" to "XYZ" would produce `[-A][+X][-B][+Y][-C][+Z]`. For human readability, consecutive non-equal character pairs are aggregated:

1. Scan the aligned pairs. Each run of consecutive `equal` characters forms a boundary point.

2. Between boundaries, collect all source characters (deletions) and target characters (insertions).

3. Emit all deletions first, then all insertions, producing `[-ABC][+XYZ]` as two blocks.

This merging significantly improves visual diff clarity. The internal char-level alignment is preserved in the `CharDiff` list of each block for detailed inspection. The `richdiff()` function consumes a list of `AlignedPair` objects (from sentence-level alignment) and produces a `RichDiffResult`:

- **Anchor pairs**: sentence pairs with similarity $\geq 0.80$ (or identical text) are treated as anchors. Identical pairs become `equal` blocks; similar but non-identical pairs become `modify` blocks with char-level diffs.

- **Gap runs**: consecutive insertion/deletion pairs between anchors are collapsed. If both deletions and insertions appear, they form a `modify` block; pure deletions or pure insertions form their respective blocks.

This two-level structure (sentence blocks → char diffs) mirrors the hierarchical nature of text revision: an editor rewrites a few sentences and deletes others, and within each rewritten sentence, only a few characters change.

# Word-Level Alignment 

Word-level alignment bridges the gap between character-level precision and sentence-level structure. It segments text into linguistic tokens (words and punctuation), aligns them via NW using the hybrid similarity function, then produces diff blocks with per-word granularity and optional character-level detail.

The default segmenter uses [jieba](https://github.com/fxsjy/jieba) for Chinese word segmentation with an English preservation rule:

1. Run jieba's `lcut()` to segment Chinese text into words.

2. Consecutive ASCII alphanumeric tokens are concatenated into a single English word (e.g., `hello` and `world` remain `["hello", "world"]` rather than being split by jieba's character-level handling).

3. Punctuation—both CJK punctuation (，。！？) and full-width forms—is split into standalone tokens via a regex post-processing pass.

Users can supply a custom segmenter `str -> list[str]` to replace the default. The segmented word lists are aligned with the full-matrix Needleman-Wunsch algorithm (linear-space is disabled here because word counts are typically small; the full matrix is needed for accurate backtracing). The similarity function is `hybrid_similarity`—the same engine used for sentence-level scoring, but operating on individual words.

Aligned word pairs are post-processed into a flat list of `WordDiffBlock` objects. The merging logic handles five cases:

| Condition                                              | Block Type            | Example                |
| ------------------------------------------------------ | --------------------- | ---------------------- |
| Identical words                                        | `equal`               | `的` ↔ `的`            |
| Matched but different                                  | `modify` (single)     | `夏日` ↔ `夏天`        |
| Consecutive deletes only (no equals interleaving)      | `delete` (individual) | `那份` → ∅, `在` → ∅   |
| Consecutive inserts only (no equals interleaving)      | `insert` (individual) | ∅ → `那一刻`, ∅ → `在` |
| Alternating deletes + inserts (no equals between them) | `modify` (merged)     | `在那一刻被` ↔ `在`    |

The merged modify case handles scenarios where the NW aligner produces alternating insert-delete pairs for what is semantically a single substitution. The source words and target words from all pairs in the run are concatenated, and a character-level diff is computed for the combined strings. Each `WordDiffBlock` carries its constituent word lists and (for modify blocks) a character-level diff that allows drilling down from "these words changed" to "these characters changed within those words".

# Change Magnitude Index (CMI)

The `AlignmentResult` from NW gives a pathway—which sentences align, which are deleted—but does not directly answer *how much* the text was changed. CMI translates the alignment into a scalar in $[0, 1]$ where $0$ means "identical texts" and $1$ means "maximally different". The computation operates at three granularities: document (whole text), paragraph (per logical segment), and sentence(per aligned pair). Paragraph-level CMI requires a paragraph → sentence index mapping, built by `splitparas()` and `splitsents()`.

Each aligned pair contributes an individual edit cost, normalised to $[0, 1]$:

- **Match / mismatch pair** with similarity $\sigma$:
  
  $$
  c = 1 - \sigma
  $$

  Identical text yields $c = 0$; completely dissimilar text yields $c = 1$.

- **Gap pair** (deletion or insertion): the affine gap penalty structure of the aligner is preserved. For a gap run of length $k$, the raw cost is

  $$
  C_{\text{gap}} = |g_o| + (k - 1) \cdot |g_e|
  $$

  where $g_o$ and $g_e$ are the aligner's gap-open and gap-extend penalties. The per-element cost is $C_{\text{gap}} / k$, then normalized by the maximum possible per-element gap cost $|g_o| + |g_e|$. A single-element gap costs $|g_o| / (|g_o| + |g_e|) \approx 0.88$ under default parameters; longer gap runs have lower per-element cost, reflecting the linguistic intuition that contiguous structural edits are a single operation rather than many independent ones.

**Document CMI** is the arithmetic mean of all per-pair costs:
$$
\text{CMI}_{\text{doc}} = \frac{1}{N} \sum_{i = 1}^{N} c_i
$$

where $N$ is the number of aligned pairs (equal to $\max(|S|, |T|)$ due to global alignment). Because the hybrid similarity function is symmetric and NW finds the same optimal score for both directions, $\text{CMI}_{\text{doc}}$ is mathematically identical for $A \rightarrow B$ and $B \rightarrow A$. **Paragraph CMI** aggregates per-pair costs by paragraph membership. Each pair's cost is attributed to the paragraph of its source element; the paragraph CMI is the arithmetic mean of costs within that paragraph. **Sentence CMI** is simply the individual per-pair cost $c_i$, directly readable as "how much was this specific sentence changed".

Paragraph-level CMI can differ between the two alignment directions when a paragraph contains gap pairs (deletions or insertions). The `merged_cmi()` function reconciles these two perspectives via a simple arithmetic mean:

$$
\text{CMI}_{\text{merged}} = \frac{\text{CMI}_{A \rightarrow B} + \text{CMI}_{B \rightarrow A}}{2}
$$

For paragraphs without gaps, the two directions are identical and the merge is a no-op. For gap-containing paragraphs, the average provides a balanced view without favoring either the source or target perspective.
