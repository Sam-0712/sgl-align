"""core.sglcmi -- Change Magnitude Index (CMI).

Computes a normalised edit-cost metric from NW alignment results at three
granularities: document, paragraph, and sentence.

CMI ∈ [0, 1].  0 = identical, 1 = maximum possible edit distance.

Because the NW global-alignment score is left-right symmetric for any
symmetric similarity function, document-level CMI is always identical
between A→B and B→A.  Paragraph-level CMI differs only when a paragraph
contains a gap (deletion / insertion) — and even then the difference is
small.  For those edge cases the module offers ``merged_cmi()`` which
averages the two directions with a simple arithmetic mean.
"""

from __future__ import annotations

from core._types import AlignmentResult, CMIResult

# ── internal helpers ────────────────────────────────────────────────────


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


# ── public API ──────────────────────────────────────────────────────────


def compute_cmi(
    alignment: AlignmentResult,
    *,
    source_sent_to_para: list[int] | None = None,
    target_sent_to_para: list[int] | None = None,
) -> CMIResult:
    """Compute single-direction CMI from an alignment result.

    Parameters
    ----------
    alignment : AlignmentResult
        Output from ``Aligner.align()`` or ``Aligner.alignfb()``.
    source_sent_to_para : list[int] | None
        For each source sentence, the index (0-based) of its paragraph.
        If provided, paragraph-level CMI is computed.
    target_sent_to_para : list[int] | None
        Same structure for the target side (fallback for paragraph
        aggregation when *source_sent_to_para* is absent).

    Returns
    -------
    CMIResult
    """
    go = abs(alignment.gap_open)
    ge = abs(alignment.gap_extend)

    pairs = alignment.pairs
    n = len(pairs)

    # ── 1. per-pair costs ────────────────────────────────────────────
    costs: list[float] = []
    i = 0
    total_edit = 0.0
    total_max = 0.0
    match_sum = 0.0
    gap_sum = 0.0
    n_match = 0
    n_gap = 0
    cost_src_para: list[int | None] = []   # source paragraph per cost entry
    cost_tgt_para: list[int | None] = []   # target paragraph per cost entry
    src_pos = 0
    tgt_pos = 0

    while i < n:
        pair = pairs[i]

        if pair.is_gap:
            # ── gap run ──
            j = i
            while j < n and pairs[j].is_gap:
                j += 1
            run_len = j - i

            # Affine gap cost: gap_open + (k-1)*gap_extend
            raw = go + (run_len - 1) * ge
            per_elem = raw / run_len
            norm = per_elem / (go + ge) if (go + ge) > 0 else 1.0

            for k in range(run_len):
                gpair = pairs[i + k]
                costs.append(norm)
                total_edit += norm
                total_max += 1.0
                gap_sum += norm
                n_gap += 1
                # paragraph tracking
                if gpair.source is not None and source_sent_to_para is not None and src_pos < len(source_sent_to_para):
                    cost_src_para.append(source_sent_to_para[src_pos])
                    src_pos += 1
                else:
                    cost_src_para.append(None)
                if gpair.target is not None and target_sent_to_para is not None and tgt_pos < len(target_sent_to_para):
                    cost_tgt_para.append(target_sent_to_para[tgt_pos])
                    tgt_pos += 1
                else:
                    cost_tgt_para.append(None)

            i = j
        else:
            # ── match / mismatch ──
            cost = 1.0 - pair.similarity
            costs.append(cost)
            total_edit += cost
            total_max += 1.0
            match_sum += cost
            n_match += 1
            # paragraph tracking
            if pair.source is not None and source_sent_to_para is not None and src_pos < len(source_sent_to_para):
                cost_src_para.append(source_sent_to_para[src_pos])
                src_pos += 1
            else:
                cost_src_para.append(None)
            if pair.target is not None and target_sent_to_para is not None and tgt_pos < len(target_sent_to_para):
                cost_tgt_para.append(target_sent_to_para[tgt_pos])
                tgt_pos += 1
            else:
                cost_tgt_para.append(None)
            i += 1

    # ── 2. document CMI ──────────────────────────────────────────────
    cmi_doc = _safe_div(total_edit, total_max)

    # ── 3. paragraph CMI ─────────────────────────────────────────────
    cmi_paras: list[float] = []
    n_paras = 0

    if source_sent_to_para is not None:
        # Aggregate on source side
        n_paras = max(source_sent_to_para) + 1 if source_sent_to_para else 0
        para_costs: dict[int, tuple[float, int]] = {}
        for idx, para_idx in enumerate(cost_src_para):
            if para_idx is None:
                continue
            if para_idx not in para_costs:
                para_costs[para_idx] = (0.0, 0)
            c, cnt = para_costs[para_idx]
            para_costs[para_idx] = (c + costs[idx], cnt + 1)
        for pi in sorted(para_costs):
            c, cnt = para_costs[pi]
            cmi_paras.append(round(_safe_div(c, cnt), 4))
    elif target_sent_to_para is not None:
        # Fallback: aggregate on target side
        n_paras = max(target_sent_to_para) + 1 if target_sent_to_para else 0
        para_costs = {}
        for idx, para_idx in enumerate(cost_tgt_para):
            if para_idx is None:
                continue
            if para_idx not in para_costs:
                para_costs[para_idx] = (0.0, 0)
            c, cnt = para_costs[para_idx]
            para_costs[para_idx] = (c + costs[idx], cnt + 1)
        for pi in sorted(para_costs):
            c, cnt = para_costs[pi]
            cmi_paras.append(round(_safe_div(c, cnt), 4))

    return CMIResult(
        cmi_document=round(cmi_doc, 4),
        cmi_paragraphs=cmi_paras,
        n_paragraphs=n_paras,
        cmi_sentences=[round(c, 4) for c in costs],
        n_pairs=n,
        direction="a_to_b",
        total_edit_cost=round(total_edit, 4),
        total_max_cost=round(total_max, 4),
        match_cost_sum=round(match_sum, 4),
        gap_cost_sum=round(gap_sum, 4),
        n_match_pairs=n_match,
        n_gap_pairs=n_gap,
        gap_open=alignment.gap_open,
        gap_extend=alignment.gap_extend,
    )


def merged_cmi(
    alignment_ab: AlignmentResult,
    alignment_ba: AlignmentResult,
    *,
    source_sent_to_para: list[int] | None = None,
    target_sent_to_para: list[int] | None = None,
) -> CMIResult:
    """Arithmetic mean of A→B and B→A CMI.

    Document-level CMI is mathematically identical in both directions;
    paragraph-level CMI differs only for paragraphs containing gaps.
    The arithmetic mean provides a simple, balanced view for those
    edge cases.

    Parameters
    ----------
    alignment_ab / alignment_ba : AlignmentResult
    source_sent_to_para / target_sent_to_para : same as ``compute_cmi``.

    Returns
    -------
    CMIResult
        ``direction="merged"``.
    """
    ab = compute_cmi(alignment_ab,
                     source_sent_to_para=source_sent_to_para,
                     target_sent_to_para=target_sent_to_para)
    ba = compute_cmi(alignment_ba,
                     source_sent_to_para=target_sent_to_para,   # swapped
                     target_sent_to_para=source_sent_to_para)

    doc = round((ab.cmi_document + ba.cmi_document) / 2.0, 4)

    n_para_ab = len(ab.cmi_paragraphs)
    n_para_ba = len(ba.cmi_paragraphs)
    merged_paras: list[float] = []
    if n_para_ab > 0 and n_para_ba > 0:
        for i in range(min(n_para_ab, n_para_ba)):
            merged_paras.append(round((ab.cmi_paragraphs[i] + ba.cmi_paragraphs[i]) / 2.0, 4))
    elif n_para_ab > 0:
        merged_paras = ab.cmi_paragraphs
    else:
        merged_paras = ba.cmi_paragraphs

    merged_sents: list[float] = []
    for i in range(min(len(ab.cmi_sentences), len(ba.cmi_sentences))):
        merged_sents.append(round((ab.cmi_sentences[i] + ba.cmi_sentences[i]) / 2.0, 4))

    return CMIResult(
        cmi_document=doc,
        cmi_paragraphs=merged_paras,
        n_paragraphs=len(merged_paras),
        cmi_sentences=merged_sents,
        n_pairs=len(merged_sents),
        direction="merged",
        total_edit_cost=round((ab.total_edit_cost + ba.total_edit_cost) / 2.0, 4),
        total_max_cost=round((ab.total_max_cost + ba.total_max_cost) / 2.0, 4),
        match_cost_sum=round((ab.match_cost_sum + ba.match_cost_sum) / 2.0, 4),
        gap_cost_sum=round((ab.gap_cost_sum + ba.gap_cost_sum) / 2.0, 4),
        n_match_pairs=min(ab.n_match_pairs, ba.n_match_pairs),
        n_gap_pairs=min(ab.n_gap_pairs, ba.n_gap_pairs),
        gap_open=alignment_ab.gap_open,
        gap_extend=alignment_ab.gap_extend,
    )


# ── convenience: paragraph-aware sentence splitting ─────────────────────


def build_para_sent_map(text: str) -> tuple[list[str], list[int]]:
    """Split *text* into sentences with paragraph indices.

    Uses ``splitparas()`` + ``splitsents()`` internally.

    Returns
    -------
    (all_sentences, sent_to_para_index)
        ``sent_to_para_index[i]`` is the 0-based paragraph index for
        ``all_sentences[i]``.
    """
    from core.sgltext import splitparas, splitsents
    paras = splitparas(text)
    all_sents: list[str] = []
    indices: list[int] = []
    for pi, para in enumerate(paras):
        sents = splitsents(para)
        all_sents.extend(sents)
        indices.extend([pi] * len(sents))
    return all_sents, indices
