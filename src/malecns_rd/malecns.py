from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from .graph import ConnectomeGraph


EXCITATORY = {"acetylcholine", "ach"}
INHIBITORY = {"gaba", "glutamate", "histamine"}


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"none of {candidates} found; columns={list(df.columns)}")


def load_malecns_feather(
    annotations_path: str | Path,
    neurotransmitters_path: str | Path,
    weights_path: str | Path,
    *,
    traced_only: bool = True,
    min_synapses: int = 3,
    modulatory_fast_weight: float = 0.0,
) -> tuple[ConnectomeGraph, pd.DataFrame]:
    """Load MaleCNS v1.0 flat Feather tables into the recurrent-depth engine.

    The code intentionally discovers common column-name variants instead of
    assuming a third-party transformed schema. PyArrow is required by pandas
    to read Feather files (install with ``pip install -e '.[malecns]'``).

    Returns the graph and the filtered annotation table in graph index order.
    """
    ann = pd.read_feather(annotations_path)
    nt = pd.read_feather(neurotransmitters_path)
    edges = pd.read_feather(weights_path)

    ann_id = _find_column(ann, ("bodyId", "body", "body_id"))
    nt_id = _find_column(nt, ("bodyId", "body", "body_id"))
    pre_col = _find_column(edges, ("body_pre", "bodyPre", "pre", "source"))
    post_col = _find_column(edges, ("body_post", "bodyPost", "post", "target"))
    weight_col = _find_column(edges, ("weight", "syn_count", "count"))

    if traced_only and "status" in ann.columns:
        ann = ann.loc[ann["status"].astype(str).str.lower() == "traced"].copy()

    # Identify the aggregate transmitter label if present. If the official
    # table changes naming, probability columns are used as a fallback.
    label_col = None
    for candidate in ("consensus_nt", "predicted_nt", "nt", "neurotransmitter", "celltype_nt"):
        if candidate in nt.columns:
            label_col = candidate
            break

    if label_col is None:
        prob_cols = [c for c in nt.columns if c.lower() in {
            "acetylcholine", "ach", "gaba", "glutamate", "histamine",
            "dopamine", "serotonin", "octopamine"
        }]
        if not prob_cols:
            raise KeyError("could not infer neurotransmitter column(s)")
        probs = nt[prob_cols].to_numpy(dtype=np.float32)
        nt = nt.copy()
        nt["_predicted_nt"] = np.asarray(prob_cols, dtype=object)[np.argmax(probs, axis=1)]
        label_col = "_predicted_nt"

    nt_map = nt.set_index(nt_id)[label_col].astype(str).str.lower()
    ann = ann.loc[ann[ann_id].isin(nt_map.index)].copy()
    body_ids = ann[ann_id].astype(np.int64).to_numpy()
    index = pd.Series(np.arange(len(body_ids), dtype=np.int64), index=body_ids)

    edges = edges.loc[edges[weight_col].astype(float) >= min_synapses, [pre_col, post_col, weight_col]].copy()
    mask = edges[pre_col].isin(index.index) & edges[post_col].isin(index.index)
    edges = edges.loc[mask]

    src_body = edges[pre_col].astype(np.int64).to_numpy()
    dst_body = edges[post_col].astype(np.int64).to_numpy()
    syn = edges[weight_col].astype(np.float32).to_numpy()
    src = index.loc[src_body].to_numpy(dtype=np.int64)
    dst = index.loc[dst_body].to_numpy(dtype=np.int64)

    transmitters = nt_map.loc[src_body].to_numpy(dtype=object)
    sign = np.empty(len(transmitters), dtype=np.float32)
    for i, t in enumerate(transmitters):
        if t in EXCITATORY:
            sign[i] = 1.0
        elif t in INHIBITORY:
            sign[i] = -1.0
        else:
            sign[i] = float(modulatory_fast_weight)

    signed_weight = syn * sign
    keep = signed_weight != 0.0
    graph = ConnectomeGraph.from_edges(
        len(body_ids), src[keep], dst[keep], signed_weight[keep], body_ids=body_ids,
        normalize_incoming=True, max_incoming_abs=1.0,
    )
    return graph, ann.reset_index(drop=True)
