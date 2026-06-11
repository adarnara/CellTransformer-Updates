"""
CellTransformer inference on all_concat_reg.h5ad
=================================================
Outputs (saved to inference_output/):
  brain_set_{i}.h5ad         One AnnData per brain_set (i = 1..15)
      .X                     neighborhood embeddings  (n_cells × 384, float32)
      .obs                   cell_id, x_spatial, y_spatial, subclass_label,
                             spatial_cluster (KMeans label 0..N_CLUSTERS-1)
      .obsm['spatial']       array [[x1,y1], [x2,y2], ...]  (original µm coords)
      .uns                   run metadata (model config, gene count, etc.)

  all_brainsets.h5ad         Same structure, all cells concatenated (handy for
                             cross-section analyses; obs also carries 'brain_set')

  brain_set7_spatial.png     Quick scatter plot coloured by spatial_cluster

  run.log                    stdout/stderr of this process

Usage:
    .venv/bin/python -u run_inference.py
"""

import os
import pathlib
import warnings

# help with memory fragmentation on small GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import anndata as ad
import colorcet as cc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder

import celltransformer
from celltransformer.data import CenterMaskSampler, collate

# ── configuration ────────────────────────────────────────────────────────────
ADATA_PATH       = "all_concat_reg.h5ad"
WEIGHTS_PATH     = "model_weights.pth"
OUTPUT_DIR       = pathlib.Path("inference_output")

N_GENES          = 500          # model trained on 500-gene AIBS MERFISH panel
CELL_CARDINALITY = 384
PATCH_SIZE       = (17, 17)     # same as demo notebook
COORD_SCALE      = 10.0         # x_spatial / 10 → same unit scale as training

# 200 matches the training config cap; keeps attention matrices manageable on
# 8 GB GPUs (encoder mask ≤ ~3200×3200 at batch_size=16, max_cells=200)
MAX_CELLS        = 250
BATCH_SIZE       = 16
NUM_WORKERS      = 4
N_CLUSTERS       = 25           # KMeans clusters per brain_set
VIS_BRAIN_SET    = 7            # which brain_set to render as PNG

# run each brain_set on alternating GPUs to use both cards during inference
GPUS             = [0, 1] if torch.cuda.device_count() >= 2 else [0]
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def build_sampler_metadata(obs: pd.DataFrame, le: LabelEncoder) -> pd.DataFrame:
    """
    Return the minimal DataFrame that CenterMaskSampler needs, indexed to match
    the adata row order.  Uses brain_set as the tissue-section grouping so that
    neighbourhood search stays within the same brain.
    """
    meta = pd.DataFrame(
        {
            "cell_label":     obs.index.astype(str),
            "x":              obs["x_spatial"].values / COORD_SCALE,
            "y":              obs["y_spatial"].values / COORD_SCALE,
            "cell_type":      le.transform(obs["class_label"].astype(str)).astype(int),
            "section_label":  obs["brain_set"].astype(str).values,
        },
        index=np.arange(len(obs)),
    )
    return meta


def load_model(n_genes: int, device: str) -> celltransformer.model.CellTransformer:
    model = celltransformer.model.CellTransformer(
        encoder_depth=4,        decoder_depth=4,
        encoder_embedding_dim=384, decoder_embedding_dim=384,
        encoder_num_heads=8,    decoder_num_heads=8,
        attn_pool_heads=8,
        n_genes=n_genes,
        cell_cardinality=CELL_CARDINALITY,
        eps=1e-9,
        bias=True,
        zero_attn=True,
    )
    weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(weights)
    model = model.to(device).eval()
    return model


def run_inference_brainset(
    model, adata_bs, meta_bs, device: str, brain_set: int
) -> np.ndarray:
    """
    Run inference for a single brain_set. Returns float32 array (n_cells, 384).
    """
    sampler = CenterMaskSampler(
        metadata=meta_bs,
        adata=adata_bs,
        patch_size=PATCH_SIZE,
        cell_id_colname="cell_label",
        cell_type_colname="cell_type",
        tissue_section_colname="section_label",
        max_num_cells=MAX_CELLS,
    )
    loader = torch.utils.data.DataLoader(
        sampler,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate,
        prefetch_factor=2,
    )

    embeds = []
    bf16 = torch.amp.autocast(device, dtype=torch.bfloat16)
    with torch.inference_mode(), bf16:
        for batch in tqdm.tqdm(
            loader,
            desc=f"  brain_set {brain_set} [{device}]",
            unit="batch",
            dynamic_ncols=True,
        ):
            emb = model(batch)["neighborhood_repr"].detach().cpu().float().numpy()
            embeds.append(emb)
    return np.concatenate(embeds, axis=0)


def make_brainset_adata(
    embeddings: np.ndarray,
    obs_df: pd.DataFrame,
    brain_set: int,
    n_clusters: int,
) -> ad.AnnData:
    """
    Build an AnnData for one brain_set:
        .X              = embeddings (cells × 384)
        .obs            = per-cell metadata + KMeans cluster label
        .obsm['spatial']= [[x_spatial, y_spatial], ...]
        .uns            = run config
    """
    print(f"  brain_set {brain_set}: {len(embeddings)} cells — "
          f"clustering into {n_clusters} groups …", flush=True)

    km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, n_init=5, batch_size=4096
    )
    labels = km.fit_predict(embeddings)

    obs = pd.DataFrame(
        {
            "cell_id":        obs_df["cell_id"].values,
            "class_label": obs_df["class_label"].values,
            "brain_set":      brain_set,
            "spatial_cluster": labels.astype(int),
        },
        index=pd.Index(obs_df["cell_id"].astype(str), name="cell_id"),
    )

    adata_out = ad.AnnData(
        X=embeddings.astype(np.float32),
        obs=obs,
    )
    adata_out.obsm["spatial"] = np.column_stack(
        [obs_df["x_spatial"].values, obs_df["y_spatial"].values]
    ).astype(np.float32)
    adata_out.uns = {
        "brain_set":      brain_set,
        "n_clusters":     n_clusters,
        "model":          "CellTransformer",
        "embed_dim":      384,
        "n_genes_input":  N_GENES,
        "patch_size":     list(PATCH_SIZE),
        "weights_file":   WEIGHTS_PATH,
        "coord_scale":    COORD_SCALE,
    }
    return adata_out


def plot_brain_set(adata_bs: ad.AnnData, brain_set: int, out_path: pathlib.Path) -> None:
    xy      = adata_bs.obsm["spatial"]
    labels  = adata_bs.obs["spatial_cluster"].values
    n_clust = int(labels.max()) + 1

    colormap = sns.color_palette(cc.glasbey, n_colors=n_clust)
    hex_map  = {k: v for k, v in zip(range(n_clust), colormap.as_hex())}
    colors   = [hex_map[lbl] for lbl in labels]

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=1.2,
               linewidths=0, alpha=0.85, rasterized=True)
    ax.invert_yaxis()
    ax.set_title(
        f"CellTransformer neighbourhood embeddings — brain_set {brain_set}  "
        f"({n_clust} spatial clusters)",
        fontsize=13,
    )
    ax.set_xlabel("x_spatial (µm)")
    ax.set_ylabel("y_spatial (µm)")
    ax.axis("off")

    from matplotlib.patches import Patch
    legend_el = [Patch(facecolor=hex_map[i], label=f"Cluster {i}") for i in range(n_clust)]
    ax.legend(handles=legend_el, loc="upper right", fontsize=6,
              ncol=2, framealpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG saved → {out_path}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load adata ──────────────────────────────────────────────────────────
    print("Loading adata …", flush=True)
    adata = ad.read_h5ad(ADATA_PATH)
    print(f"  shape: {adata.shape}  ({adata.shape[0]:,} cells × {adata.shape[1]:,} genes)",
          flush=True)

    # Replace X with log2(raw_count + 1) to match AIBS training normalization.
    # The X matrix in this dataset is z-scored (scale_data); raw_count + log2p
    # is what the model was trained on (AIBS: raw MERFISH counts → log2(x+1)).
    print("  Applying log2(raw_count + 1) normalization to match training …", flush=True)
    adata.X = np.log2(adata.layers["raw_count"] + 1)

    # Select the 500 AIBS training genes in the exact training order.
    # Gene order matters — expression_projection weights are position-dependent.
    with open("aibs_500_genes.txt") as fh:
        training_genes = [l.strip() for l in fh if l.strip()]
    user_gene_list = list(adata.var["gene"].values)
    user_gene_idx  = {g: i for i, g in enumerate(user_gene_list)}
    missing = [g for g in training_genes if g not in user_gene_idx]
    if missing:
        raise ValueError(f"Training genes missing from data: {missing}")
    col_order  = [user_gene_idx[g] for g in training_genes]
    adata_filt = adata[:, col_order].copy()
    print(f"  Selected {len(training_genes)} genes in AIBS training order  "
          f"(first 5: {training_genes[:5]})", flush=True)

    # 2. Metadata & label encoder ────────────────────────────────────────────
    print("Encoding cell types …", flush=True)
    all_subclasses = sorted(adata.obs["class_label"].astype(str).unique())
    le = LabelEncoder()
    le.fit(all_subclasses)
    print(f"  {len(all_subclasses)} class_label types  "
          f"(cardinality cap={CELL_CARDINALITY})", flush=True)

    meta_df = build_sampler_metadata(adata.obs, le)

    # 3. Load one model per GPU (73 MB each — negligible) ────────────────────
    print(f"Loading model onto {len(GPUS)} GPU(s): {GPUS} …", flush=True)
    models = {}
    for gpu_id in GPUS:
        device = f"cuda:{gpu_id}"
        m = load_model(N_GENES, device)
        m.put_device = device   # forward() uses this to move tensors
        models[gpu_id] = m
    n_params = sum(p.numel() for p in models[GPUS[0]].parameters())
    print(f"  {n_params:,} params per model  "
          f"(bs={BATCH_SIZE}, max_cells={MAX_CELLS})", flush=True)

    # 4. Per-brain_set inference, saving h5ad immediately ────────────────────
    brain_sets  = sorted(int(bs) for bs in adata.obs["brain_set"].unique())
    per_bs_adatas = []

    print(f"\nProcessing {len(brain_sets)} brain_sets …", flush=True)

    for idx, bs in enumerate(brain_sets):
        gpu_id = GPUS[idx % len(GPUS)]
        device = f"cuda:{gpu_id}"
        model  = models[gpu_id]

        bs_mask = adata.obs["brain_set"].astype(int).values == bs
        adata_bs = adata_filt[bs_mask]
        obs_bs   = adata.obs[bs_mask]
        meta_bs  = build_sampler_metadata(obs_bs, le)
        # align adata rows to meta_bs order
        adata_bs = adata_bs[meta_bs["cell_label"].values]

        print(f"\nbrain_set {bs}  ({bs_mask.sum():,} cells)  → GPU {gpu_id}", flush=True)
        embeddings_bs = run_inference_brainset(model, adata_bs, meta_bs, device, bs)
        print(f"  embeddings: {embeddings_bs.shape}", flush=True)
        torch.cuda.empty_cache()

        obs_flat_bs = pd.DataFrame(
            {
                "cell_id":        obs_bs.index.astype(str),
                "class_label": obs_bs["class_label"].astype(str).values,
                "brain_set":      bs,
                "x_spatial":      obs_bs["x_spatial"].values,
                "y_spatial":      obs_bs["y_spatial"].values,
            }
        ).reset_index(drop=True)

        adata_bs_out = make_brainset_adata(embeddings_bs, obs_flat_bs, bs, N_CLUSTERS)

        out_path = OUTPUT_DIR / f"brain_set_{bs}.h5ad"
        adata_bs_out.write_h5ad(out_path, compression="gzip")
        print(f"  Saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)",
              flush=True)

        per_bs_adatas.append(adata_bs_out)

        if bs == VIS_BRAIN_SET:
            plot_brain_set(
                adata_bs_out, bs,
                out_path=OUTPUT_DIR / f"brain_set{VIS_BRAIN_SET}_spatial.png",
            )

    # clean up models
    for m in models.values():
        del m
    torch.cuda.empty_cache()

    # 5. Combined h5ad ───────────────────────────────────────────────────────
    print("\nConcatenating all brain_sets → all_brainsets.h5ad …", flush=True)
    combined = ad.concat(
        per_bs_adatas, join="outer",
        label="brain_set", keys=[str(bs) for bs in brain_sets],
    )
    combined.obsm["spatial"] = np.concatenate(
        [a.obsm["spatial"] for a in per_bs_adatas], axis=0
    )
    combined.uns = {
        "description":              "CellTransformer neighbourhood embeddings, all brain_sets",
        "n_clusters_per_brainset":  N_CLUSTERS,
        "model":                    "CellTransformer",
        "embed_dim":                384,
        "n_genes_input":            N_GENES,
        "patch_size":               list(PATCH_SIZE),
        "max_cells_per_neighbourhood": MAX_CELLS,
        "weights_file":             WEIGHTS_PATH,
    }
    combined_path = OUTPUT_DIR / "all_brainsets.h5ad"
    combined.write_h5ad(combined_path, compression="gzip")
    print(f"  Saved → {combined_path}  ({combined_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
