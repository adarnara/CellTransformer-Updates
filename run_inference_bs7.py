"""
Re-run inference for brain_set 7 only with MAX_CELLS=250.
Overwrites inference_output/brain_set_7.h5ad.
"""

import os
import pathlib

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import anndata as ad
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder

import celltransformer
from celltransformer.data import CenterMaskSampler, collate

ADATA_PATH       = "all_concat_reg.h5ad"
WEIGHTS_PATH     = "model_weights.pth"
OUTPUT_DIR       = pathlib.Path("inference_output")

N_GENES          = 500
CELL_CARDINALITY = 384
PATCH_SIZE       = (17, 17)
COORD_SCALE      = 10.0
MAX_CELLS        = 250
BATCH_SIZE       = 16
NUM_WORKERS      = 4
N_CLUSTERS       = 25
TARGET_BS        = 7
DEVICE           = "cuda:0"

OUTPUT_DIR.mkdir(exist_ok=True)


def build_sampler_metadata(obs, le):
    return pd.DataFrame(
        {
            "cell_label":    obs.index.astype(str),
            "x":             obs["x_spatial"].values / COORD_SCALE,
            "y":             obs["y_spatial"].values / COORD_SCALE,
            "cell_type":     le.transform(obs["class_label"].astype(str)).astype(int),
            "section_label": obs["brain_set"].astype(str).values,
        },
        index=np.arange(len(obs)),
    )


def load_model(n_genes, device):
    model = celltransformer.model.CellTransformer(
        encoder_depth=4, decoder_depth=4,
        encoder_embedding_dim=384, decoder_embedding_dim=384,
        encoder_num_heads=8, decoder_num_heads=8,
        attn_pool_heads=8,
        n_genes=n_genes,
        cell_cardinality=CELL_CARDINALITY,
        eps=1e-9, bias=True, zero_attn=True,
    )
    weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(weights)
    return model.to(device).eval()


def main():
    print("Loading adata …", flush=True)
    adata = ad.read_h5ad(ADATA_PATH)

    # Replace X with log2(raw_count + 1) to match AIBS training normalization.
    print("Applying log2(raw_count + 1) normalization …", flush=True)
    adata.X = np.log2(adata.layers["raw_count"] + 1)

    with open("aibs_500_genes.txt") as fh:
        training_genes = [l.strip() for l in fh if l.strip()]
    user_gene_list = list(adata.var["gene"].values)
    user_gene_idx  = {g: i for i, g in enumerate(user_gene_list)}
    missing = [g for g in training_genes if g not in user_gene_idx]
    if missing:
        raise ValueError(f"Training genes missing from data: {missing}")
    col_order  = [user_gene_idx[g] for g in training_genes]
    adata_filt = adata[:, col_order].copy()
    print(f"Selected {len(training_genes)} genes in AIBS training order (first 5: {training_genes[:5]})", flush=True)

    all_subclasses = sorted(adata.obs["class_label"].astype(str).unique())
    le = LabelEncoder().fit(all_subclasses)

    print(f"Loading model on {DEVICE} …", flush=True)
    model = load_model(N_GENES, DEVICE)

    bs_mask  = adata.obs["brain_set"].astype(int).values == TARGET_BS
    adata_bs = adata_filt[bs_mask]
    obs_bs   = adata.obs[bs_mask]
    meta_bs  = build_sampler_metadata(obs_bs, le)
    adata_bs = adata_bs[meta_bs["cell_label"].values]

    print(f"brain_set {TARGET_BS}: {bs_mask.sum():,} cells  MAX_CELLS={MAX_CELLS}", flush=True)

    sampler = CenterMaskSampler(
        metadata=meta_bs, adata=adata_bs,
        patch_size=PATCH_SIZE, cell_id_colname="cell_label",
        cell_type_colname="cell_type", tissue_section_colname="section_label",
        max_num_cells=MAX_CELLS,
    )
    loader = torch.utils.data.DataLoader(
        sampler, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=collate, prefetch_factor=2,
    )

    embeds = []
    bf16 = torch.amp.autocast(DEVICE, dtype=torch.bfloat16)
    with torch.inference_mode(), bf16:
        for batch in tqdm.tqdm(loader, desc=f"brain_set {TARGET_BS}", unit="batch", dynamic_ncols=True):
            emb = model(batch)["neighborhood_repr"].detach().cpu().float().numpy()
            embeds.append(emb)
    embeddings = np.concatenate(embeds, axis=0)
    print(f"embeddings: {embeddings.shape}", flush=True)

    print(f"Clustering into {N_CLUSTERS} groups …", flush=True)
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=5, batch_size=4096)
    labels = km.fit_predict(embeddings)

    obs_flat = pd.DataFrame(
        {
            "cell_id":         obs_bs.index.astype(str),
            "class_label":     obs_bs["class_label"].astype(str).values,
            "brain_set":       TARGET_BS,
            "spatial_cluster": labels.astype(int),
        },
        index=pd.Index(obs_bs.index.astype(str), name="cell_id"),
    ).reset_index(drop=True)
    obs_flat.index = pd.Index(obs_bs.index.astype(str), name="cell_id")

    adata_out = ad.AnnData(X=embeddings.astype(np.float32), obs=obs_flat)
    adata_out.obsm["spatial"] = np.column_stack(
        [obs_bs["x_spatial"].values, obs_bs["y_spatial"].values]
    ).astype(np.float32)
    adata_out.uns = {
        "brain_set": TARGET_BS, "n_clusters": N_CLUSTERS,
        "model": "CellTransformer", "embed_dim": 384,
        "n_genes_input": N_GENES, "patch_size": list(PATCH_SIZE),
        "max_cells": MAX_CELLS, "weights_file": WEIGHTS_PATH,
        "coord_scale": COORD_SCALE,
    }

    out = OUTPUT_DIR / f"brain_set_{TARGET_BS}.h5ad"
    adata_out.write_h5ad(out, compression="gzip")
    print(f"Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
