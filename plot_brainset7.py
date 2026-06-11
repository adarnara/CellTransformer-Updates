"""
Plot brain_set 7 — black background, K=10 and K=25, no axis inversion.
Output: inference_output/brain_set7_spatial_k10.png
        inference_output/brain_set7_spatial_k25.png
"""

import pathlib
import anndata as ad
import colorcet as cc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

H5AD_PATH = pathlib.Path("inference_output/brain_set_7.h5ad")

print("Loading brain_set_7.h5ad …", flush=True)
adata      = ad.read_h5ad(H5AD_PATH)
embeds_cat = np.array(adata.X)
xy         = adata.obsm["spatial"]


def plot_k(n_clusters, out_path):
    print(f"Clustering {len(embeds_cat):,} cells into {n_clusters} groups …", flush=True)
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusterer.fit(embeds_cat)
    labels = clusterer.predict(embeds_cat)

    colormap      = sns.color_palette(cc.glasbey, n_colors=n_clusters)
    hex_values    = {k: v for k, v in zip(range(len(colormap)), colormap.as_hex())}
    cluster_color = [hex_values[lbl] for lbl in labels]

    fig, axs = plt.subplots(1, figsize=(10, 5), facecolor="black")
    axs.set_facecolor("black")
    plt.scatter(x=xy[:, 0], y=xy[:, 1], c=cluster_color, s=1, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved → {out_path}", flush=True)


plot_k(10,  pathlib.Path("inference_output/brain_set7_spatial_k10.png"))
plot_k(34,  pathlib.Path("inference_output/brain_set7_spatial_k34.png"))
