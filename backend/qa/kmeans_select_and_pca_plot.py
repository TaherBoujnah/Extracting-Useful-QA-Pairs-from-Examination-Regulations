# backend/qa/kmeans_select_and_pca_plot.py
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def cosine_dist_to_centroid(emb: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    X = emb[idxs]
    centroid = X.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = X @ centroid
    return 1.0 - sims

def choose_diverse_subset(labels: np.ndarray, emb: np.ndarray, target_total: int) -> List[int]:
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    base_alloc = target_total // k
    remainder = target_total % k
    
    counts_per_cluster = {lab: base_alloc for lab in unique_labels}
    
    cluster_sizes = {lab: np.sum(labels == lab) for lab in unique_labels}
    sorted_by_size = sorted(unique_labels, key=lambda x: cluster_sizes[x], reverse=True)
    for i in range(remainder):
        counts_per_cluster[sorted_by_size[i]] += 1
        
    selected_idxs = []
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0: continue
            
        dists = cosine_dist_to_centroid(emb, idxs)
        sorted_local = np.argsort(dists)
        
        n_to_take = min(counts_per_cluster[lab], len(idxs))
        selected_idxs.extend(idxs[sorted_local[:n_to_take]])
        
    return selected_idxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--plot_png", required=True)
    ap.add_argument("--embed_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--n_clusters", type=int, default=12)
    ap.add_argument("--target_total", type=int, default=50)

    args = ap.parse_args()

    # 1. Load data
    rows = read_jsonl(Path(args.input_jsonl))
    rows = [r for r in rows if (r.get("question") or "").strip() and (r.get("answer") or "").strip()]
    questions = [r["question"].strip() for r in rows]

    # 2. Embed data
    print("Embedding questions...")
    model = SentenceTransformer(args.embed_model)
    emb = model.encode(questions, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    # 3. K-Means Clustering
    print("Running K-Means clustering...")
    km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(emb)

    # 4. Select Subset
    chosen_idxs = choose_diverse_subset(labels, emb, args.target_total)
    selected = [rows[i] for i in chosen_idxs]
    write_jsonl(Path(args.out_jsonl), selected)

    # 5. PCA Dimensionality Reduction
    print("Running PCA for visualization...")
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(emb)

    # 6. Plotting
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Create a mask for discarded questions
    mask = np.ones(len(rows), dtype=bool)
    mask[chosen_idxs] = False
    
    # Plot Discarded Background
    ax.scatter(
        reduced_embeddings[mask, 0], 
        reduced_embeddings[mask, 1], 
        c='lightgrey', 
        alpha=0.4, 
        s=20, 
        label='Discarded FAQs'
    )

    # Plot Selected Questions (Colored by Cluster)
    cmap = plt.get_cmap('tab20')
    unique_chosen_labels = np.unique(labels[chosen_idxs])
    
    for cluster_id in unique_chosen_labels:
        idx_for_this_cluster = [idx for idx in chosen_idxs if labels[idx] == cluster_id]
        if idx_for_this_cluster:
            ax.scatter(
                reduced_embeddings[idx_for_this_cluster, 0], 
                reduced_embeddings[idx_for_this_cluster, 1], 
                color=cmap(cluster_id % 20), 
                edgecolor='black',
                linewidth=1,
                s=80, 
                alpha=0.9,
                label=f'Selected (Cluster {cluster_id})'
            )

    ax.set_title(f'PCA Projection of FAQ Knowledge Base\n(50 Selected Questions across {args.n_clusters} Clusters)', fontsize=14)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    
    # Legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save and show
    fig.savefig(args.plot_png, bbox_inches='tight')
    print(f"✅ Saved {len(selected)} FAQs and PCA Plot successfully to {args.plot_png}!")
    plt.show()

if __name__ == "__main__":
    main()