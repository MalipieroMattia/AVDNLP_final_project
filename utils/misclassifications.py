import numpy as np
import pandas as pd
import umap
import hdbscan
import re
import spacy
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
import matplotlib.pyplot as plt


# Load spaCy stopwords globally
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = nlp.Defaults.stop_words


class EdgeCaseAnalyzer:

    # -----------------------------------------------------
    # Constructor — requires in-memory model + tokenizer
    # -----------------------------------------------------
    def __init__(
        self,
        df,
        model,
        tokenizer,
        text_column="text",
        batch_size=32,
        max_length=256
    ):
        """
        df          : misclassified dataframe
        model       : FINETUNED MODEL OBJECT already in memory
        tokenizer   : tokenizer object already in memory
        text_column : name of text column
        """
        self.df = df.copy()
        self.texts = df[text_column].tolist()
        self.batch_size = batch_size
        self.max_length = max_length

        # MUST use in-memory model + tokenizer
        self.model = model
        self.tokenizer = tokenizer

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # internal state
        self.embeddings = None
        self.coords = None
        self.labels = None
        self.cluster_keywords = None

        # compute embeddings from the provided model
        self._compute_embeddings()


    # -----------------------------------------------------
    # Compute embeddings from the in-memory model
    # -----------------------------------------------------
    def _compute_embeddings(self):
        all_embeddings = []

        for i in range(0, len(self.texts), self.batch_size):
            batch = self.texts[i : i + self.batch_size]

            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**tokens)

            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

        self.embeddings = np.vstack(all_embeddings)


    # -----------------------------------------------------
    # Auto-label clusters
    # -----------------------------------------------------
    def auto_label_clusters(self, top_k=8):

        labels = self.labels
        texts = self.texts
        cluster_labels = {}

        for cid in sorted(set(labels)):
            if cid == -1:
                continue

            cluster_texts = [t for t, l in zip(texts, labels) if l == cid]

            words = []
            for t in cluster_texts:
                words.extend(re.findall(r"[A-Za-z]+", t.lower()))

            words = [w for w in words if w not in spacy_stopwords and len(w) > 2]

            if not words:
                cluster_labels[cid] = "(no keywords)"
            else:
                top = [w for w, c in Counter(words).most_common(top_k)]
                cluster_labels[cid] = ", ".join(top)

        self.cluster_keywords = cluster_labels
        return cluster_labels


    # -----------------------------------------------------
    # UMAP + HDBSCAN clustering
    # -----------------------------------------------------
    def run_clustering(self, min_cluster_size=10, n_neighbors=20, min_dist=0.05):

        norm = StandardScaler().fit_transform(self.embeddings)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=42
        )
        coords = reducer.fit_transform(norm)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean"
        )
        labels = clusterer.fit_predict(coords)

        self.coords = coords
        self.labels = labels

        return labels, coords


    # -----------------------------------------------------
    # Embedding distance score
    # -----------------------------------------------------
    def compute_distance_scores(self):

        coords = self.coords
        labels = self.labels
        unique = [c for c in set(labels) if c != -1]

        centroids = np.array([
            coords[labels == cid].mean(axis=0)
            for cid in unique
        ])

        _, dist = pairwise_distances_argmin_min(coords, centroids)
        self.df["dist_score"] = dist
        return dist


    # -----------------------------------------------------
    # Structural anomaly score
    # -----------------------------------------------------
    def compute_structural_scores(self):

        if not all(k in self.df.columns
                   for k in ["token_count", "max_dep_depth", "clause_count"]):

            self.df["token_count"] = [len(t.split()) for t in self.texts]
            self.df["max_dep_depth"] = self.df["token_count"]  # placeholder
            self.df["clause_count"] = 1

        cols = ["token_count", "max_dep_depth", "clause_count"]
        score = self.df[cols].rank(pct=True).mean(axis=1)

        self.df["struct_score"] = score
        return score


    # -----------------------------------------------------
    # Probability-aware edge-case score
    # -----------------------------------------------------
    def compute_edge_scores(self):

        dist = self.compute_distance_scores()
        struct = self.compute_structural_scores()

        # defaults
        low_conf = np.zeros(len(self.df))
        wrong_conf = np.zeros(len(self.df))
        true_conf_score = np.zeros(len(self.df))

        if "predicted_prob" in self.df.columns and "true_prob" in self.df.columns:
            pred_conf = self.df["predicted_prob"]
            true_conf = self.df["true_prob"]

            low_conf = 1 - pred_conf
            wrong_conf = pred_conf
            true_conf_score = 1 - true_conf

        noise = (self.labels == -1).astype(float)

        combined = (
            (dist / dist.max()) * 0.30 +
            struct * 0.20 +
            noise * 0.30 +
            low_conf * 0.15 +
            wrong_conf * 0.15 +
            true_conf_score * 0.10
        )

        self.df["edge_score"] = combined
        return combined


    # -----------------------------------------------------
    # Top N edge cases
    # -----------------------------------------------------
    def get_top_edge_cases(self, n=20):
        if "edge_score" not in self.df.columns:
            self.compute_edge_scores()

        return self.df.sort_values("edge_score", ascending=False).head(n)


    # -----------------------------------------------------
    # Tiny clusters
    # -----------------------------------------------------
    def get_tiny_clusters(self, min_size=20):
        counts = Counter(self.labels)
        return [cid for cid, c in counts.items() if 0 < cid != -1 and c < min_size]


    # -----------------------------------------------------
    # Representative examples per cluster
    # -----------------------------------------------------
    def get_cluster_centroid_texts(self, cid, k=5):

        coords = self.coords
        labels = self.labels

        pts = coords[labels == cid]
        center = pts.mean(axis=0)

        dists = np.linalg.norm(pts - center, axis=1)
        idx = np.argsort(dists)[:k]

        return [self.texts[i] for i in np.where(labels == cid)[0][idx]]


    # -----------------------------------------------------
    # Plot clusters (returns figure for W&B)
    # -----------------------------------------------------
    def plot_clusters(self, title="Edge Case Clusters"):

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            c=self.labels,
            cmap="Spectral",
            s=10,
            alpha=0.85
        )
        ax.set_title(title)
        plt.colorbar(sc, ax=ax)
        return fig


    # -----------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------
    def run_full_analysis(self, top_n=20, tiny_threshold=20, title="Edge Case Clusters"):

        self.run_clustering()
        self.auto_label_clusters()
        self.compute_edge_scores()

        return {
            "top_edge_cases": self.get_top_edge_cases(top_n),
            "tiny_clusters": self.get_tiny_clusters(tiny_threshold),
            "cluster_keywords": self.cluster_keywords,
            "coords": self.coords,
            "labels": self.labels,
            "plot": self.plot_clusters(title)
        }
