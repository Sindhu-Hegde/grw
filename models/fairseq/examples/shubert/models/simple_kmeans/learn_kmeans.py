# Modified hubert learn_kmeans.py

import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")

def get_km_model(
    n_clusters, 
    init, 
    max_iter, 
    batch_size, 
    tol, 
    max_no_improvement, 
    n_init, 
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )

def load_feature(manifest_path, column_index, seed, percent):
    assert percent <= 1.0
    features = []
    with open(manifest_path, "r") as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) != 5:
                # raise error that it should be 5 columns
                raise ValueError(f"Expected 5 columns, got {len(items)}")
            feature_path = items[column_index]
            feature = np.load(feature_path, mmap_mode="r")

            num_samples = int(len(feature) * percent)
            indices = np.random.choice(len(feature), num_samples, replace=False)
            feature = feature[indices]
            features.append(feature)
    
    features = np.concatenate(features, axis=0)

    logging.info(f"Loaded feature with dimension {features.shape}")
    return features

def learn_kmeans(
    manifest_path, 
    column_index, 
    km_path, 
    n_clusters, 
    seed, 
    percent, 
    init, 
    max_iter, 
    batch_size, 
    tol, 
    n_init, 
    reassignment_ratio, 
    max_no_improvement,
):
    np.random.seed(seed)
    feat = load_feature(manifest_path, column_index, seed, percent)
    km_model = get_km_model(
        n_clusters, 
        init, 
        max_iter, 
        batch_size, 
        tol, 
        max_no_improvement, 
        n_init, 
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)
    
    inertia = -km_model.score(feat) / len(feat)
    logger.info("total inertia: %.5f", inertia)
    logger.info("finished successfully")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=str)
    parser.add_argument("column_index", type=int, choices=[0, 1, 2, 3], help="0: face, 1: left_hand, 2: right_hand, 3: body")
    parser.add_argument("km_path", type=str)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--percent", default=1.0, type=float, help="sample a subset; between 0 and 1")
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))

# python learn_kmeans.py ${manifest_path} ${column_index} ${km_path} ${n_clusters} [options]
# e.g. python learn_kmeans.py /path/to/manifest.tsv 0 /path/to/models/face_kmeans_500.pkl 500 --percent 0.1