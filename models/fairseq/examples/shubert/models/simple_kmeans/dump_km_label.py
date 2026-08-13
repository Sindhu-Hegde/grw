import logging
import os
import sys

import numpy as np

import joblib
import torch
import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

def get_feature_iterator(manifest_path, part):
    
    if part == "face":
        column_index = 0
    elif part == "lhand":
        column_index = 1
    elif part == "rhand":
        column_index = 2
    elif part == "body":
        column_index = 3
    else:
        raise ValueError(f"Invalid manifest path {manifest_path}. Must contain face, lhand, rhand, or body")
 
    with open(manifest_path, "r") as f:
        #root_dir = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert len(items) == 5
            feat_path = items[column_index]
            yield np.load(feat_path)

def dump_label(manifest_path, km_path, lab_dir):
    split_list = km_path.split("/")[-1].rsplit(".", 1)[0].split("_")
    part,_,num_centroids = split_list[0], split_list[1], split_list[2]
    lab_dir = f"{lab_dir}/{part}"
    apply_kmeans = ApplyKmeans(km_path)
    iterator = get_feature_iterator(manifest_path, part)
    os.makedirs(lab_dir, exist_ok=True)
    # km path e.g. /xxx/models/body_kmeans_1024.pkl

            
    lab_path = f"{lab_dir}/{part}_{num_centroids}.km"
    
    with open(lab_path, "w") as f:
        for feat in tqdm.tqdm(iterator):
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")    
    logger.info(f"finished successfully {part}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", help="Path to the manifest file")
    parser.add_argument("km_path", help="Path to the k-means model")
    parser.add_argument("lab_dir", help="Directory to save the output labels")
    args = parser.parse_args()
    
    logging.info(str(args))
    dump_label(**vars(args))

# python dump_km_label.py /path/to/manifest.tsv /path/to/models/face_kmeans_500.pkl /path/to/km_labels
# note the format of the name of the kmeans model is face_kmeans_500.pkl, lhand_kmeans_500.pkl, rhand_kmeans_500.pkl, body_kmeans_500.pkl