# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F

from .. import FairseqDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.compute_mask_indices = compute_mask_indices
        if self.compute_mask_indices:
            self.mask_compute_kwargs = mask_compute_kwargs
            self._features_size_map = {}
            self._C = mask_compute_kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats):
        if feats.dim() == 2:
            feats = feats.mean(-1)



        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats


    def crop_to_min_size(self, feat_tensor, target_size, start=None, end=None):
        
        # for labels consistency
        if start is not None and end is not None:
            for part in feat_tensor.keys():
                feat_tensor[part] = feat_tensor[part][start:end, :]
            return feat_tensor, start, end 
        
        size = feat_tensor["face"].size(0)    
        diff = size - target_size
        if diff <= 0:
            return feat_tensor, 0, size


        start = np.random.randint(0, diff + 1)
        end = start + target_size
        for part in feat_tensor.keys():
            feat_tensor[part] = feat_tensor[part][start:end, :]

        return feat_tensor, start, end


    def _compute_mask_indices(self, dims, padding_mask):
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks=2,
                no_overlap=self.mask_compute_kwargs["no_mask_overlap"],
                min_space=self.mask_compute_kwargs["mask_min_space"],
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap=self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space=self.mask_compute_kwargs["mask_channel_min_space"],
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)
    
    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s["face"]) for s in sources]
        target_size = min(min(sizes), self.max_sample_size)

        collated_samples = []

        for i, source in enumerate(sources):
            cropped_source, start_crop, end_crop = self.crop_to_min_size(source, target_size)
            labels = samples[i]["kmeans_labels"]
            cropped_labels, _, _ = self.crop_to_min_size(labels, target_size, start_crop, end_crop)
                
            sample = {
                "face": cropped_source["face"],
                "left_hand": cropped_source["left_hand"],
                "right_hand": cropped_source["right_hand"],
                "body_posture": cropped_source["body_posture"],
                "label_face": cropped_labels["face"],
                "label_left_hand": cropped_labels["left_hand"],
                "label_right_hand": cropped_labels["right_hand"],
                "label_body_posture": cropped_labels["body_posture"],
            }

            collated_samples.append(sample)
            
        input = {"source": collated_samples}
        out = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": input,
        }

        return out     


    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        text_compression_level=TextCompressionLevel.none,
        kmeans_label_paths=None,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            kmeans_label_paths=kmeans_label_paths,
            **mask_compute_kwargs,
        )

        self.text_compressor = TextCompressor(level=text_compression_level)

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 5, line
                sz = int(int(items[-1]))
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append({
                    "face": self.text_compressor.compress(items[0]),
                    "left_hand": self.text_compressor.compress(items[1]),
                    "right_hand": self.text_compressor.compress(items[2]),
                    "body_posture": self.text_compressor.compress(items[3]),
                })
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples") 


        self.sizes = np.array(sizes, dtype=np.int64)

        self.kmeans_labels = self.load_kmeans_labels(kmeans_label_paths)


        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)

    def load_kmeans_labels(self, kmeans_label_paths):
        kmeans_labels = {}
        for part, label_path in kmeans_label_paths.items():
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append([int(x) for x in line.strip().split()])
            kmeans_labels[part] = labels
            
            # Verify that the number of labels matches the number of samples
            assert len(labels) == len(self.fnames), f"Mismatch in number of samples for {part} kmeans labels"
            
        return kmeans_labels    

    def __getitem__(self, index):

        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        feats = {}
        for part in ["face", "left_hand", "right_hand", "body_posture"]:
            fn_part = fn[part]
            fn_part = self.text_compressor.decompress(fn_part)
            path_part = fn_part
            feats_part = torch.from_numpy(np.load(path_part)).float()
            assert feats_part.dim() == 2, feats_part.dim()
            feats[part] = feats_part
        
        item = {"id": index, "source": feats}
        
        item["kmeans_labels"] = {part: self.kmeans_labels[part][index] for part in self.kmeans_labels}
        # convert to tensor of same shape as feat_part i.e. dimension should be (T, 1)
        for part in item["kmeans_labels"]:
            item["kmeans_labels"][part] = torch.tensor(item["kmeans_labels"][part]).unsqueeze(1)
            
        return item


class BinarizedAudioDataset(RawAudioDataset):
    def __init__(
        self,
        data_dir,
        split,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        from fairseq.data import data_utils, Dictionary

        self.fnames_dict = Dictionary.load(os.path.join(data_dir, "dict.txt"))

        root_path = os.path.join(data_dir, f"{split}.root")
        if os.path.exists(root_path):
            with open(root_path, "r") as f:
                self.root_dir = next(f).strip()
        else:
            self.root_dir = None

        fnames_path = os.path.join(data_dir, split)
        self.fnames = data_utils.load_indexed_dataset(fnames_path, self.fnames_dict)
        lengths_path = os.path.join(data_dir, f"{split}.lengths")

        with open(lengths_path, "r") as f:
            for line in f:
                sz = int(int(line.rstrip()))
                assert (
                    sz >= min_sample_size
                ), f"Min sample size is not supported for binarized dataset, but found a sample with size {sz}"
                self.sizes.append(sz)

        self.sizes = np.array(self.sizes, dtype=np.int64)

        self.set_bucket_info(num_buckets)
        logger.info(f"loaded {len(self.fnames)} samples")

    def __getitem__(self, index):
        import soundfile as sf

        fname = self.fnames_dict.string(self.fnames[index], separator="")
        if self.root_dir:
            fname = os.path.join(self.root_dir, fname)

        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats)
        return {"id": index, "source": feats}
