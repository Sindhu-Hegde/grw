import os
import numpy as np

import torch
from torch.utils import data
from torch import nn
import torch.nn.functional as F

import random
import math
from functools import partial

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

class DataGenerator_Gestures(data.Dataset):

	def __init__(self, df, feature_dir, fps=25, apply_augmentations=False):

		self.df = df
		self.feature_dir = feature_dir
		self.fps = fps
		self.apply_augmentations = apply_augmentations
		
	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):

		data = self.df.iloc[index]

		file = data.fname
		
		feat_fname = os.path.join(self.feature_dir, file+".npy")
		if not os.path.exists(feat_fname):
			print("Feature file does not exist: ", feat_fname)
			return None

		word = data.target_word
		
		visual_feats, visual_mask, src_idx = self.load_visual_feats_shubert(feat_fname)
		if visual_feats is None:
			# print("Visual feats is None: ", feat_fname)
			return None

		visual_feats = torch.FloatTensor(np.array(visual_feats))

		# print("Visual feats shape: ", visual_feats.shape)
		# print("Visual mask shape: ", visual_mask.shape)			

		out_dict = {
			'visual_feats': visual_feats,
			'visual_mask': visual_mask,
			'word': word,
			'file': file,
			'info': data,
			'src_idx': src_idx,
		}

		return out_dict

	def load_visual_feats_shubert(self, fname):

		try:
			visual_feats = np.load(fname)			
			visual_feats = np.transpose(visual_feats, (1, 0, 2))  # (T, 12, 768)
			# print("Visual features: ", visual_feats.shape)

			# Identity mapping when no augmentation is applied, so callers can map frame
			# annotations through src_idx unconditionally
			src_idx = torch.arange(len(visual_feats))
			if self.apply_augmentations:
				aug = TemporalAug(use_interpolate_for_speed=False)
				visual_feats, src_idx = aug.apply_random_temporal_augs(visual_feats)

			visual_mask = torch.ones((len(visual_feats)))

		except:
			# print("Error in loading Shubert feats: ", fname)
			return None, None, None


		return visual_feats, visual_mask, src_idx


class TemporalAug:
    def __init__(self, use_interpolate_for_speed=False):
        """
        use_interpolate_for_speed: if False, speed resampling uses index-based nearest resample (fast).
                                   if True, uses F.interpolate (smoother but slower).
        """
        self.use_interpolate_for_speed = use_interpolate_for_speed
        self._cached_shape = None
        self._cached_C = None

    @staticmethod
    def _ensure_tensor(feat):
        if not torch.is_tensor(feat):
            return torch.as_tensor(feat)
        return feat

    @staticmethod
    def _to_flat_NCL_no_copy(x):
        T = x.shape[0]
        rest_shape = x.shape[1:]
        C = int(math.prod(rest_shape)) if rest_shape else 1
        flat = x.reshape(T, C).transpose(0, 1).unsqueeze(0)  # (1, C, T)
        return flat, rest_shape

    @staticmethod
    def _from_flat_NCL_no_copy(flat, rest_shape):
        out = flat.squeeze(0).transpose(0, 1)  # (T, C)
        if rest_shape:
            out = out.reshape(out.shape[0], *rest_shape)
        else:
            out = out.reshape(out.shape[0], )
        return out

    def speed_resample_fast(self, feat, speed_factor, min_frames=1):
        """
        Fast index-based resample (nearest neighbor in time).
        This avoids F.interpolate allocations.
        Returns (resampled_feat, idx), where idx[t] is the input frame taken at output
        position t. idx is non-decreasing, so frames stay in temporal order.
        """
        feat = self._ensure_tensor(feat)
        T = feat.shape[0]
        if T <= 1:
            return feat, torch.arange(T)

        target_T = max(min_frames, int(round(T / float(speed_factor))))
        if target_T == T:
            return feat, torch.arange(T)

        # nearest neighbor index mapping
        # create float positions in source space that map to indices
        # pos_new = linspace(0, T-1, target_T)
        pos = torch.linspace(0, T - 1, steps=target_T, device=feat.device)
        idx = pos.round().long().clamp(0, T - 1)  # nearest neighbor
        # index along dim 0 (time)
        return feat[idx], idx.cpu()

    def speed_resample_interp(self, feat, speed_factor, min_frames=1):
        # slower but smoother
        # Returns (resampled_feat, idx). Each output frame blends its neighbours, so idx
        # records the nearest source frame -- an approximation used only for bookkeeping.
        feat = self._ensure_tensor(feat)
        T = feat.shape[0]
        if T <= 1:
            return feat, torch.arange(T)
        target_T = max(min_frames, int(round(T / float(speed_factor))))
        if target_T == T:
            return feat, torch.arange(T)

        flat, rest_shape = self._to_flat_NCL_no_copy(feat)
        flat = flat.contiguous().to(torch.float32)  # ensure contiguous
        flat_res = F.interpolate(flat, size=target_T, mode="linear", align_corners=False)
        out = self._from_flat_NCL_no_copy(flat_res, rest_shape)

        pos = torch.linspace(0, T - 1, steps=target_T)
        idx = pos.round().long().clamp(0, T - 1)

        return out.to(dtype=feat.dtype), idx

    def drop_random_frames(self, feat, max_drop_frac=0.2, min_frames=1):
        """
        Drop a random subset of frames. The kept indices are sorted before indexing, so
        the surviving frames stay in their original temporal order.
        Returns (kept_feat, keep_idx) with keep_idx the surviving input positions.
        """
        feat = self._ensure_tensor(feat)
        T = feat.shape[0]
        if T <= min_frames:
            return feat, torch.arange(T)
        max_drop = int(math.floor(T * max_drop_frac))
        if max_drop < 1:
            return feat, torch.arange(T)
        # sample keep_count indices (faster than randperm if keep_count ~ T)
        drop_count = random.randint(1, max_drop)
        keep_count = max(min_frames, T - drop_count)
        # If keep_count is close to T, selecting indices to drop is cheaper, but torch.randperm is still OK.
        # Use torch.multinomial on uniform weights for faster random selection without full perm:
        probs = torch.ones(T, device=feat.device)
        idx = torch.multinomial(probs, num_samples=keep_count, replacement=False)
        idx, _ = torch.sort(idx)
        return feat[idx], idx.cpu()

    def shift_with_zerofill(self, feat, shift, src_idx=None):
        """
        Translate the sequence in time, zero-filling the frames that scroll into view.
        Returns (shifted_feat, shifted_src_idx), where the zero-filled positions carry
        -1 so they can be told apart from real frames.
        """
        feat = self._ensure_tensor(feat)
        T = feat.shape[0]
        if src_idx is None:
            src_idx = torch.arange(T)

        if shift == 0 or T <= 1:
            return feat, src_idx

        # preallocate output to avoid roll copies
        out = torch.zeros_like(feat)
        out_src = torch.full((T,), -1, dtype=torch.long)
        if shift > 0:
            # shift right: new[t] = old[t-shift] for t=shift..T-1
            out[shift:] = feat[:T - shift]
            out_src[shift:] = src_idx[:T - shift]
        else:
            # shift < 0: shift left
            k = -shift
            out[:T - k] = feat[k:]
            out_src[:T - k] = src_idx[k:]
        return out, out_src

    def apply_random_temporal_augs(self, feat,
                                   p_drop=0.3, max_drop_frac=0.3,
                                   p_shift=0.3, max_shift=8,
                                   p_speed=0.3, min_speed=0.75, max_speed=1.25,
                                   min_frames=4, allow_none=True, random_state=None):
        """Returns (augmented_feat, src_idx), where src_idx[t] is the index of the input
        frame sitting at output position t, and -1 marks a zero-filled frame. Ground-truth
        frame annotations are mapped through src_idx, so they stay aligned whichever
        combination of augmentations fired. src_idx is non-decreasing over the real
        frames: none of the ops reorders the sequence."""
        if random_state is not None:
            random.seed(random_state)

        feat = self._ensure_tensor(feat)
        src_idx = torch.arange(feat.shape[0])

        include_drop = bool(random.random() < float(p_drop))
        include_shift = bool(random.random() < float(p_shift))
        include_speed = bool(random.random() < float(p_speed))

        if (not include_drop) and (not include_shift) and (not include_speed) and (not allow_none):
            choice = random.choice(["drop", "shift", "speed"])
            include_drop = (choice == "drop")
            include_shift = (choice == "shift")
            include_speed = (choice == "speed")

        # Applied in a fixed order (drop -> speed -> shift) rather than a shuffled one, so
        # that composing the index mappings is unambiguous. Each op maps the running
        # src_idx exactly the way it maps the features.
        cur = feat

        if include_drop:
            cur, keep_idx = self.drop_random_frames(cur, max_drop_frac=max_drop_frac, min_frames=min_frames)
            src_idx = src_idx[keep_idx]

        if include_speed:
            speed_factor = random.uniform(min_speed, max_speed)
            if self.use_interpolate_for_speed:
                cur, resample_idx = self.speed_resample_interp(cur, speed_factor=speed_factor, min_frames=min_frames)
            else:
                cur, resample_idx = self.speed_resample_fast(cur, speed_factor=speed_factor, min_frames=min_frames)
            src_idx = src_idx[resample_idx]

        if include_shift:
            shift_amt = random.randint(-max_shift, max_shift)
            cur, src_idx = self.shift_with_zerofill(cur, shift_amt, src_idx=src_idx)

        return cur, src_idx


def collate_data(data):

	visual_feats = []
	visual_mask = []
	words = []
	files = []	
	info = []
	src_idxs = []

	for sample in data:

		if sample is None:
			continue

		feats = sample['visual_feats']
		feat_mask = sample['visual_mask']
		word = sample['word']
		file = sample['file']
		inf = sample['info']
		si = sample.get('src_idx', None)


		visual_feats.append(feats)
		visual_mask.append(feat_mask)
		words.append(word)
		files.append(file)
		info.append(inf)
		src_idxs.append(si)
		
		
	if len(visual_feats) > 0:
		visual_feats = nn.utils.rnn.pad_sequence(visual_feats, batch_first=True, padding_value=0)
		visual_mask = nn.utils.rnn.pad_sequence(visual_mask, batch_first=True, padding_value=0)
	else:
		return 0


	out_dict = {
			'visual_feats': visual_feats,
			'visual_mask': visual_mask,
			'word': words,
			'file': files,
			'info': info,
			'src_idx': src_idxs,
		}

	return out_dict		