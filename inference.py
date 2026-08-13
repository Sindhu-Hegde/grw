"""
Runs inference for a single video's precomputed SHuBERT features:
  1. Semantic classification: is the target interval a semantic ("iconic")
     gesture, or non-semantic?
  2. Word recognition + localization: top-5 predicted gesture word, and the
     predicted gesture start/end frame within the clip.
"""
import argparse
import torch
from torch import nn
import numpy as np
import os
from dataloader import *
from models.grw_models import *


parser = argparse.ArgumentParser(description='Run semantic classification + word recognition/localization inference on a single video')
parser.add_argument('--model_semantic_classification', required=True, help='Path of the semantic classification checkpoint', default=None, type=str)
parser.add_argument('--model_recognition_localization', required=True, help='Path of the word recognition/localization checkpoint', default=None, type=str)
parser.add_argument('--input_feature', required=True, help='Path of the input SHuBERT feature (.npy) file')
parser.add_argument('--target_intervals', required=False, default=None, help='Target frame interval "[start,end]" for semantic classification (default: the full clip)')
parser.add_argument('--semantic_thr', required=False, default=0.7, type=float, help='Confidence threshold above which a clip is classified as semantic')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

word_label_dict = {'bye': 0, 'below': 1, 'push': 2, 'specific': 3, 'expand': 4, 'together': 5,
					'hello': 6, 'direction': 7, 'move': 8, 'whole': 9, 'switch': 10, 'above': 11,
					'huge': 12, 'throw': 13, 'turn': 14, 'straight': 15, 'five': 16, 'open': 17,
					'three': 18, 'mix': 19, 'small': 20, 'tiny': 21, 'four': 22, 'no': 23,
					'raise': 24, 'large': 25, 'top': 26, 'circle': 27, 'two': 28, 'entire': 29,
					'wide': 30, 'long': 31, 'lower': 32, 'grow': 33, 'balance': 34, 'down': 35,
					'press': 36, 'separate': 37, 'hold': 38, 'full': 39, 'high': 40, 'lift': 41,
					'spin': 42, 'block': 43, 'force': 44, 'big': 45, 'increase': 46, 'front': 47,
					'focus': 48, 'develop': 49, 'layer': 50, 'cross': 51, 'process': 52, 'flow': 53,
					'combine': 54, 'grab': 55, 'strong': 56, 'deep': 57, 'roll': 58, 'connect': 59,
					'wrap': 60, 'build': 61, 'close': 62, 'flip': 63, 'global': 64, 'curve': 65,
					'shake': 66, 'around': 67, 'height': 68, 'narrow': 69, 'stop': 70, 'short': 71,
					'many': 72, 'us': 73, 'quick': 74, 'various': 75, 'begin': 76, 'broad': 77,
					'count': 78, 'loop': 79, 'round': 80, 'track': 81, 'run': 82, 'stretch': 83,
					'rotate': 84, 'little': 85, 'explode': 86, 'horizontal': 87, 'join': 88,
					'point': 89, 'shrink': 90, 'perfect': 91, 'reduce': 92, 'back': 93,
					'heavy': 94, 'decrease': 95, 'engage': 96, 'twist': 97, 'absorb': 98,
					'transition': 99, 'she': 100, 'evolve': 101, 'wave': 102, 'zoom': 103,
					'bottom': 104, 'fight': 105, 'condense': 106, 'angle': 107, 'peak': 108,
					'wait': 109, 'less': 110, 'merge': 111, 'tie': 112, 'spiral': 113,
					'slide': 114, 'collect': 115, 'elevate': 116, 'knock': 117, 'compress': 118,
					'interaction': 119, 'her': 120, 'descend': 121, 'transform': 122,
					'yes': 123, 'catch': 124, 'tight': 125, 'break': 126, 'grasp': 127,
					'tilt': 128, 'pieces': 129, 'eat': 130, 'boost': 131, 'walk': 132,
					'hashtag': 133, 'hug': 134, 'collide': 135, 'cup': 136, 'bundle': 137,
					'pause': 138, 'gigantic': 139, 'embrace': 140, 'bounce': 141,
					'overlap': 142, 'call': 143, 'few': 144, 'branch': 145, 'trap': 146,
					'link': 147, 'arc': 148, 'beautiful': 149, 'ascend': 150, 'unify': 151,
					'stack': 152, 'look': 153, 'barrier': 154
					}

def load_checkpoint(model, path):
	
	if use_cuda:
		checkpoint = torch.load(path)
	else:
		checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

	s = checkpoint["state_dict"]
	new_s = {}
	
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	print("Loaded checkpoint from: {}".format(path))

	return model.eval()

def load_visual_feats_shubert(feat_path):
	"""Load per-layer SHuBERT features saved as (L, T, D) and rearrange to (T, L, D) for the models' layer-pooling."""
	try:
		visual_feats = np.load(feat_path)
		visual_feats = np.transpose(visual_feats, (1, 0, 2))  # (T, 12, 768)

		visual_feats = torch.FloatTensor(visual_feats)
		visual_mask = torch.ones((len(visual_feats)))
		print("Loaded SHuBERT visual features: ", visual_feats.shape)
	except Exception as e:
		print(f"Error in loading SHuBERT visual features from {feat_path}: {e}")
		return None, None
	return visual_feats, visual_mask

def merge_segments(segments, merge_gap_frames=1):
	"""
	Merge list of (s_idx, e_idx) pairs if gap <= merge_gap_frames.
	segments assumed sorted by start.
	"""
	if len(segments) == 0:
		return []
	merged = []
	cur_s, cur_e = segments[0]
	for s, e in segments[1:]:
		if s - cur_e <= merge_gap_frames:
			# extend
			cur_e = max(cur_e, e)
		else:
			merged.append((cur_s, cur_e))
			cur_s, cur_e = s, e
	merged.append((cur_s, cur_e))
	return merged

def probs_to_single_segment_by_scoring(probs: np.ndarray, fps: int = 25, thr: float = 0.5,
									   min_duration_s: float = 0.05, merge_gap_s: float = 0.04):
	"""
	1) threshold probs to get islands (frame indices),
	2) merge islands with small gaps,
	3) score each merged island by sum(probs) and return the single best one.
	If no islands found, fallback to expanding around the global peak.
	Returns (s_sec, e_sec) as seconds.
	"""
	T = len(probs)
	mask = probs >= thr
	# get contiguous segments in frame indices
	segments = []
	i = 0
	while i < T:
		if not mask[i]:
			i += 1
			continue
		j = i
		while j + 1 < T and mask[j+1]:
			j += 1
		segments.append((i, j))
		i = j + 1

	# merge tiny gaps (convert seconds->frames)
	merge_gap_frames = max(1, int(round(merge_gap_s * fps)))
	segments = merge_segments(segments, merge_gap_frames=merge_gap_frames)

	# keep only segments meeting min duration
	min_d_frames = max(1, int(round(min_duration_s * fps)))
	segments = [s for s in segments if (s[1] - s[0] + 1) >= min_d_frames]

	if len(segments) == 0:
		# fallback: expand around peak +/- 1 frame (or wider)
		peak = int(np.argmax(probs))
		s_idx = max(0, peak - 1)
		e_idx = min(T - 1, peak + 1)
		return s_idx, e_idx

	# score each segment by sum(probabilities within it)
	seg_scores = []
	for (s_idx, e_idx) in segments:
		score = probs[s_idx:(e_idx+1)].sum()  # prefer longer + higher prob
		seg_scores.append(score)

	best_idx = int(np.argmax(seg_scores))
	best_s, best_e = segments[best_idx]
	return best_s, best_e


def infer(model_semantic_classification, model_recognition_localization, visual_feats, visual_mask, target_intervals, semantic_thr, min_duration_s=0.05, fps=25, thr=0.5):
	"""Run both models on one clip's features and print their predictions."""

	# Semantic classification: is the target interval a semantic gesture?
	with torch.no_grad():
		semantic_logits, _ = model_semantic_classification(visual_feats, target_intervals=target_intervals, x_mask=visual_mask.unsqueeze(1))
		semantic_conf = torch.sigmoid(semantic_logits).detach().cpu().numpy()[0]
		if semantic_conf >= semantic_thr:
			semantic_pred = "'Semantic' class"
		else:
			semantic_pred = "'Non-semantic' class"
		print("\nSemantic classification prediction: \n  {}".format(semantic_pred))

	# Word recognition + localization
	with torch.no_grad():
		output = model_recognition_localization(visual_feats, visual_mask.unsqueeze(1))
		cls_logits = output['cls_logits']       # (B, C)
		cls_probs = nn.functional.softmax(cls_logits.detach(), dim=1)  # (B, C)
		inv_word_label_dict = {v: k for k, v in word_label_dict.items()}
		_, top5_indices = torch.topk(cls_probs[0], k=5)
		print("\nWord classification top-5 predictions:")
		for rank, idx in enumerate(top5_indices.tolist()):
			print(f"  {rank + 1}. {inv_word_label_dict[idx]:15s}")

		loc_logits = output['loc_logits']       # (B, T)
		loc_probs = torch.sigmoid(loc_logits).detach().cpu().numpy()[0]  # T
		pred_boundaries = probs_to_single_segment_by_scoring(loc_probs, fps=fps, thr=thr, min_duration_s=min_duration_s, merge_gap_s=0.04)
		print("\nGesture localization (boundary) prediction: \n  Gesture start frame: {} | Gesture end frame: {}\n".format(pred_boundaries[0], pred_boundaries[1]))


if __name__ == "__main__":

	# Load class dictionary
	all_words = list(word_label_dict.keys())

	# Load semantic classification model. Architecture args (use_layer_weights)
	# must match how the checkpoint was trained (see evaluate_semantic_classification.py).
	model_semantic_classification = Semantic_Classifier(input_dim=768, num_classes=1, use_layer_weights=True).cuda()
	model_semantic_classification = load_checkpoint(model_semantic_classification, args.model_semantic_classification)

	# Load word recognition + localization model. Architecture args (N, h, d_model,
	# use_layer_weights) must match how the checkpoint was trained (see
	# evaluate_recognition_localization.py).
	model_recognition_localization = Word_Recognition_Localization(
		input_dim=768, num_classes=len(all_words), N=6, h=12, d_model=768, use_layer_weights=True).cuda()
	model_recognition_localization = load_checkpoint(model_recognition_localization, args.model_recognition_localization)

	# Load feature file
	if not os.path.exists(args.input_feature):
		print("Error: Input feature file not found: ", args.input_feature)
		exit(0)
	visual_feats, visual_mask = load_visual_feats_shubert(args.input_feature)
	visual_feats = visual_feats.unsqueeze(0).cuda()
	visual_mask = visual_mask.unsqueeze(0).cuda()

	# Get target intervals
	start, end = 0, visual_feats.shape[1]-1
	if args.target_intervals is not None:
		# Support both string and list types for -- target_intervals argument
		if isinstance(args.target_intervals, str):
			target_intervals = [list(map(int, args.target_intervals.strip().strip("[](){}").split(",")))]
		else:
			target_intervals = [list(args.target_intervals)]
		print("Target intervals: ", target_intervals)
	else:
		target_intervals = [[start, end]]
	target_intervals = torch.tensor(target_intervals)

	print("----------------------------------")
	print("----------- RESULTS --------------")
	print("----------------------------------")

	# Infer
	infer(model_semantic_classification, model_recognition_localization, visual_feats, visual_mask, target_intervals, args.semantic_thr)
