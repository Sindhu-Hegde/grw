import argparse
import torch
from torch import nn
from torch.utils import data as data_utils
import numpy as np
from dataloader import *
from models.grw_models import *
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple

parser = argparse.ArgumentParser(description='Code to evaluate the classifier')
parser.add_argument('--test_csv', type=str, required=True, help='Path of the test-data csv file')
parser.add_argument('--checkpoint_path', required=True, help='Path of the trained model', default=None, type=str)
parser.add_argument('--feature_dir', type=str, required=True, help='Feature directory')
parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size to train the model')
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


test_words = ['tiny', 'bye', 'together', 'full', 'height', 'entire', 'below', 'specific', 'push',
			  'direction', 'large', 'straight', 'wide', 'huge', 'slide', 'many', 'block', 'close', 'front',
			  'reduce', 'broad', 'open', 'quick', 'combine', 'no', 'track', 'process', 'twist', 'angle',
			  'wrap', 'balance', 'down', 'knock', 'three', 'evolve', 'mix', 'expand', 'above', 'press',
			  'turn', 'layer', 'throw', 'explode', 'hello', 'two', 'switch', 'big', 'wait', 'fight',
			  'high', 'begin', 'hold', 'top', 'move', 'whole', 'raise', 'round', 'circle', 'her',
			  'curve', 'flip', 'join', 'lift', 'five', 'cross', 'four', 'force', 'global', 'around', 'short',
			  'stretch', 'connect', 'deep', 'small', 'grab', 'stop', 'lower', 'merge', 'grow', 'she',
			  'increase', 'develop', 'interaction', 'spin', 'various', 'decrease', 'shake', 'catch', 'us',
			  'separate', 'run', 'focus', 'strong', 'heavy', 'wave', 'build', 'flow', 'perfect', 'long', 'horizontal']


# Inverse mapping: original label -> word
label_to_word_dict = {v: k for k, v in word_label_dict.items()}

def read_data(path):

	df = pd.read_csv(path)
	print("Total files: ", len(df))

	return df

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

def segment_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
	s1, e1 = pred
	s2, e2 = gt
	inter_s = max(s1, s2)
	inter_e = min(e1, e2)
	inter = max(0.0, inter_e - inter_s)
	union = max(e1, e2) - min(s1, s2)
	if union <= 0: return 0.0
	return inter / union

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
	Returns (s_idx, e_idx) as frame indices.
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
		peak = int(np.argmax(probs))
		s_idx = max(0, peak - 1)
		e_idx = min(T - 1, peak + 1)
		return s_idx, e_idx

	# score each segment by sum(probabilities within it)
	seg_scores = [probs[s_idx:(e_idx+1)].sum() for (s_idx, e_idx) in segments]

	best_idx = int(np.argmax(seg_scores))
	return segments[best_idx]


def compute_metrics(all_probs: np.ndarray, all_gts: np.ndarray, ious: np.ndarray,
					 ks: List[int] = [1, 5, 10]):
	"""
	Acc@K: fraction of videos where the GT class is among the top-K predictions.
	Also reports the mean temporal IoU between predicted and ground-truth segments.

	all_probs: (N, C) softmax probabilities
	all_gts:   (N,)   integer GT labels
	ious:      (N,)   per-video temporal IoU between predicted and GT segments
	"""
	topk_max = max(ks)
	top_indices = np.argsort(all_probs, axis=1)[:, ::-1][:, :topk_max]

	print("--- Gesture Recognition Evaluation (Acc@K) ---")
	for k in ks:
		correct = np.any(top_indices[:, :k] == all_gts[:, None], axis=1)
		print(f"  Acc@{k}: {correct.mean()*100:.2f} %")

	print("--- Gesture Localization Evaluation (mIoU) ---")
	miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
	print(f"  Mean IoU (mIoU): {miou:.2f}")

	return


def save_preds_csv(info_dicts, predictions, all_probs, ious, pred_starts, pred_ends, output_path="preds.csv"):
	pred_words = [label_to_word_dict[p] for p in predictions]
	confs = [float(all_probs[i, predictions[i]]) for i in range(len(predictions))]

	df = pd.DataFrame(info_dicts)
	df['pred_word'] = pred_words
	df['conf'] = confs
	df['pred_start'] = pred_starts
	df['pred_end'] = pred_ends
	df['iou'] = ious

	df = df.rename(columns={'gesture_start': 'gt_start', 'gesture_end': 'gt_end'})

	df = df[['fname', 'speech_start', 'speech_end', 'gt_start', 'gt_end', 'pred_word', 'conf', 'pred_start', 'pred_end', 'iou']]
	df.to_csv(output_path, index=False)
	print(f"Saved predictions to: {output_path}")


def evaluate(model, test_data_loader, min_duration_s=0.05, fps=25, thr=0.5):

	print('Evaluating for {} steps'.format(len(test_data_loader)))

	num_classes = len(word_label_dict)
	test_class_indices = set(word_label_dict[w] for w in test_words if w in word_label_dict)
	test_class_mask = torch.zeros(num_classes, dtype=torch.float32, device='cuda')
	for idx in test_class_indices:
		test_class_mask[idx] = 1.0

	per_video_max_ious = []
	predictions, gts = [], []
	all_probs = []
	info_dicts = []
	pred_starts = []
	pred_ends = []

	for batch_sample in tqdm(test_data_loader):

		try:
			if batch_sample == 0:
				continue

			visual_feats = batch_sample["visual_feats"].cuda()
			visual_mask = batch_sample["visual_mask"].cuda()
			words = batch_sample["word"]

			labels = [word_label_dict[word] for word in words]
			B = visual_feats.shape[0]

			with torch.no_grad():
				out = model(visual_feats, visual_mask.unsqueeze(1))
				loc_logits = out['loc_logits']  # (B, T)
				cls_logits = out['cls_logits']  # (B, C)
				probs = torch.sigmoid(loc_logits).cpu().numpy()  # (B, T)

				# --------- classification prediction ---------
				# Mask non-test logits to -inf so softmax only normalizes over test classes
				masked_logits = cls_logits.detach().clone()
				masked_logits[:, test_class_mask == 0] = float('-inf')
				pred_all_classes = nn.functional.softmax(masked_logits, dim=1)

				all_probs.append(pred_all_classes.cpu().numpy())
				pred = torch.argmax(pred_all_classes, dim=1)
				predictions.extend(pred.cpu().numpy().tolist())
				gts.extend(labels)
				# ----------------------------------------------

				# ----- localization prediction per-sample -----
				for i in range(B):
					prob = probs[i]  # (T,)
					pred_seg = probs_to_single_segment_by_scoring(
						prob, fps=fps, thr=thr, min_duration_s=min_duration_s, merge_gap_s=0.04)

					gt = (batch_sample['info'][i].gesture_start, batch_sample['info'][i].gesture_end)
					max_iou = segment_iou(pred_seg, gt)
					per_video_max_ious.append(max_iou)

					info_dicts.append(batch_sample['info'][i].to_dict())
					pred_starts.append(pred_seg[0])
					pred_ends.append(pred_seg[1])

		except Exception as e:
			print("Error: ", e)
			continue

	predictions = np.array(predictions)
	gts = np.array(gts)
	all_probs = np.concatenate(all_probs, axis=0)  # (N, C)
	per_video_ious = np.array(per_video_max_ious)

	compute_metrics(all_probs, gts, per_video_ious)
	save_preds_csv(info_dicts, predictions, all_probs, per_video_ious, pred_starts, pred_ends)

	return


if __name__ == "__main__":

	# Dataset and Dataloader setup
	df_test = read_data(args.test_csv)

	all_words = list(word_label_dict.keys())

	batch_size = args.batch_size
	checkpoint_path = args.checkpoint_path

	test_dataset = DataGenerator_Gestures(df_test, args.feature_dir)
	test_data_loader = data_utils.DataLoader(
		test_dataset, batch_size=batch_size, num_workers=4, collate_fn=lambda x: collate_data(x))

	model = Word_Recognition_Localization(num_classes=len(all_words)).cuda()
	model.cuda()
	model = load_checkpoint(model, checkpoint_path)

	evaluate(model, test_data_loader)
