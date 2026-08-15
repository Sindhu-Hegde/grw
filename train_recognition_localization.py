import os, argparse
from os.path import join
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data as data_utils

from dataloader import *
from models.grw_models import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


parser = argparse.ArgumentParser(description='Code to train the gesture word recognition and localization network')

parser.add_argument('--train_csv', type=str, required=True, help='Path of the train-data csv file')
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Folder to save the trained checkpoints')
parser.add_argument('--feature_dir', type=str, required=True, help='Feature directory')

parser.add_argument('--checkpoint_path', type=str, default=None, help='Resume training from this checkpoint')
parser.add_argument('--apply_augmentations', action='store_true', help='Enable temporal augmentation and shift GT boundaries accordingly')
parser.add_argument('--fps', type=int, default=25, required=False, help='Frame rate of the extracted features')
parser.add_argument('--num_epochs', type=int, default=20, required=False, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size to train the model')
parser.add_argument('--num_workers', type=int, default=8, required=False, help='Number of workers')
parser.add_argument('--lr', type=float, default=1e-4, required=False, help='Learning rate')
parser.add_argument('--val_size', type=int, default=500, required=False, help='Number of files held out from the train set for validation')
parser.add_argument('--seed', type=int, default=42, required=False, help='Random seed for the train/val split')
args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

scaler = torch.amp.GradScaler('cuda')

# The 155 semantic gesture words of the GRW benchmark, mapped to class indices.
# The order defines the classifier output layer and must not be changed, since
# the released checkpoints are trained against exactly this mapping.
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


# ---------------------------
# Read the annotation csv
# ---------------------------
def read_data(path):
	"""
	Read a GRW annotation csv into a dataframe.
	"""
	df = pd.read_csv(path)
	print("Total files: ", len(df))

	return df


# ---------------------------
# Build the train / validation split
# ---------------------------
def train_val_split(df, val_size, seed):
	"""
	Hold out a small validation set from the train split.

	The sample is stratified by target word and takes at least one clip per class,
	so every gesture word is represented in the validation set. The excess is then
	trimmed from the largest classes only, which keeps the rare classes intact and
	the split balanced. Returns (df_train, df_val) with the two sets disjoint.
	"""
	frac = min(1.0, val_size / len(df))
	df_val = df.groupby('target_word', group_keys=False).apply(
					lambda g: g.sample(n=max(1, int(round(len(g)*frac))), random_state=seed))

	while len(df_val) > val_size:
		counts = df_val['target_word'].value_counts()
		# Every class is down to a single clip -- stop here rather than drop a class
		if counts.max() == 1:
			print("Val size raised to {} to keep all {} classes".format(len(df_val), len(counts)))
			break
		drop_idx = df_val[df_val['target_word'] == counts.idxmax()].sample(n=1, random_state=seed).index
		df_val = df_val.drop(drop_idx)

	df_train = df.drop(df_val.index).reset_index(drop=True)
	df_val = df_val.reset_index(drop=True)

	return df_train, df_val


# ---------------------------
# Label conversion: gesture boundaries -> per-frame labels
# ---------------------------
def boundary_to_act_labels(boundary: Tuple[float, float], T: int):
	"""
	Convert a (start_frame, end_frame) boundary into an actionness mask of shape (T,),
	with 1.0 on the frames covered by the gesture and 0.0 elsewhere. Frame indices are
	clamped to [0, T-1], and a reversed boundary degenerates to a single frame.
	"""
	start_frame, end_frame = boundary
	start_idx = int(max(0, min(T-1, start_frame)))
	end_idx = int(max(0, min(T-1, end_frame)))

	act = np.zeros((T,), dtype=np.float32)
	if end_idx >= start_idx:
		act[start_idx:end_idx+1] = 1.0
	else:
		act[start_idx] = 1.0

	return act


# ---------------------------
# Ground-truth boundary, aligned to the temporal augmentation
# ---------------------------
def get_gt_boundary(batch_sample, i, T: int):
	"""
	Ground-truth (start_frame, end_frame) of the gesture for sample i of a batch.

	Temporal augmentation can drop frames, resample the clip to a different speed and
	shift it in time, so the annotated boundary cannot simply be offset. Instead the
	dataloader reports src_idx, giving the input frame behind every output frame, and the
	boundary is the span of output frames whose source falls inside the annotation. With
	no augmentation src_idx is the identity and this returns the annotation unchanged.
	"""
	gs = batch_sample['info'][i].gesture_start
	ge = batch_sample['info'][i].gesture_end

	src_idx = batch_sample['src_idx'][i]
	if src_idx is None:
		return int(max(0, min(T-1, gs))), int(max(0, min(T-1, ge)))

	src_idx = np.asarray(src_idx)
	inside = np.where((src_idx >= gs) & (src_idx <= ge))[0]

	# The whole gesture was dropped or shifted out of view: fall back to the surviving
	# frame closest to it, so the target stays on a real frame instead of a zero-filled one
	if len(inside) == 0:
		valid = np.where(src_idx >= 0)[0]
		if len(valid) == 0:
			return 0, 0
		nearest = valid[np.argmin(np.abs(src_idx[valid] - (gs + ge) / 2.0))]
		return int(nearest), int(nearest)

	return int(inside[0]), int(inside[-1])


# ---------------------------
# Temporal IoU between two segments
# ---------------------------
def segment_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
	"""
	Temporal IoU between a predicted and a ground-truth (start, end) segment.
	"""
	s1, e1 = pred
	s2, e2 = gt

	inter = max(0.0, min(e1, e2) - max(s1, s2))
	union = max(e1, e2) - min(s1, s2)
	if union <= 0:
		return 0.0

	return inter / union


# ---------------------------
# Merge segments separated by short gaps
# ---------------------------
def merge_segments(segments, merge_gap_frames=1):
	"""
	Merge consecutive (start_idx, end_idx) pairs separated by a gap of at most
	merge_gap_frames, so that a gesture broken into fragments by a few low-probability
	frames is recovered as one segment. Segments are assumed sorted by start index.
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


# ---------------------------
# Decode per-frame probabilities into a single gesture segment
# ---------------------------
def probs_to_single_segment(probs: np.ndarray, fps: int = 25, thr: float = 0.5,
							min_duration_s: float = 0.05, merge_gap_s: float = 0.04):
	"""
	Decode a per-frame actionness probability sequence into one gesture segment:
	  1) threshold the probabilities to get contiguous islands of gesture frames,
	  2) merge islands separated by a short gap,
	  3) drop islands that are too short to be a gesture,
	  4) return the island with the highest summed probability.
	Each clip contains a single gesture, hence the single-segment output. If nothing
	survives the thresholding, fall back to a one-frame window around the peak.
	Returns (start_frame, end_frame) as inclusive frame indices.
	"""
	T = len(probs)
	mask = probs >= thr

	# Collect contiguous runs of above-threshold frames
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

	merge_gap_frames = max(1, int(round(merge_gap_s * fps)))
	segments = merge_segments(segments, merge_gap_frames=merge_gap_frames)

	min_d_frames = max(1, int(round(min_duration_s * fps)))
	segments = [s for s in segments if (s[1] - s[0] + 1) >= min_d_frames]

	# Nothing crossed the threshold: fall back to a narrow window around the peak
	if len(segments) == 0:
		peak = int(np.argmax(probs))
		return max(0, peak - 1), min(T - 1, peak + 1)

	# Score by summed probability, which favours segments that are both long and confident
	seg_scores = [probs[s_idx:(e_idx+1)].sum() for (s_idx, e_idx) in segments]
	best_s, best_e = segments[int(np.argmax(seg_scores))]

	return best_s, best_e


# ---------------------------
# Training loop
# ---------------------------
def train(model, train_data_loader, test_data_loader, optimizer, checkpoint_dir=None,
			nepochs=None, fps=25, thr=0.5, min_duration_s=0.05):
	"""
	Train the joint word-recognition and gesture-localization model.

	The two heads are optimized together: cross-entropy over the 155 gesture words for
	recognition, and per-frame binary cross-entropy for localization. 
	After every epoch the model is checkpointed and validated.
	"""
	global global_step, global_epoch

	while global_epoch < nepochs:

		print("Epoch: {}".format(global_epoch))
		model.train()

		running_loss = []
		running_loss_cls = []
		running_loss_loc = []
		running_accuracy = []
		running_miou = []

		prog_bar = tqdm(train_data_loader)

		for batch_sample in prog_bar:

			# collate_data returns 0 when every clip in the batch failed to load
			if batch_sample == 0:
				continue

			optimizer.zero_grad()

			visual_feats = batch_sample["visual_feats"].cuda()
			visual_mask = batch_sample["visual_mask"].cuda()
			words = batch_sample["word"]

			label = torch.tensor([word_label_dict[w] for w in words]).cuda()	# (B,)
			B, T = visual_feats.shape[0], visual_feats.shape[1]

			# Per-frame localization targets, aligned to the augmented features
			gt_boundaries = [get_gt_boundary(batch_sample, i, T) for i in range(B)]
			loc_targets = np.stack([boundary_to_act_labels(gt, T=T) for gt in gt_boundaries])
			loc_targets = torch.tensor(loc_targets, dtype=torch.float32).cuda()	# (B,T)

			with torch.amp.autocast('cuda'):

				out = model(visual_feats, visual_mask.unsqueeze(1))

				cls_logits = out['cls_logits']		# (B,C)
				loc_logits = out['loc_logits']		# (B,T)

				loss_classifier = F.cross_entropy(cls_logits, label)
				loss_localizer = F.binary_cross_entropy_with_logits(loc_logits, loc_targets)

				loss = loss_classifier + loss_localizer

			if torch.isnan(loss).any():
				print("Nan loss!")
				continue

			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()

			global_step += 1

			# Batch metrics: word accuracy and mean IoU of the decoded gesture boundary
			with torch.no_grad():
				pred_cls = torch.argmax(cls_logits, dim=1)
				accuracy = (pred_cls == label).sum().item() / label.size(0) * 100.0

				probs_batch = torch.sigmoid(loc_logits).float().cpu().numpy()	# (B,T)
				ious = []
				for i in range(B):
					pred_seg = probs_to_single_segment(probs_batch[i], fps=fps, thr=thr,
														min_duration_s=min_duration_s)
					ious.append(segment_iou(pred_seg, gt_boundaries[i]))

			running_loss.append(loss.item())
			running_loss_cls.append(loss_classifier.item())
			running_loss_loc.append(loss_localizer.item())
			running_accuracy.append(accuracy)
			running_miou.append(float(np.mean(ious)))

			prog_bar.set_description('Total Loss: {:.4f} | Classifier-loss: {:.4f}, Localization-loss: {:.4f} || Accuracy: {:.4f}, mIoU: {:.4f} '.format(
				sum(running_loss) / len(running_loss),
				sum(running_loss_cls) / len(running_loss_cls),
				sum(running_loss_loc) / len(running_loss_loc),
				sum(running_accuracy) / len(running_accuracy),
				sum(running_miou) / len(running_miou)
			))

		save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
		validate(test_data_loader, model, fps=fps, thr=thr, min_duration_s=min_duration_s)

		global_epoch += 1


# ---------------------------
# Validation on the held-out set
# ---------------------------
def validate(model_data_loader, model, thr: float = 0.5, tiou_thresholds: List[float] = None,
				fps: int = 25, min_duration_s: float = 0.05):
	"""
	Evaluate the model on the held-out validation set.

	Reports word classification accuracy, and mIoU of the predicted gesture boundary, 
	alongside the two validation losses.
	"""
	if tiou_thresholds is None:
		tiou_thresholds = [0.3, 0.5, 0.7]

	model.eval()

	all_ious = []
	total_correct = 0
	total_samples = 0
	sum_loss_cls = 0.0
	sum_loss_loc = 0.0

	for batch_sample in tqdm(model_data_loader, total=len(model_data_loader)):

		if batch_sample == 0:
			continue

		visual_feats = batch_sample["visual_feats"].cuda()
		visual_mask = batch_sample["visual_mask"].cuda()
		words = batch_sample["word"]

		label = torch.tensor([word_label_dict[w] for w in words]).cuda()	# (B,)
		B, T = visual_feats.shape[0], visual_feats.shape[1]

		gt_boundaries = [get_gt_boundary(batch_sample, i, T) for i in range(B)]
		loc_targets = np.stack([boundary_to_act_labels(gt, T=T) for gt in gt_boundaries])
		loc_targets = torch.tensor(loc_targets, dtype=torch.float32).cuda()	# (B,T)

		with torch.no_grad():
			out = model(visual_feats, visual_mask.unsqueeze(1))

			cls_logits = out['cls_logits']		# (B,C)
			loc_logits = out['loc_logits']		# (B,T)

			# Weighted by B so that a smaller trailing batch does not skew the mean
			sum_loss_cls += F.cross_entropy(cls_logits, label).item() * B
			sum_loss_loc += F.binary_cross_entropy_with_logits(loc_logits, loc_targets).item() * B

			total_correct += (torch.argmax(cls_logits, dim=1) == label).sum().item()
			total_samples += B

			probs = torch.sigmoid(loc_logits).float().cpu().numpy()	# (B,T)
			for i in range(B):
				pred_seg = probs_to_single_segment(probs[i], fps=fps, thr=thr,
													min_duration_s=min_duration_s)
				all_ious.append(segment_iou(pred_seg, gt_boundaries[i]))

	if total_samples == 0:
		print("No validation samples could be loaded, skipping evaluation")
		return {}

	summary = {
		'classifier_loss': sum_loss_cls / total_samples,
		'localization_loss': sum_loss_loc / total_samples,
		'classification_accuracy': total_correct / total_samples * 100.0,
		'mean_iou': float(np.mean(all_ious)),
		
	}

	print("EVAL summary -- clips: {}".format(total_samples))
	print("Eval losses -- classifier: {:.4f}, localization: {:.4f}".format(
			summary['classifier_loss'], summary['localization_loss']))
	print("Classification: Accuracy = {:.2f} %".format(summary['classification_accuracy']))
	print("Localiation: mIoU = {:.4f}".format(summary['mean_iou']))

	return summary


# ---------------------------
# Save a checkpoint
# ---------------------------
def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
	"""
	Save the model and optimizer state at the end of an epoch.
	"""
	checkpoint_path = join(checkpoint_dir, "checkpoint_step{:05d}.pth".format(epoch))
	torch.save({
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"global_step": step,
		"global_epoch": epoch,
	}, checkpoint_path)
	print("Saved checkpoint:", checkpoint_path)


# ---------------------------
# Load a checkpoint
# ---------------------------
def load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False):
	"""
	Restore a checkpoint and resume the step/epoch counters from it. The 'module.'
	prefix left by DataParallel training is stripped so that both single-GPU and
	multi-GPU checkpoints load into the same model.
	"""
	global global_step, global_epoch

	if use_cuda:
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

	new_s = {k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items()}
	model.load_state_dict(new_s)

	if not reset_optimizer and checkpoint["optimizer"] is not None:
		print("Load optimizer state from {}".format(checkpoint_path))
		optimizer.load_state_dict(checkpoint["optimizer"])

	global_step = checkpoint["global_step"] + 1
	global_epoch = checkpoint["global_epoch"] + 1

	print("Loaded checkpoint from: {}".format(checkpoint_path))

	return model


# ---------------------------
# Data, model and training setup
# ---------------------------
if __name__ == '__main__':

	os.makedirs(args.checkpoint_dir, exist_ok=True)

	# Dataset and dataloader setup
	df = read_data(args.train_csv)
	df_train, df_val = train_val_split(df, args.val_size, args.seed)

	print("Total train files: ", len(df_train))
	print("Total val files: ", len(df_val))
	print("Total classes: ", len(word_label_dict))

	train_dataset = DataGenerator_Gestures(df_train, args.feature_dir, apply_augmentations=args.apply_augmentations)
	val_dataset = DataGenerator_Gestures(df_val, args.feature_dir)

	train_data_loader = data_utils.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
		num_workers=args.num_workers, collate_fn=lambda x: collate_data(x))

	val_data_loader = data_utils.DataLoader(
		val_dataset, batch_size=args.batch_size,
		num_workers=args.num_workers, collate_fn=lambda x: collate_data(x))

	print("Total train batch: ", len(train_data_loader))

	# Initialize the model
	model = Word_Recognition_Localization(input_dim=768, num_classes=len(word_label_dict),
											N=6, h=12, d_model=768).cuda()
	print('Total trainable params {:.3f}M'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000))

	optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
							lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)

	# Resume from a checkpoint, or initialize the weights for training from scratch
	if args.checkpoint_path is not None:
		model = load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
	else:
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	torch.backends.cudnn.benchmark = True

	# Train, validate and save the trained model
	train(model, train_data_loader, val_data_loader, optimizer,
			checkpoint_dir=args.checkpoint_dir, nepochs=args.num_epochs, fps=args.fps)
