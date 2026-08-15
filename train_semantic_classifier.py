import os, argparse
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils import data as data_utils
from torch.utils.data import WeightedRandomSampler

from dataloader import *
from models.grw_models import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


parser = argparse.ArgumentParser(description='Code to train the semantic gesture classifier network')

parser.add_argument('--train_csv', type=str, required=True, help='Path of the train-data csv file')
parser.add_argument('--checkpoint_dir', type=str, required=True, help='Folder to save the trained checkpoints')
parser.add_argument('--feature_dir', type=str, required=True, help='Feature directory')

parser.add_argument('--checkpoint_path', type=str, default=None, help='Resume training from this checkpoint')
parser.add_argument('--num_epochs', type=int, default=20, required=False, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size to train the model')
parser.add_argument('--num_workers', type=int, default=8, required=False, help='Number of workers')
parser.add_argument('--lr', type=float, default=1e-4, required=False, help='Learning rate')
parser.add_argument('--val_size', type=int, default=500, required=False, help='Number of files held out from the train set for validation')
parser.add_argument('--seed', type=int, default=42, required=False, help='Random seed for the train/val split')
args = parser.parse_args()

global_step = 0
global_epoch = 0
scaler = torch.amp.GradScaler('cuda')
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))



# ---------------------------
# Read the annotation csv
# ---------------------------
def read_data(path):
	"""
	Read a GRW annotation csv into a dataframe.

	word_form is blank for the non-semantic clips, which otherwise leaves the column as
	a mix of strings and NaN. Those rows fall back to the target word itself, so that
	word_form is a plain string column throughout.
	"""
	df = pd.read_csv(path)
	print("Total files: ", len(df))

	return df


# ---------------------------
# Build the train / validation split
# ---------------------------
def train_val_split(df, val_size, seed, stratify_col='gesture_label'):
	"""
	Hold out a small validation set from the train split.

	The sample is stratified by the binary gesture label and takes at least one clip
	per class, so both the semantic and the non-semantic class are represented in the
	validation set. The excess is then trimmed from the largest class only, which keeps
	the rarer class intact. Returns (df_train, df_val) with the two sets disjoint.
	"""
	frac = min(1.0, val_size / len(df))
	df_val = df.groupby(stratify_col, group_keys=False).apply(
					lambda g: g.sample(n=max(1, int(round(len(g)*frac))), random_state=seed))

	while len(df_val) > val_size:
		counts = df_val[stratify_col].value_counts()
		# Every class is down to a single clip -- stop here rather than drop a class
		if counts.max() == 1:
			print("Val size raised to {} to keep all {} classes".format(len(df_val), len(counts)))
			break
		drop_idx = df_val[df_val[stratify_col] == counts.idxmax()].sample(n=1, random_state=seed).index
		df_val = df_val.drop(drop_idx)

	df_train = df.drop(df_val.index).reset_index(drop=True)
	df_val = df_val.reset_index(drop=True)

	return df_train, df_val


# ---------------------------
# Class-balanced sampling for the imbalanced train set
# ---------------------------
def build_sampler_and_pos_weight(df):
	"""
	Semantic gestures are far rarer than non-semantic ones, so the two classes are
	rebalanced in two complementary ways: a WeightedRandomSampler that draws each class
	equally often across an epoch, and a pos_weight for the loss that up-weights the
	positive class. Returns (sampler, pos_weight).
	"""
	class_counts = df["gesture_label"].astype(int).value_counts().sort_index()
	num_neg, num_pos = class_counts[0], class_counts[1]
	print("Train distribution:", class_counts.to_dict())

	# Weight every clip by the inverse frequency of its class
	class_weights = 1.0 / torch.tensor([num_neg, num_pos], dtype=torch.float)
	sample_weights = df["gesture_label"].astype(int).map({0: class_weights[0], 1: class_weights[1]}).astype(float).values

	sampler = WeightedRandomSampler(
		weights=torch.DoubleTensor(sample_weights),
		num_samples=len(sample_weights),
		replacement=True
	)

	pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float).cuda()

	return sampler, pos_weight


# ---------------------------
# Target word interval for each clip of a batch
# ---------------------------
def get_target_intervals(batch_sample):
	"""
	Frame interval of the spoken target word within each clip, as a (B,2) tensor. The
	model cross-attends from these frames into the full clip, so the classification
	decision is anchored on the gesture that co-occurs with the target word.
	"""
	intervals = [[info.target_start, info.target_end] for info in batch_sample["info"]]

	return torch.tensor(intervals)


# ---------------------------
# Training loop
# ---------------------------
def train(model, train_data_loader, val_data_loader, optimizer, criterion,
			checkpoint_dir=None, nepochs=None):
	"""
	Train the binary semantic gesture classifier.

	Each clip is labelled according to whether the gesture co-occurring with the target
	word is semantically meaningful, and the model is optimized with a class-weighted
	binary cross-entropy. After every epoch the model is checkpointed and validated.
	"""
	global global_step, global_epoch

	while global_epoch < nepochs:

		print("Epoch: {}".format(global_epoch))
		model.train()

		running_loss = []
		running_accuracy = []

		prog_bar = tqdm(train_data_loader)

		for batch_sample in prog_bar:

			# collate_data returns 0 when every clip in the batch failed to load
			if batch_sample == 0:
				continue

			optimizer.zero_grad()

			visual_feats = batch_sample["visual_feats"].cuda()
			visual_mask = batch_sample["visual_mask"].cuda()

			
			label = torch.tensor([float(info.gesture_label) for info in batch_sample["info"]]).float().unsqueeze(1).cuda()
			target_intervals = get_target_intervals(batch_sample)

			try:
				with torch.amp.autocast('cuda'):

					# visual_emb: (B,1) raw logit for "the gesture is semantic"
					visual_emb, _ = model(visual_feats, x_mask=visual_mask.unsqueeze(1),
											target_intervals=target_intervals)

					loss = criterion(visual_emb, label)

			except RuntimeError as e:
				if 'CUDA out of memory' in str(e):
					print('[Runner] - CUDA out of memory at step {}'.format(global_step))
					torch.cuda.empty_cache()
					optimizer.zero_grad()
					continue
				raise

			if torch.isnan(loss).any():
				print("Nan loss!")
				continue

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			global_step += 1

			with torch.no_grad():
				pred = (torch.sigmoid(visual_emb) > 0.5).long()
				accuracy = (pred == label).sum().item() / label.size(0) * 100.0

			running_loss.append(loss.item())
			running_accuracy.append(accuracy)

			prog_bar.set_description('Loss: {:.4f} | Accuracy: {:.4f} '.format(
				sum(running_loss) / len(running_loss),
				sum(running_accuracy) / len(running_accuracy)))

		save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
		validate(val_data_loader, model, criterion)

		global_epoch += 1


# ---------------------------
# Validation on the held-out set
# ---------------------------
def validate(val_data_loader, model, criterion):
	"""
	Evaluate the classifier on the held-out validation set, reporting the mean loss and
	the binary classification accuracy.
	"""
	print('Evaluating for {} steps'.format(len(val_data_loader)))

	model.eval()

	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	for batch_sample in tqdm(val_data_loader, total=len(val_data_loader)):

		if batch_sample == 0:
			continue

		visual_feats = batch_sample["visual_feats"].cuda()
		visual_mask = batch_sample["visual_mask"].cuda()

		
		label = torch.tensor([float(info.gesture_label) for info in batch_sample["info"]]).float().unsqueeze(1).cuda()
		target_intervals = get_target_intervals(batch_sample)
		B = visual_feats.shape[0]

		with torch.no_grad():
			with torch.amp.autocast('cuda'):
				visual_emb, _ = model(visual_feats, x_mask=visual_mask.unsqueeze(1),
										target_intervals=target_intervals)
				loss = criterion(visual_emb, label)

			if torch.isnan(loss).any():
				continue

			# Weighted by B so that a smaller trailing batch does not skew the mean
			total_loss += loss.item() * B
			total_correct += ((torch.sigmoid(visual_emb) > 0.5).long() == label).sum().item()
			total_samples += B

	if total_samples == 0:
		print("No validation samples could be loaded, skipping evaluation")
		return

	print('Validation loss: {:.4f} | Accuracy: {:.4f}'.format(
			total_loss / total_samples, total_correct / total_samples * 100.0))

	return


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

	train_dataset = DataGenerator_Gestures(df_train, args.feature_dir)
	val_dataset = DataGenerator_Gestures(df_val, args.feature_dir)

	# Rebalance the two classes, both in the sampler and in the loss
	sampler, pos_weight = build_sampler_and_pos_weight(df_train)

	train_data_loader = data_utils.DataLoader(
		train_dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True,
		num_workers=args.num_workers, collate_fn=lambda x: collate_data(x))

	val_data_loader = data_utils.DataLoader(
		val_dataset, batch_size=args.batch_size,
		num_workers=args.num_workers, collate_fn=lambda x: collate_data(x))

	print("Total train batch: ", len(train_data_loader))

	# Initialize the model
	model = Semantic_Classifier(input_dim=768, num_classes=1, use_layer_weights=True).cuda()
	print('Total trainable params {:.3f}M'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000))

	optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
							lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)

	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()

	# Resume from a checkpoint, or initialize the weights for training from scratch
	if args.checkpoint_path is not None:
		model = load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)
	else:
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	torch.backends.cudnn.benchmark = True

	train(model, train_data_loader, val_data_loader, optimizer, criterion,
			checkpoint_dir=args.checkpoint_dir, nepochs=args.num_epochs)
