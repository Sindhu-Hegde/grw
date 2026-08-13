"""
Evaluates a binary gesture-presence (semantic) classifier: reports overall
accuracy/precision/recall, plus accuracy restricted to the subset of
predictions the model is highly confident about.
"""
import argparse
import torch
from torch.utils import data as data_utils
import numpy as np
from dataloader import *
from models.grw_models import *
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


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


def compute_metrics(predictions, gts, confidences=None, infos=None, confidence_threshold=0.9):
	"""
	Reports overall binary accuracy/precision/recall, and, if per-sample
	`confidences` (sigmoid outputs) and `infos` (metadata rows) are given,
	also reports accuracy restricted to the subset of samples whose
	confidence exceeds `confidence_threshold`.
	"""
	accuracy = accuracy_score(gts, predictions)
	precision = precision_score(gts, predictions, average='binary', pos_label=1)
	recall = recall_score(gts, predictions, average='binary', pos_label=1)

	print("--- Semantic Gesture Classification ---")
	print(f"Overall Accuracy: {accuracy*100:.2f} %")
	print(f"Precision: {precision*100:.2f} %")
	print(f"Recall: {recall*100:.2f} %")

	if confidences is None or infos is None or len(confidences) == 0:
		return

	if len(confidences) != len(infos):
		print(f"\nWarning: Mismatch between confidences ({len(confidences)}) and infos ({len(infos)}) lengths")
		return

	confidences = np.array(confidences)
	high_conf_indices = np.where(confidences > confidence_threshold)[0]

	print(f"\n--- High-Confidence Subset (confidence > {confidence_threshold}) ---")
	if len(high_conf_indices) == 0:
		print(f"No rows found with confidence > {confidence_threshold}")
		return

	subset_df = pd.DataFrame([infos[i] for i in high_conf_indices])
	high_conf_acc = (subset_df["gesture_label"] == 1).sum() / len(subset_df)
	print(f"High-confidence accuracy: {high_conf_acc*100:.2f} %")


def evaluate(model, test_data_loader):

	print('Evaluating for {} steps'.format(len(test_data_loader)))

	predictions, gts = [], []
	confidences = []
	infos = []
	for batch_sample in tqdm(test_data_loader):

		try:
			if batch_sample == 0:
				continue

			visual_feats = batch_sample["visual_feats"].cuda()
			visual_mask = batch_sample["visual_mask"].cuda()
			batch_infos = batch_sample["info"]

			labels = []
			target_intervals = []
			for info in batch_infos:
				labels.append(float(info.gesture_label))

				start, end = info.target_start, info.target_end
				target_intervals.append([start, end])

			target_intervals = torch.tensor(target_intervals)  # (B, 2)

			with torch.amp.autocast('cuda'):
				with torch.no_grad():
					# visual_emb: (B, 1) raw logit for "gesture present"
					visual_emb, visual_emb_temporal = model(
						visual_feats, x_mask=visual_mask.unsqueeze(1), target_intervals=target_intervals)

			batch_confidences = torch.sigmoid(visual_emb).cpu().numpy()
			pred = (torch.sigmoid(visual_emb) > 0.5).long().detach()

			confidences.extend(batch_confidences)
			infos.extend(batch_infos)
			predictions.extend(pred.squeeze(1).cpu().numpy().tolist())  # makes it 1D
			gts.extend(labels)

		except RuntimeError as e:
			if 'CUDA out of memory' in str(e):
				print(f'[Runner] - CUDA out of memory')
				torch.cuda.empty_cache()
				continue
			else:
				print(e)
				print("Batch sample: ", batch_sample["info"][0].unique_fname_test)
				continue

	predictions = np.array(predictions)
	gts = np.array(gts)

	compute_metrics(predictions, gts, confidences=confidences, infos=infos)

	return


if __name__ == "__main__":

	# Dataset and Dataloader setup
	df_test = read_data(args.test_csv)
	all_words = list(word_label_dict.keys())

	batch_size = args.batch_size
	checkpoint_path = args.checkpoint_path

	# Dataloader
	test_dataset = DataGenerator_Train_WordClassifier(df_test, args.feature_dir)
	test_data_loader = data_utils.DataLoader(
		test_dataset, batch_size=batch_size, num_workers=4, collate_fn=lambda x: collate_data_wordclassifier(x))

	# Load the model
	model = Semantic_Classifier(input_dim=768, num_classes=1, use_layer_weights=True).cuda()
	model = load_checkpoint(model, checkpoint_path)

	# Evaluate and obtain the metrics
	evaluate(model, test_data_loader)
