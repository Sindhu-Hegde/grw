"""
Single-video SHuBERT feature extraction pipeline.

Given an input video of a signer/speaker, this script:
  1. Detects and crops the person (YOLO + optical-flow-guided tracking, in case
     more than one person appears in frame).
  2. Extracts per-frame hand and body-pose landmarks (MediaPipe Holistic +
     HandLandmarker).
  3. Crops out left/right hand videos from those landmarks.
  4. Extracts DINOv2 embeddings per hand crop, and normalized pose keypoints
     for the body.
  5. Feeds hands + body pose into SHuBERT to extract per-layer embeddings.

Each stage's output (pose/hand DINOv2 features, final SHuBERT features) is
cached to <output_dir>/<video_name>/ as .npy files.
"""
import os, sys
import warnings
warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import subprocess
import json
import argparse

from PIL import Image
import numpy as np

from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.fairseq.examples.shubert.models.shubert import SHubertModel, SHubertConfig

import torch
import torch.nn as nn
from torchvision import transforms
from decord import VideoReader


parser = argparse.ArgumentParser()
parser.add_argument('--input_video', type=str, required=True, help='Path to the input video')
parser.add_argument('--yolo_ckpt', type=str, required=True, help='Path to the yolo model')
parser.add_argument('--hand_kp_ckpt', type=str, required=True, help='Path to the hand keypoints model')
parser.add_argument('--dino_ckpt', type=str, required=True, help='Path to the DINO hand model')
parser.add_argument('--shubert_ckpt', type=str, required=True, help='Path to the shubert model')
parser.add_argument('--output_dir', type=str, required=False, default='results', help='Path to the output directory to save the features')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_dino_finetuned_downloaded(dino_path):
	"""Load DINOv2-ViT-S/14 and swap in hand-finetuned weights."""
	model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)

	pretrained = torch.load(dino_path, map_location=device)

	# Finetuned checkpoint only has the teacher backbone (student/DINO-head are
	# training-only), so keep just those weights and drop the 'backbone.' prefix.
	new_state_dict = {}
	for key, value in pretrained['teacher'].items():
		if 'dino_head' not in key:
			new_key = key.replace('backbone.', '')
			new_state_dict[new_key] = value

	# Finetuning used a different input resolution than the pretrained hub
	# checkpoint, so the positional embedding shape must be reset to match
	# before loading the state dict.
	pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
	model.pos_embed = pos_embed

	model.load_state_dict(new_state_dict, strict=True)
	model.to(device)

	return model


def load_model_shubert(shubert_path):
	"""Load the pretrained SHuBERT model used for feature extraction."""
	cfg = SHubertConfig()
	model = SHubertModel(cfg)

	checkpoint = torch.load(shubert_path)
	state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

	# strict=False: the checkpoint may include training-only submodules
	# (e.g. quantizers/predictors) that aren't part of feature extraction.
	model.load_state_dict(state_dict, strict=False)
	model.eval()
	model.cuda()

	return model


def get_optical_flow(images):
	"""Per-frame optical-flow magnitude map, used to score how much motion is happening in each region."""
	prv_gray = None
	motion_mags = []
	for frame in images:
		cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray_size = (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))
		cur_gray = cv2.resize(cur_gray, gray_size)
		if prv_gray is not None:
			flow = cv2.calcOpticalFlowFarneback(prv_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
			mag = (255.0*(mag-mag.min())/max(float(mag.max()-mag.min()), 1)).astype(np.uint8)
			mag = cv2.resize(mag, (frame.shape[1], frame.shape[0]))
		else:
			mag = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
		prv_gray = cur_gray
		motion_mags.append(mag)
	return motion_mags


def get_iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def find_target_bbox(bbox_arr, opts, iou_thr=0.5, len_ratio_thr=0.5):
	"""
	When YOLO detects more than one person per frame, link detections across
	frames into per-person tracks ("tubes") via greedy IoU matching, then pick
	the track that (a) is present for a large-enough fraction of the clip and
	(b) overlaps the most with optical-flow motion — i.e. the person who is
	actually moving/gesturing, as opposed to a bystander.
	"""
	tubes = []
	num_rest = sum([len(x) for x in bbox_arr])
	while num_rest > 0:
		for i, bboxes in enumerate(bbox_arr):
			if len(bboxes) > 0:
				anchor = [i, bbox_arr[i].pop()]
				break
		tube = [anchor]
		for i in range(len(bbox_arr)):
			bboxes = bbox_arr[i]
			if anchor[0] == i or len(bboxes) == 0:
				continue
			ious = np.array([get_iou(anchor[1], bbox) for bbox in bboxes])
			j = ious.argmax()
			if ious[j] > iou_thr:
				target_bbox = bboxes.pop(j)
				tube.append([i, target_bbox])
		tubes.append(tube)
		num_rest = sum([len(x) for x in bbox_arr])

	max_val, max_tube = -1, None
	for itube, tube in enumerate(tubes):
		mean_val = 0
		for iframe, bbox in tube:
			x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
			mean_val += opts[iframe][max(y0, 0): y1, max(x0, 0): x1].mean()
		mean_val /= len(tube)

		if len(tube)/len(opts) > len_ratio_thr:
			if mean_val > max_val:
				max_val, max_tube = mean_val, tube

	if max_tube is not None:
		target_bbox = np.array([bbox[1] for bbox in max_tube]).mean(axis=0).tolist()
	else:
		target_bbox = None
	return target_bbox, tubes


def crop_clip(video_path, yolo_model, output_video_path, out_folder):
	"""Detect the signer with YOLO in every frame and crop the clip to their (tracked) bounding box."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Error: Could not open video file {video_path}")
		return None

	fps = cap.get(cv2.CAP_PROP_FPS)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	new_width, new_height = width, height

	fourcc = cv2.VideoWriter_fourcc(*'MP4V')

	# Expand the detected person bbox a bit, more so to the sides since
	# gesturing hands often extend outside a tight person bounding box.
	up_exp, down_exp, left_exp, right_exp = 0.01, 0.01, 0.3, 0.3

	bboxes = []
	frames = []
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = yolo_model(frame_rgb, verbose=False)

		# Filter for person class (class 0 in COCO dataset)
		person_bboxes = []
		for r in results:
			for box in r.boxes:
				if box.cls == 0:
					x1, y1, x2, y2 = box.xyxy[0].tolist()
					person_bboxes.append([x1, y1, x2, y2])

		bboxes.append(person_bboxes)
		frames.append(frame)

	cap.release()

	if len(frames) == 0:
		return None

	for i in range(len(bboxes)):
		for j in range(len(bboxes[i])):
			x0, y0, x1, y1 = bboxes[i][j]
			w, h = x1-x0+1, y1-y0+1
			x0, y0, x1, y1 = x0-w*left_exp, y0-h*up_exp, x1+w*right_exp, y1+h*down_exp
			bboxes[i][j] = [x0, y0, x1, y1]

	if max([len(x) for x in bboxes]) == 1:
		# Only ever one person detected: no need for cross-frame tracking.
		bboxes = list(filter(lambda x: len(x) == 1, bboxes))
		bbox = np.array(bboxes).mean(axis=0)[0].tolist()
		tubes = []
	else:
		opts = get_optical_flow(frames)
		bbox, tubes = find_target_bbox(bboxes, opts)

	if bbox is None and len(tubes) > 0:
		# find_target_bbox found no track passing len_ratio_thr; fall back to
		# the track with the largest total bbox area instead.
		if max([len(x) for x in tubes]) > 0:
			total_sizes = []
			for tube in tubes:
				total_size = sum([(bbox[3]-bbox[0])*(bbox[2]-bbox[1]) for _, bbox in tube])
				total_sizes.append(total_size)
			idx = np.array(total_sizes).argmax()
			bbox = np.array([x for _, x in tubes[idx]]).mean(axis=0).tolist()

	if bbox is not None:
		x_0 = int(max(bbox[0], 0))
		y_0 = int(max(bbox[1], 0))
		x_1 = int(min(bbox[2], new_width))
		y_1 = int(min(bbox[3], new_height))

		for i in range(len(frames)):
			frames[i] = frames[i][y_0:y_1, x_0:x_1]

	clip_width, clip_height = frames[0].shape[1], frames[0].shape[0]
	resized_clip = cv2.VideoWriter(
		output_video_path,
		fourcc, fps, (clip_width, clip_height),
	)

	if not resized_clip.isOpened():
		print(f"Error: Could not create video writer for {output_video_path}")
		return None

	for frame in frames:
		resized_clip.write(frame)
	resized_clip.release()

	return frames


def detect_holistic(image):
	"""Run MediaPipe Holistic (pose) and HandLandmarker on a single RGB frame."""
	results = mp_holistic.process(image)

	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
	hand_prediction = hand_detector.detect(mp_image)

	bounding_boxes = {}
	landmarks_data = {}

	if hand_prediction.hand_landmarks:
		bounding_boxes['#hands'] = len(hand_prediction.hand_landmarks)
		landmarks_data['hand_landmarks'] = []
		for hand in hand_prediction.hand_landmarks:
			landmarks_hand = []
			for landmark in hand:
				landmarks_hand.append([landmark.x, landmark.y, landmark.z])
			landmarks_data['hand_landmarks'].append(landmarks_hand)
	else:
		bounding_boxes['#hands'] = 0
		landmarks_data['hand_landmarks'] = None

	if results.pose_landmarks:
		bounding_boxes['#pose'] = 1
		landmarks_data['pose_landmarks'] = []
		landmarks_data['pose_landmarks'].append([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark])
	else:
		bounding_boxes['#pose'] = 0
		landmarks_data['pose_landmarks'] = None

	return bounding_boxes, landmarks_data


def resize_frame(frame, frame_size):
	if frame is not None and frame.size > 0:
		return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
	else:
		return None


def crop_frame(image, bounding_box):
	x, y, w, h = bounding_box
	cropped_frame = image[y:y + h, x:x + w]
	return cropped_frame


def get_bounding_box(landmarks, image_shape, scale_factor=1.2):
	"""Square bounding box around a set of normalized landmarks, padded by scale_factor."""
	ih, iw, _ = image_shape
	landmarks_px = np.array([(int(l[0] * iw), int(l[1] * ih)) for l in landmarks])
	center_x, center_y = np.mean(landmarks_px, axis=0, dtype=int)
	xb, yb, wb, hb = cv2.boundingRect(landmarks_px)
	box_size = max(wb, hb)
	half_size = box_size // 2
	x = center_x - half_size
	y = center_y - half_size
	w = box_size
	h = box_size

	w_padding = int((scale_factor - 1) * w / 2)
	h_padding = int((scale_factor - 1) * h / 2)
	x -= w_padding
	y -= h_padding
	w += 2 * w_padding
	h += 2 * h_padding

	return x, y, w, h


def adjust_bounding_box(bounding_box, image_shape):
	x, y, w, h = bounding_box
	ih, iw, _ = image_shape

	# Adjust x-coordinate if the bounding box extends beyond the image's right edge
	if x + w > iw:
		x = iw - w

	# Adjust y-coordinate if the bounding box extends beyond the image's bottom edge
	if y + h > ih:
		y = ih - h

	# Ensure bounding box's x and y coordinates are not negative
	x = max(x, 0)
	y = max(y, 0)

	return x, y, w, h


def select_hands(pose_landmarks, hand_landmarks, image_shape):
	"""
	MediaPipe's HandLandmarker doesn't guarantee which detection is the left
	vs. right hand, so match each detected hand's wrist landmark to the
	nearest pose wrist landmark to assign handedness.
	"""
	if hand_landmarks is None:
		return None, None

	left_wrist_from_pose = pose_landmarks[15]
	right_wrist_from_pose = pose_landmarks[16]

	wrist_from_hand = []
	for i in range(0, len(hand_landmarks)):
		wrist_from_hand.append(hand_landmarks[i][0])

	# Euclidean distance between the two points using only the first 2 coordinates.
	if right_wrist_from_pose is not None:
		right_hand_landmarks = hand_landmarks[0]
		minimum_distance = 100
		for i in range(0, len(hand_landmarks)):
			distance = np.linalg.norm(np.array(right_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
			if distance < minimum_distance:
				minimum_distance = distance
				right_hand_landmarks = hand_landmarks[i]

		if minimum_distance >= 0.1:
			right_hand_landmarks = None
	else:
		right_hand_landmarks = None

	if left_wrist_from_pose is not None:
		left_hand_landmarks = hand_landmarks[0]
		minimum_distance = 100
		for i in range(0, len(hand_landmarks)):
			distance = np.linalg.norm(np.array(left_wrist_from_pose[0:2]) - np.array(wrist_from_hand[i][0:2]))
			if distance < minimum_distance:
				minimum_distance = distance
				left_hand_landmarks = hand_landmarks[i]
		if minimum_distance >= 0.1:
			left_hand_landmarks = None
	else:
		left_hand_landmarks = None

	return left_hand_landmarks, right_hand_landmarks


def get_hand_crops(video, result_dict, output_video_path_hand1, output_video_path_hand2):
	"""
	Crop each frame around the detected left/right hand landmarks into two
	224x224 videos. When a hand isn't detected in a frame, reuse the previous
	frame's crop (or a blank frame if none exists yet) so both output videos
	stay in sync, frame-for-frame, with the input video.
	"""
	try:
		left_hand_crops = []
		right_hand_crops = []

		prev_hand1_frame = None
		prev_hand2_frame = None
		prev_result_dict = None

		fourcc_hand1 = cv2.VideoWriter_fourcc(*'mp4v')
		out_hand1 = cv2.VideoWriter(output_video_path_hand1, fourcc_hand1, 25, (224, 224))

		fourcc_hand2 = cv2.VideoWriter_fourcc(*'mp4v')
		out_hand2 = cv2.VideoWriter(output_video_path_hand2, fourcc_hand2, 25, (224, 224))

		for i in range(len(video)):
			frame = video[i].asnumpy()
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			if result_dict[str(i)] is None:  # no pose_landmarks detected
				if prev_result_dict is not None:  # use the previous pose_landmarks
					result_dict[str(i)] = prev_result_dict
				else:  # no pose_landmarks was ever detected yet for this clip
					continue
			else:  # store pose_landmarks detected as last known pose_landmarks
				prev_result_dict = result_dict[str(i)]

			if result_dict[str(i)]['pose_landmarks'] is None:
				if prev_hand1_frame is not None:
					out_hand1.write(prev_hand1_frame)
					left_hand_crops.append(prev_hand1_frame)
				else:
					hand1_frame = np.zeros((224, 224, 3), dtype=np.uint8)
					out_hand1.write(hand1_frame)
					left_hand_crops.append(hand1_frame)
				if prev_hand2_frame is not None:
					out_hand2.write(prev_hand2_frame)
					right_hand_crops.append(prev_hand2_frame)
				else:
					hand2_frame = np.zeros((224, 224, 3), dtype=np.uint8)
					out_hand2.write(hand2_frame)
					right_hand_crops.append(hand2_frame)
				continue

			result_dict[str(i)]['left_hand_landmarks'], result_dict[str(i)]['right_hand_landmarks'] = select_hands(result_dict[str(i)]['pose_landmarks'][0], result_dict[str(i)]['hand_landmarks'], frame_rgb.shape)

			if result_dict[str(i)]['left_hand_landmarks'] is not None:
				hand1_box = get_bounding_box(result_dict[str(i)]['left_hand_landmarks'], frame_rgb.shape, scale_factor=1.5)
				hand1_box = adjust_bounding_box(hand1_box, frame_rgb.shape)
				hand1_frame = crop_frame(frame_rgb, hand1_box)
				hand1_frame = resize_frame(hand1_frame, (224, 224))
				out_hand1.write(hand1_frame)
				left_hand_crops.append(hand1_frame)
				prev_hand1_frame = hand1_frame
			elif prev_hand1_frame is not None:
				out_hand1.write(prev_hand1_frame)
				left_hand_crops.append(prev_hand1_frame)
			else:
				hand1_frame = np.zeros((224, 224, 3), dtype=np.uint8)
				out_hand1.write(hand1_frame)
				left_hand_crops.append(hand1_frame)

			if result_dict[str(i)]['right_hand_landmarks'] is not None:
				hand2_box = get_bounding_box(result_dict[str(i)]['right_hand_landmarks'], frame_rgb.shape, scale_factor=1.5)
				hand2_box = adjust_bounding_box(hand2_box, frame_rgb.shape)
				hand2_frame = crop_frame(frame_rgb, hand2_box)
				hand2_frame = resize_frame(hand2_frame, (224, 224))
				out_hand2.write(hand2_frame)
				right_hand_crops.append(hand2_frame)
				prev_hand2_frame = hand2_frame
			elif prev_hand2_frame is not None:
				out_hand2.write(prev_hand2_frame)
				right_hand_crops.append(prev_hand2_frame)
			else:
				hand2_frame = np.zeros((224, 224, 3), dtype=np.uint8)
				out_hand2.write(hand2_frame)
				right_hand_crops.append(hand2_frame)

		out_hand1.release()
		out_hand2.release()
		del out_hand1
		del out_hand2
		return left_hand_crops, right_hand_crops

	except Exception as e:
		print(f"Error in get_hand_crops: {e}")
		return None, None


def normalize_pose_keypoints(pose_landmarks):
	"""
	Map pose keypoints into a canonical "signing space" — a head-unit-scaled
	box anchored on the shoulders/eyes/nose — so keypoints are invariant to
	the signer's position and scale within the frame.
	"""
	left_shoulder = np.array(pose_landmarks[11][:2])
	right_shoulder = np.array(pose_landmarks[12][:2])
	left_eye = np.array(pose_landmarks[2][:2])
	nose = np.array(pose_landmarks[0][:2])

	head_unit = np.linalg.norm(right_shoulder - left_shoulder) / 2

	signing_space_width = 6 * head_unit
	signing_space_height = 7 * head_unit

	signing_space_top = left_eye[1] - 0.5 * head_unit
	signing_space_bottom = signing_space_top + signing_space_height
	signing_space_left = nose[0] - signing_space_width / 2
	signing_space_right = signing_space_left + signing_space_width

	translation_matrix = np.array([[1, 0, -signing_space_left],
								   [0, 1, -signing_space_top],
								   [0, 0, 1]])
	scale_matrix = np.array([[1 / signing_space_width, 0, 0],
							 [0, 1 / signing_space_height, 0],
							 [0, 0, 1]])
	shift_matrix = np.array([[1, 0, -0.5],
							 [0, 1, -0.5],
							 [0, 0, 1]])
	transformation_matrix = shift_matrix @ scale_matrix @ translation_matrix

	normalized_keypoints = []
	for landmark in pose_landmarks:
		keypoint = np.array([landmark[0], landmark[1], 1])
		normalized_keypoint = transformation_matrix @ keypoint
		normalized_keypoints.append(normalized_keypoint[:2])

	return normalized_keypoints


def get_body_embeddings(result_dict, out_feat_path_pose):
	"""Extract, normalize, and save the per-frame upper-body pose keypoints used as SHuBERT's body-posture input."""
	try:
		prev_pose = None
		video_pose_landmarks = []
		for i in range(len(result_dict)):
			if result_dict[str(i)] is None:
				if prev_pose is not None:
					frame_pose_landmarks = prev_pose
				else:
					frame_pose_landmarks = np.full((7, 2), -9999)
			elif result_dict[str(i)]['pose_landmarks'] is not None:
				frame_pose_landmarks = result_dict[str(i)]['pose_landmarks'][0]
				# 0: nose, 11/12: left/right shoulder, 13/14: left/right elbow, 15/16: left/right wrist
				indices = [0, 11, 12, 13, 14, 15, 16]
				frame_pose_landmarks = normalize_pose_keypoints(frame_pose_landmarks[0:25])
				frame_pose_landmarks = [frame_pose_landmarks[j] for j in indices]
				frame_pose_landmarks = np.array(frame_pose_landmarks).flatten()
				prev_pose = frame_pose_landmarks
			elif prev_pose is not None:
				frame_pose_landmarks = prev_pose
			else:
				frame_pose_landmarks = np.full((7, 2), -9999).flatten()
			video_pose_landmarks.append(frame_pose_landmarks)

		video_pose_landmarks = np.array(video_pose_landmarks)
		np.save(out_feat_path_pose, video_pose_landmarks)
		return video_pose_landmarks

	except Exception as e:
		print(f"Error in get_body_embeddings: {e}")
		return None


def preprocess_frame(frame):
	"""Preprocess a single frame for DINOv2 (ImageNet normalization)."""
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	image = Image.fromarray(frame)
	return transform(image)[:3]  # Ensure only RGB channels are considered


def preprocess_video(video_file, yolo_model, out_folder, out_feat_path_pose):
	"""Crop the signer, extract per-frame landmarks, and crop out per-hand videos."""
	fname = os.path.splitext(os.path.basename(video_file))[0]

	# Get person crops
	video_path = os.path.join(out_folder, fname + "_cropped.mp4")
	cropped_frames = crop_clip(video_file, yolo_model, video_path, out_folder)
	if cropped_frames is None:
		print(f"No cropped frames found for {video_file}")
		return None
	print("Total cropped frames: ", np.array(cropped_frames).shape)

	# Keypoint extraction from the cropped frames
	landmark_json_path = os.path.join(out_folder, fname + "_pose.json")
	stats_json_path = os.path.join(out_folder, fname + "_stats.json")

	video = VideoReader(video_path)

	result_dict = {}
	stats = {}
	err_count = 0
	for i in range(len(video)):
		try:
			frame_rgb = video[i].asnumpy()
			bounding_boxes, result_dict[str(i)] = detect_holistic(frame_rgb)
		except Exception as e:
			print(f"Error: {e}")
			err_count += 1
			result_dict[str(i)] = None
			continue

		stats[str(i)] = bounding_boxes

	if err_count/len(video) > 0.7:
		print(f"Keypoint detection error rate is too high: {err_count/len(video)}")
		return None

	with open(landmark_json_path, 'w') as rd:
		json.dump(result_dict, rd)
	with open(stats_json_path, 'w') as st:
		json.dump(stats, st)
	print("Keypoints extracted ({}) and saved ({})".format(len(result_dict), landmark_json_path))

	# Crop left and right hands based on hand bbox
	left_hand_video_path = os.path.join(out_folder, fname + "_hand1.mp4")
	right_hand_video_path = os.path.join(out_folder, fname + "_hand2.mp4")
	left_hand_crops, right_hand_crops = get_hand_crops(video, result_dict, left_hand_video_path, right_hand_video_path)
	if left_hand_crops is None or right_hand_crops is None:
		return None

	# Extract and save body pose features
	pose_embeddings = get_body_embeddings(result_dict, out_feat_path_pose)
	if pose_embeddings is None:
		return None
	print("Body pose embeddings extracted: ", pose_embeddings.shape)

	return left_hand_video_path, right_hand_video_path, pose_embeddings


def get_dinov2_embeddings(video_path, out_feat_path, model, batch_size=128):
	"""Run the fine-tuned DINOv2 model over every frame of a hand-crop video."""
	try:
		vr = VideoReader(video_path, width=224, height=224)
		total_frames = len(vr)

		all_embeddings = []
		for idx in range(0, total_frames, batch_size):
			batch_frames = vr.get_batch(range(idx, min(idx + batch_size, total_frames))).asnumpy()
			batch_tensors = torch.stack([preprocess_frame(frame) for frame in batch_frames]).cuda()

			with torch.no_grad():
				batch_embeddings = model(batch_tensors.to('cuda')).cpu().numpy()

			all_embeddings.append(batch_embeddings)

		embeddings = np.concatenate(all_embeddings, axis=0)
		np.save(out_feat_path, embeddings)

		return embeddings

	except Exception as e:
		print(f"Error in extracting DINO hand features: {e}")
		return None


def extract_dino_feats(model_hands, inp_hand1_video, inp_hand2_video, out_feat_path_hand1, out_feat_path_hand2):
	"""Extract and save DINOv2 embeddings for both hand-crop videos."""
	hand1_feats = get_dinov2_embeddings(inp_hand1_video, out_feat_path_hand1, model_hands)
	if hand1_feats is None:
		print(f"Error in extracting hand-1 features for {inp_hand1_video}")
	else:
		print("Hand-1 embeddings extracted: ", hand1_feats.shape)

	hand2_feats = get_dinov2_embeddings(inp_hand2_video, out_feat_path_hand2, model_hands)
	if hand2_feats is None:
		print(f"Error in extracting hand-2 features for {inp_hand2_video}")
	else:
		print("Hand-2 embeddings extracted: ", hand2_feats.shape)

	return hand1_feats, hand2_feats


def get_shubert_feats(model, hand1_feats, hand2_feats, body_feats):
	"""Run SHuBERT's feature extractor over hand + body-pose features and stack every layer's output."""
	hand1 = torch.from_numpy(hand1_feats).float().cuda()
	hand2 = torch.from_numpy(hand2_feats).float().cuda()
	body = torch.from_numpy(body_feats).float().cuda()
	# Face modality isn't extracted in this pipeline; SHuBERT still expects the input slot.
	face = torch.zeros_like(hand1)
	length = hand1.shape[0]

	# Dummy zero labels: only required because extract_features shares its
	# input format with the (label-conditioned) masked-pretraining code path.
	source = [{
		"face": face,
		"left_hand": hand1,
		"right_hand": hand2,
		"body_posture": body,
		"label_face": torch.zeros((length, 1)).cuda(),
		"label_left_hand": torch.zeros((length, 1)).cuda(),
		"label_right_hand": torch.zeros((length, 1)).cuda(),
		"label_body_posture": torch.zeros((length, 1)).cuda()
	}]

	with torch.no_grad():
		result = model.extract_features(source, padding_mask=None, kmeans_labels=None, mask=False)

	# Stack every transformer layer's output into a single [L, T, D] array
	# (batch size is always 1 here, so squeeze it out).
	layer_outputs = []
	for layer in result['layer_results']:
		layer_output = layer[-1]
		layer_output = layer_output.squeeze(1)  # Shape: [T, D]
		layer_outputs.append(layer_output.cpu().numpy())

	features = np.stack(layer_outputs, axis=0)  # Shape: [L, T, D]

	return features


if __name__ == "__main__":

	# Create results directory to save the video crops and features
	out_dir = args.output_dir
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# Load YOLO model
	yolo_model = YOLO(args.yolo_ckpt, verbose=False)
	yolo_model.to(device)

	# Keypoint initialization
	base_options_hand = python.BaseOptions(model_asset_path=args.hand_kp_ckpt)
	options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand,
												num_hands=6, min_hand_detection_confidence=0.05)
	hand_detector = vision.HandLandmarker.create_from_options(options_hand)
	mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.1)

	# Load DinoV2 model
	model_hands = get_dino_finetuned_downloaded(args.dino_ckpt)
	print("DinoV2 model for hands loaded")

	# Load SHubert model
	model_shubert = load_model_shubert(args.shubert_ckpt)
	print("SHuBERT model loaded")
	print("--------------------------------")

	video_file = args.input_video

	try:
		print("Video file: ", video_file)

		if not os.path.exists(video_file):
			print(f"Skipping {video_file} - video file does not exist")
			exit()

		video_id = os.path.splitext(os.path.basename(video_file))[0]
		out_folder = os.path.join(out_dir, video_id)
		if not os.path.exists(out_folder):
			os.makedirs(out_folder)

		out_feat_path_pose = os.path.join(out_folder, video_id + "_pose.npy")
		out_feat_path_hand1 = os.path.join(out_folder, video_id + "_hand1.npy")
		out_feat_path_hand2 = os.path.join(out_folder, video_id + "_hand2.npy")
		out_feat_path_shubert = os.path.join(out_folder, video_id + "_shubert.npy")

		if os.path.exists(out_feat_path_pose) and os.path.exists(out_feat_path_hand1) and os.path.exists(out_feat_path_hand2) and os.path.exists(out_feat_path_shubert):
			print(f"Skipping {video_file} - output already exists")
			exit()

		cap = cv2.VideoCapture(video_file)
		if not cap.isOpened():
			print(f"Error: Could not open video file {video_file}")
			exit()
		fps = cap.get(cv2.CAP_PROP_FPS)

		if fps != 25:
			print("Resampling video to 25 FPS")
			video_path_25fps = os.path.join(out_folder, 'video_25fps.avi')
			command = ("ffmpeg -hide_banner -loglevel panic -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (video_file, video_path_25fps))
			subprocess.call(command, shell=True, stdout=None)
			video_file = video_path_25fps

		inp_hand1_video, inp_hand2_video, body_pose_feats = preprocess_video(video_file, yolo_model, out_folder, out_feat_path_pose)
		print(f"Successfully extracted video crops and body pose features for {video_file}")
		print("--------------------------------")

		hand1_feats, hand2_feats = extract_dino_feats(model_hands=model_hands, inp_hand1_video=inp_hand1_video, inp_hand2_video=inp_hand2_video, out_feat_path_hand1=out_feat_path_hand1, out_feat_path_hand2=out_feat_path_hand2)
		print(f"Successfully extracted hand features for {video_file}")
		print("--------------------------------")

		shubert_feats = get_shubert_feats(model_shubert, hand1_feats, hand2_feats, body_pose_feats)
		print("SHuBERT embeddings extracted: ", shubert_feats.shape)
		np.save(out_feat_path_shubert, shubert_feats)
		print("Successfully extracted SHuBERT features and saved to:", out_feat_path_shubert)
		print("--------------------------------")

	except Exception as e:
		print(f"Error processing {video_file}: {e}")
		exit()
