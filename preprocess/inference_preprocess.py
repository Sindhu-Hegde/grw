import os
import argparse
import pickle
import subprocess
import cv2
import numpy as np
from shutil import rmtree
from tqdm import tqdm
from scipy.interpolate import interp1d
from ultralytics import YOLO
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict

parser = argparse.ArgumentParser()
parser.add_argument('--video_file', type=str, required=True, help='Path of the video file to be pre-processed')
parser.add_argument('--preprocessed_root', type=str, required=False, default="results", help='Path to save the output crops')

parser.add_argument('--crop_scale', type=float, default=0, help='Scale bounding box')
parser.add_argument('--min_track', type=int, default=10, help='Minimum facetrack duration')
parser.add_argument('--min_frame_size', type=int, default=64, help='Minimum frame size in pixels')
parser.add_argument('--num_failed_det', type=int, default=25, help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate')
opt = parser.parse_args()

# Load the YOLO model
yolo_model = YOLO("yolov9m.pt")

# Initialize the mediapipe holistic keypoint detection model
mp_holistic = mp.solutions.holistic


def bb_intersection_over_union(boxA, boxB):

	'''
	This function calculates the intersection over union of two bounding boxes

	Args:
		- boxA (list): Bounding box A.
		- boxB (list): Bounding box B.
	Returns:
		- iou (float): Intersection over union of the two bounding boxes
	'''

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA) * max(0, yB - yA)

	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

def track_speakers(opt, scenefaces):

	'''
	This function tracks the speakers in the video using the detected bounding boxes

	Args:
		- opt (argparse): Argument parser.
		- dets (list): List of detections.
	Returns:
		- tracks (list): List of tracks.
	'''

	iouThres = 0.5  
	tracks = []

	while True:
		track = []
		for framefaces in scenefaces:
			for face in framefaces:
				if track == []:
					track.append(face)
					framefaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						framefaces.remove(face)
						continue
				else:
					break

		if track == []:
			break
		elif len(track) > opt.min_track:
			framenum = np.array([f['frame'] for f in track])
			bboxes = np.array([np.array(f['bbox']) for f in track])

			frame_i = np.arange(framenum[0], framenum[-1] + 1)

			bboxes_i = []
			for ij in range(0, 4):
				interpfn = interp1d(framenum, bboxes[:, ij])
				bboxes_i.append(interpfn(frame_i))
			bboxes_i = np.stack(bboxes_i, axis=1)

			if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_frame_size:
				tracks.append({'frame': frame_i, 'bbox': bboxes_i})

	return tracks


def check_leg_visibility(bbox_array, aspect_thresh=1.6, min_leg_ratio=0.8):

	'''
	Check if legs are visible based on aspect ratio alone.

	Args:
		bbox_array (np.ndarray): (N, 4) array of [x1, y1, x2, y2].
		aspect_thresh (float): Minimum H/W ratio to assume full-body.
		min_leg_ratio (float): Fraction of frames where legs must be visible.

	Returns:
		bool: True if legs likely visible in majority of frames.
	'''

	widths = bbox_array[:, 2] - bbox_array[:, 0]
	heights = bbox_array[:, 3] - bbox_array[:, 1]
	aspect_ratios = np.divide(heights, widths, out=np.zeros_like(heights), where=widths!=0)

	# Condition: tall enough and not cut off at bottom
	leg_visible_mask = aspect_ratios > aspect_thresh
	leg_visible_ratio = np.sum(leg_visible_mask) / len(bbox_array)

	return leg_visible_ratio >= min_leg_ratio

def get_keypoints(frames):

	'''
	This function extracts the keypoints from the frames using MediaPipe Holistic pipeline

	Args:
		- frames (list) : List of frames extracted from the video
	Returns:
		- all_frame_kps (list) : List of keypoints for each frame
	'''

	resolution = frames[0].shape

	all_frame_kps = []
	holistic = mp_holistic.Holistic(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	)
	for i, frame in tqdm(enumerate(frames), total=len(frames)):

		results = holistic.process(frame)

		if results.pose_landmarks is not None:
			pose = protobuf_to_dict(results.pose_landmarks)['landmark']

			pose_kps = []
			for kps in pose:
				if kps is not None:
					pose_kps.append([
									round(kps["x"] * resolution[1], 2),
									round(kps["y"] * resolution[0], 2),
									round(kps["visibility"], 2)
								])

			pose_kps = np.array(pose_kps, dtype=np.float16)

			all_frame_kps.append(pose_kps)

		# Periodically recreate the holistic model to bound its memory growth
		# over long videos.
		if i % 1000 == 0:
			holistic.close()
			holistic = mp_holistic.Holistic(
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5
			)

	if len(all_frame_kps) == 0:
		return None
	else:
		return all_frame_kps

def adjust_bbox_kps(frames, all_frame_kps, padding_x=25, padding_y=-15):

	'''
	This function crops frames based on the detected keypoints
	
	Args:
		- frames (list of numpy arrays): List of cropped frames.
		- all_frame_kps (numpy array): T x num_keypoints x 3 (x, y, confidence).
		- padding_x (int): Extra padding for cropping.
		- padding_y (int): Extra padding for cropping.
	
	Returns:
		- cropped_frames (list of numpy arrays): Cropped frames according to adjusted bounding box from keypoints
	'''
	
	LEFT_KP_INDICES = [12, 14, 16, 18, 20, 22, 24]
	RIGHT_KP_INDICES = [11, 13, 15, 17, 19, 21, 23]
	LEFT_HIP_IDX = 23
	RIGHT_HIP_IDX = 24

	left_xs = []
	right_xs = []
	waist_ys = []

	for keypoints in all_frame_kps:
		# Extract left and right keypoints that have confidence > 0.7
		left_kps = [keypoints[i] for i in LEFT_KP_INDICES if keypoints[i][2] > 0.7]
		right_kps = [keypoints[i] for i in RIGHT_KP_INDICES if keypoints[i][2] > 0.7]

		# Compute spatial limits if keypoints exist
		if left_kps:
			left_xs.append(min(kp[0] for kp in left_kps))  # Leftmost x
		if right_kps:
			right_xs.append(max(kp[0] for kp in right_kps))  # Rightmost x

		# Compute waistline from hips if both are detected with confidence > 0.7
		left_hip, right_hip = keypoints[LEFT_HIP_IDX], keypoints[RIGHT_HIP_IDX]
		if left_hip[2] > 0.7 and right_hip[2] > 0.7:
			waist_ys.append((left_hip[1] + right_hip[1]) / 2)

	# Compute global cropping limits
	frame_height, frame_width = frames[0].shape[:2]

	if len(left_xs) > 0 and len(left_xs)/len(all_frame_kps) > 0.7:
		left_x = int(min(left_xs)) - padding_x
	else:
		left_x = 0

	if len(right_xs) > 0 and len(right_xs)/len(all_frame_kps) > 0.7:
		right_x = int(max(right_xs)) + padding_x
	else:
		right_x = frame_width

	if len(waist_ys) > 0 and len(waist_ys)/len(all_frame_kps) > 0.7:
		upper_body_estimate = int(np.mean(waist_ys))  # Use average waist position
		new_y2 = upper_body_estimate + padding_y
	else:
		new_y2 = frame_height

	# Ensure within frame bounds
	left_x = max(0, left_x)
	right_x = min(frame_width, right_x)
	new_y2 = min(new_y2, frame_height)

	# Crop all frames
	cropped_frames = [frame[:new_y2, left_x:right_x] for frame in frames]

	legs_bbox = [left_x, right_x, new_y2]

	return cropped_frames, legs_bbox


def compute_aspect_resize_dims(original_height, original_width, target_height=None, target_width=None):

	'''
	This function computes the resize dimensions that preserve the original aspect ratio,
	given either a target height or a target width.

	Args:
		- original_height (int): Height of the original frame.
		- original_width (int): Width of the original frame.
		- target_height (int): Desired output height (mutually exclusive with target_width).
		- target_width (int): Desired output width (mutually exclusive with target_height).
	Returns:
		- (height, width) (tuple): Resize dimensions matching the original aspect ratio.
	'''

	if target_height is not None:
		scale = target_height / original_height
		new_width = int(original_width * scale)
		return target_height, new_width
	elif target_width is not None:
		scale = target_width / original_width
		new_height = int(original_height * scale)
		return new_height, target_width
	else:
		raise ValueError("Either target_height or target_width must be specified.")


def detect_speaker(opt, padding=10, work_dir=None):

	'''
	This function detects the speaker in the video using YOLOv9 model

	Args:
		- opt (argparse): Argument parser.
		- padding (int): Extra padding for cropping.
		- work_dir (str): Directory to save the person.pkl file.
	Returns:
		- alltracks (list): List of tracks.
	'''
	
	videofile = os.path.join(opt.avi_dir, 'video.avi')
	vidObj = cv2.VideoCapture(videofile)

	dets = []
	fidx = 0
	alltracks = []

	while True:
		success, image = vidObj.read()
		if not success:
			break

		image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Perform person detection
		results = yolo_model(image_np, verbose=False)
		detections = results[0].boxes

		dets.append([])
		for i, det in enumerate(detections):
			x1, y1, x2, y2 = det.xyxy[0]
			cls = det.cls[0]
			conf = det.conf[0]  
			if int(cls) == 0 and conf>0.6:  # Class 0 is 'person' in COCO dataset
				x1 = max(0, int(x1) - padding)
				y1 = max(0, int(y1) - padding)
				x2 = min(image_np.shape[1], int(x2) + padding)
				y2 = min(image_np.shape[0], int(y2) + padding)
				dets[-1].append({'frame': fidx, 'bbox': [x1, y1, x2, y2], 'conf': conf})

		fidx += 1

	if len(dets) >= opt.min_track and np.abs(len(dets) - fidx) <= 5:
		alltracks.extend(track_speakers(opt, dets[0:len(dets)]))
	
		savepath = os.path.join(work_dir, 'person.pkl')
		with open(savepath, 'wb') as fil:
			pickle.dump(dets, fil)
		
		return alltracks

	else:
		print("Num. of frames = {} | Num. of detections = {}".format(fidx, len(dets)))
		return None


def crop_video(opt, track, cropfile, tight_scale=1):
	
	'''
	This function crops the video based on the detected bounding boxes.

	Args:
		- opt (argparse): Argument parser.
		- track (dict): Person tracks obtained.
		- cropfile (str): Path to save the cropped video.
		- tight_scale (float): Tight scale for cropping.
	Returns:
		- dets (dict): Detections.
	'''

	bbox_array = np.array(track['bbox'])

	# Get global bbox over all frames
	x1_min = np.min(bbox_array[:, 0])
	y1_min = np.min(bbox_array[:, 1])
	x2_max = np.max(bbox_array[:, 2])
	y2_max = np.max(bbox_array[:, 3])

	# Convert to center and scale
	width = (x2_max - x1_min) * tight_scale
	height = (y2_max - y1_min) * tight_scale
	center_x = (x1_min + x2_max) / 2
	center_y = (y1_min + y2_max) / 2
	scale = max(width, height) / 2

	# Apply fixed crop for all frames
	T = len(track['bbox'])
	dets = {
		'x': [center_x] * T,
		'y': [center_y] * T,
		's': [scale] * T
	}

	videofile = os.path.join(opt.avi_dir, 'video.avi')
	frame_indices = track['frame']
	frame_no_to_start = track['frame'][0]

	video_stream = cv2.VideoCapture(videofile)
	video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no_to_start)

	# Precompute size from first frame
	first_frame = video_stream.read()[1]
	bs = dets['s'][0]
	cs = opt.crop_scale
	bsi = int(bs * (1 + 2 * cs))  # padding size

	# For very long videos, resize the video to 480p to manage memory
	resize_needed = False
	if len(frame_indices) > 3000:

		# Apply padding to first frame to compute final dimensions
		first_frame_padded = np.pad(first_frame, ((bsi, bsi), (bsi, bsi), (0, 0)), mode='constant', constant_values=110)

		padded_center_x = center_x + bsi
		padded_center_y = center_y + bsi

		x1 = int(max(0, padded_center_x - bs * (1 + cs)))
		x2 = int(min(first_frame_padded.shape[1], padded_center_x + bs * (1 + cs)))
		y1 = int(max(0, padded_center_y - bs))
		y2 = int(min(first_frame_padded.shape[0], padded_center_y + bs * (1 + 2 * cs)))

		crop = first_frame_padded[y1:y2, x1:x2]
		height, width = crop.shape[:2]
		
		target_height = 480
		target_size = compute_aspect_resize_dims(height, width, target_height=target_height)
		height, width = target_size
		resize_needed = True

	cropped_frames_fixed_size = []

	video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no_to_start)
	fixed_bbox = []

	for fidx, frame in enumerate(track['frame']):
		image = video_stream.read()[1]
		frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), mode='constant', constant_values=110)

		# Use padded center
		padded_center_x = center_x + bsi
		padded_center_y = center_y + bsi

		# Clamp indices
		H, W = frame.shape[:2]
		x1 = int(max(0, padded_center_x - bs * (1 + cs)))
		x2 = int(min(W, padded_center_x + bs * (1 + cs)))
		y1 = int(max(0, padded_center_y - bs))
		y2 = int(min(H, padded_center_y + bs * (1 + 2 * cs)))

		crop = frame[y1:y2, x1:x2]
		fixed_bbox.append([x1, y1, x2, y2])

		# Validate crop shape
		if crop.shape[0] == 0 or crop.shape[1] == 0:
			print(f"Warning: Skipping empty crop at frame {fidx}")
			continue

		if resize_needed:
			crop = cv2.resize(crop, (width, height))

		cropped_frames_fixed_size.append(crop.astype(np.uint8))

	if not cropped_frames_fixed_size:
		print(f"Error: No valid frames for {cropfile}")
		return dets

	# Get leg visibility
	leg_visible = check_leg_visibility(track['bbox'], aspect_thresh=1.6, min_leg_ratio=0.8)

	legs_bbox = None
	if leg_visible:
		# Get the keypoints for the cropped frames
		all_frame_kps = get_keypoints(cropped_frames_fixed_size)

		if all_frame_kps is None:
			upper_body_cropped_frames = cropped_frames_fixed_size
		else:
			upper_body_cropped_frames, legs_bbox = adjust_bbox_kps(cropped_frames_fixed_size, all_frame_kps)
			cropped_frames_fixed_size = None  # already copied into upper_body_cropped_frames; free early
	else:
		upper_body_cropped_frames = cropped_frames_fixed_size

	print("Cropping done: ", len(upper_body_cropped_frames))

	# Write the cropped frames to a video file
	shape_video = (upper_body_cropped_frames[0].shape[1], upper_body_cropped_frames[0].shape[0])
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	vOut = cv2.VideoWriter(cropfile + '.avi', fourcc, opt.frame_rate, shape_video)

	for i in range(len(upper_body_cropped_frames)):
		frame = upper_body_cropped_frames[i]
		vOut.write(np.uint8(frame))
		upper_body_cropped_frames[i] = None  # Free memory

	print("Successfully written cropped frames to video file")
	vOut.release()

	new_tracks = {"fname": os.path.basename(cropfile), "fixed_bbox": np.array(fixed_bbox), "legs_bbox": legs_bbox}

	return new_tracks


def process_video(file, preprocessed_root):

	'''
	This function processes the video

	Args:
		- file (str): Path to the video file.
		- preprocessed_root (str): Path to save the preprocessed data.
	'''

	folder_name = "preprocessed"
	dest_folder = os.path.join(preprocessed_root, folder_name)

	setattr(opt, 'videofile', file)

	if os.path.exists(opt.work_dir):
		rmtree(opt.work_dir)

	if os.path.exists(opt.crop_dir):
		rmtree(opt.crop_dir)

	if os.path.exists(opt.avi_dir):
		rmtree(opt.avi_dir)

	if os.path.exists(opt.frames_dir):
		rmtree(opt.frames_dir)

	if os.path.exists(opt.tmp_dir):
		rmtree(opt.tmp_dir)

	os.makedirs(opt.work_dir)
	os.makedirs(opt.crop_dir)
	os.makedirs(opt.avi_dir)
	os.makedirs(opt.frames_dir)
	os.makedirs(opt.tmp_dir)
	os.makedirs(dest_folder, exist_ok=True)

	print("Resampling video...")
	command = ("ffmpeg -hide_banner -loglevel panic -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (opt.videofile,
																os.path.join(opt.avi_dir,
																'video.avi')))
	output = subprocess.call(command, shell=True, stdout=None)
	if output != 0:
		print("Error in resampling video")
		return

	# Detect the speaker in the video using YOLO model
	spk_tracks = detect_speaker(opt, work_dir=dest_folder)
	if spk_tracks is None:
		print("No tracks found for ", file)
		return

	# Crop the video based on the detected bounding boxes
	vidtracks = []
	for ii, track in enumerate(spk_tracks):
		vidtracks.append(crop_video(opt, track, os.path.join(dest_folder, '%05d' % ii)))

	if len(vidtracks) > 0:
		savepath = os.path.join(dest_folder, 'tracks.pkl')
		with open(savepath, 'wb') as fil:
			pickle.dump(vidtracks, fil)

	rmtree(opt.tmp_dir)

	print("Saved gesture crops for video: ", file)


if __name__ == "__main__":

	file = opt.video_file
	fname = file.split("/")[-1].split(".")[0]
	print(f"Processing video: {file}")

	# Create the necessary directories
	opt.preprocessed_root = os.path.join(opt.preprocessed_root, fname)
	opt.temp_dir = os.path.join(opt.preprocessed_root, "temp")
	os.makedirs(opt.preprocessed_root, exist_ok=True)
	os.makedirs(opt.temp_dir, exist_ok=True)

	# Set the necessary attributes
	setattr(opt, 'avi_dir', os.path.join(opt.temp_dir, 'pyavi'))
	setattr(opt, 'tmp_dir', os.path.join(opt.temp_dir, 'pytmp'))
	setattr(opt, 'work_dir', os.path.join(opt.temp_dir, 'pywork'))
	setattr(opt, 'crop_dir', os.path.join(opt.temp_dir, 'pycrop'))
	setattr(opt, 'frames_dir', os.path.join(opt.temp_dir, 'pyframes'))

	# Process the video
	process_video(file, opt.preprocessed_root)
