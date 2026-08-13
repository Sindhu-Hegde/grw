import argparse
import pandas as pd
import numpy as np
import os, subprocess
from tqdm import tqdm

import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ["LC_ALL"]="en_US.utf-8"
os.environ["LANG"]="en_US.utf-8"

parser = argparse.ArgumentParser(description="Code to download the videos from the input csv file")
parser.add_argument('--input_csv', required=False, default="files/test_reconition_localization.csv")
parser.add_argument('--result_dir', type=str, default="videos")

args = parser.parse_args()

def is_valid_video(file_path):
	
	'''
	This function validates the downloaded video file by checking if the duration > 0 seconds
	
	Args:
		- file_path (str): Path to the video file.
	Returns:
		- True if the video is valid, False otherwise.
	'''

	if not os.path.exists(file_path):
		return False  # File does not exist

	# Use ffmpeg to get duration
	try:
		result = subprocess.run(
			["ffmpeg", "-i", file_path, "-f", "null", "-"],
			stdout=subprocess.DEVNULL, 
			stderr=subprocess.DEVNULL  
		)
		valid = result.returncode == 0  # If return code is 0, it's a valid video
		if not valid:
			print(f"Invalid video (ffmpeg failed): {file_path}")
			os.remove(file_path)
			return False
	except Exception:
		return False  # If ffmpeg fails, assume it's invalid

	return True
	


def mp_handler(i, df, result_dir):

	'''
	This function handles the multiprocessing of the video download

	Args:
		- i (int): Index of the video.
		- df (pd.DataFrame): DataFrame containing the video information.
		- result_dir (str): Directory to save the video.
	'''

	try:
		data = df.iloc[i]
		vid = data['source_file']
		video_link = "https://www.youtube.com/watch?v={}".format(vid)
		start = data['source_video_start']
		end = data['source_video_end']
		start = format(float(start), '.2f')
		end = format(float(end), '.2f')
		time = "*{}-{}".format(start,end)
		# print(vid, video_link, start, end, time)

		output_fname = os.path.join(result_dir, "{}_{}-{}.mp4".format(vid, start, end))
	
		if os.path.exists(output_fname):
			return

		# Download the video
		cmd = ("yt-dlp --geo-bypass --cookies cookies.txt --download-sections {} "
			   "--force-keyframes-at-cuts -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' "
			   "--merge-output-format mp4 -o {} {}").format(time, output_fname, video_link)
		subprocess.call(cmd, shell=True)


	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()

def download_data(args):

	'''
	This function downloads the videos from the given csv file

	Args:
		- args (argparse.Namespace): Arguments.
	'''

	# Read the csv file
	df = pd.read_csv(args.input_csv)
	print("Total files: ", len(df))

	# Create the result directory
	if not os.path.exists(args.result_dir):
		os.makedirs(args.result_dir)

	# Create the multiprocessing pool and submit the jobs to download the videos
	jobs = [idx for idx in range(len(df))]
	p = ThreadPoolExecutor(8)
	futures = [p.submit(mp_handler, j, df, args.result_dir) for j in jobs]
	res = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

if __name__ == '__main__':

	# Download the videos
	download_data(args)