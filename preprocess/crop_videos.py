"""
Spatially crop downloaded video clips to the person track used in the dataset.

Given a CSV of clip metadata and a directory of clips downloaded by 
download_videos.py, this script reproduces the crop for each clip: 
crop according to the bounding box, pad the frame when needed, and optionally 
resize before writing the final MP4.
"""

import argparse
import ast
import os
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Gray padding added around each frame before cropping, so bounding boxes that
# extend past the original frame edges still yield a valid crop.
PAD_VALUE = 110

# Tracks longer than this many frames are resized to TARGET_HEIGHT.
RESIZE_FRAME_THRESHOLD = 3000
TARGET_HEIGHT = 480

# Command line arguments
parser = argparse.ArgumentParser(
    description="Spatially crop downloaded videos to the dataset's person tracks"
)
parser.add_argument("--input_csv", type=str, required=True,
                    help="Input CSV file containing bounding-box co-ordinates")
parser.add_argument("--video_dir", type=str, required=True,
                    help="Directory with videos downloaded by download_videos.py")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Output root; crops saved as {output_dir}/{fname}.mp4")

parser.add_argument("--fps", type=int, default=25,
                    help="Output frame rate")
parser.add_argument("--resize_frame_threshold", type=int, default=RESIZE_FRAME_THRESHOLD,
                    help="Resize the crop when the track's num_frames exceeds this")
parser.add_argument("--overwrite", action="store_true",
                    help="Recrop rows whose output file already exists (default: skip them)")
args = parser.parse_args()


def parse_bbox(bbox_value):
    if isinstance(bbox_value, str):
        return ast.literal_eval(bbox_value)
    return list(bbox_value)


def get_downloaded_video_path(video_dir, source_file, source_video_start, source_video_end):
    """Build the path download_videos.py used when saving this clip."""
    start = format(float(source_video_start), '.2f')
    end = format(float(source_video_end), '.2f')
    return os.path.join(video_dir, f"{source_file}_{start}-{end}.mp4")


def compute_aspect_resize_dims(original_height, original_width, target_height=None, target_width=None):
    """Compute output dimensions that preserve aspect ratio for a given target height or width."""
    if target_height is not None:
        scale = target_height / original_height
        new_width = int(original_width * scale)
        return target_height, new_width
    if target_width is not None:
        scale = target_width / original_width
        new_height = int(original_height * scale)
        return new_height, target_width
    raise ValueError("Either target_height or target_width must be specified.")


def resample_video_to_25fps(input_path, output_path):
    """Resample a video to a fixed frame rate before cropping, so frame indices line up."""
    command = (
        f"ffmpeg -hide_banner -loglevel panic -y -i {input_path} "
        f"-qscale:v 2 -async 1 -r 25 {output_path}"
    )
    return subprocess.call(command, shell=True, stdout=None) == 0


def crop_frame_with_padding(frame, bbox, pad):
    """
    Pad the frame by `pad` pixels of gray on every side, then slice out bbox.

    bbox is [x1, y1, x2, y2] in padded-frame pixel coordinates, so it is always
    within the padded frame's bounds by construction and needs no clamping here.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop_h = y2 - y1
    crop_w = x2 - x1
    if crop_h <= 0 or crop_w <= 0:
        return None

    padded_frame = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=PAD_VALUE)
    return padded_frame[y1:y2, x1:x2]


def crop_video(
    input_path,
    output_path,
    bbox,
    pad,
    source_width,
    source_height,
    frame_rate=25,
    resize_needed=False,
    target_height=TARGET_HEIGHT,
    expected_frames=None,
):
    """
    Crop a single downloaded clip: resample to a fixed frame rate, optionally
    resize each frame back to the source resolution, pad and crop to bbox,
    optionally resize the crop, and write the result with its original audio.

    expected_frames, when given, is the clip's intended frame count. The
    re-downloaded clip's duration is never bit-exact, so the frame-rate resample
    can emit one extra frame; only trim down to expected_frames when there's
    excess, never pad out a short clip.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        resampled_path = os.path.join(tmp_dir, "video.avi")
        if not resample_video_to_25fps(input_path, resampled_path):
            print(f"  ffmpeg resample failed for {input_path}")
            return False

        video_stream = cv2.VideoCapture(resampled_path)
        if not video_stream.isOpened():
            print(f"  Could not open resampled video: {resampled_path}")
            return False

        resize_size = None
        cropped_frames = []

        while True:
            ok, image = video_stream.read()
            if not ok:
                break

            if (
                source_width
                and source_height
                and (image.shape[1] != source_width or image.shape[0] != source_height)
            ):
                image = cv2.resize(
                    image,
                    (int(source_width), int(source_height)),
                    interpolation=cv2.INTER_LINEAR,
                )

            crop = crop_frame_with_padding(image, bbox, pad)
            if crop is None:
                continue

            if resize_needed:
                if resize_size is None:
                    crop_h, crop_w = crop.shape[:2]
                    out_h, out_w = compute_aspect_resize_dims(
                        crop_h, crop_w, target_height=target_height,
                    )
                    resize_size = (out_w, out_h)
                crop = cv2.resize(crop, resize_size)

            cropped_frames.append(crop.astype(np.uint8))

        video_stream.release()

        if not cropped_frames:
            print(f"  No valid cropped frames for {input_path}")
            return False

        if expected_frames is not None and len(cropped_frames) > expected_frames:
            cropped_frames = cropped_frames[:expected_frames]

        temp_avi = os.path.join(tmp_dir, "cropped.avi")
        shape_video = (cropped_frames[0].shape[1], cropped_frames[0].shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(temp_avi, fourcc, frame_rate, shape_video)
        if not writer.isOpened():
            print(f"  Could not create video writer for {output_path}")
            return False

        for frame in cropped_frames:
            writer.write(np.uint8(frame))
        writer.release()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        command = (
            f"ffmpeg -hide_banner -loglevel panic -y "
            f"-i {temp_avi} -i {resampled_path} "
            f"-map 0:v:0 -map 1:a:0? "
            f"-vf scale=in_range=full:out_range=tv -color_range tv -pix_fmt yuv420p "
            f"-c:v libx264 -crf 18 -preset fast "
            f"-c:a aac -b:a 128k -shortest "
            f"{output_path}"
        )
        if subprocess.call(command, shell=True, stdout=None) != 0:
            print(f"  ffmpeg mux failed for {output_path}")
            return False

    return True


def main():
    
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    if "num_frames" not in df.columns:
        print("Warning: CSV has no num_frames column; resize will not be applied")

    os.makedirs(args.output_dir, exist_ok=True)

    success = 0
    failed = 0
    skipped = 0
    resized = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping"):
        fname = str(row["fname"])
        output_path = os.path.join(args.output_dir, f"{fname}.mp4")

        if not args.overwrite and os.path.exists(output_path):
            skipped += 1
            continue

        input_path = get_downloaded_video_path(
            args.video_dir,
            row["source_file"],
            row["source_video_start"],
            row["source_video_end"],
        )

        if not os.path.exists(input_path):
            print(f"  Downloaded video not found: {input_path}")
            failed += 1
            continue

        num_frames = int(row["num_frames"]) if "num_frames" in row and pd.notna(row["num_frames"]) else None
        resize_needed = num_frames is not None and num_frames > args.resize_frame_threshold
        if resize_needed:
            resized += 1

        expected_frames = None
        if pd.notna(row.get("source_video_start")) and pd.notna(row.get("source_video_end")):
            clip_duration = float(row["source_video_end"]) - float(row["source_video_start"])
            expected_frames = round(clip_duration * args.fps)

        bbox = parse_bbox(row["bbox"])
        pad = int(row["pad"])
        source_width = row.get("source_width")
        source_height = row.get("source_height")
        if pd.isna(source_width) or pd.isna(source_height):
            source_width = None
            source_height = None
        else:
            source_width = int(source_width)
            source_height = int(source_height)

        ok = crop_video(
            input_path,
            output_path,
            bbox,
            pad,
            source_width,
            source_height,
            frame_rate=args.fps,
            resize_needed=resize_needed,
            target_height=args.target_height,
            expected_frames=expected_frames,
        )
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\nDone. Success: {success}, Failed: {failed}, Skipped: {skipped}, Resized: {resized}")


if __name__ == "__main__":
    main()
