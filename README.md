<h1 align="center">Recognition and Localization of Semantic Co-speech Gestures</h1>

<p align="center">
  This code is for our paper titled: <b>Recognizing Co-speech Gestures in-the-Wild</b>
</p>

<p align="center">
  <a href="https://eccv.ecva.net/Conferences/2026"><img src="https://img.shields.io/badge/ECCV-2026-6f42c1?style=for-the-badge&labelColor=1a1a2e" alt="ECCV 2026"></a>
</p>

<p align="center">
  <a href="https://sindhu-hegde.github.io">Sindhu Hegde</a>
  &nbsp;·&nbsp;
  <a href="https://www.robots.ox.ac.uk/~prajwal/">K R Prajwal</a>
  &nbsp;·&nbsp;
  <a href="https://scholar.google.com/citations?hl=en&user=UZ5wscMAAAAJ">Andrew Zisserman</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.31589"><img src="https://img.shields.io/badge/arXiv-2605.31589-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.robots.ox.ac.uk/~vgg/research/grw/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://www.robots.ox.ac.uk/~vgg/research/grw/dataset"><img src="https://img.shields.io/badge/Dataset-Visualization-green.svg" alt="GRW Dataset Visualization"></a>
  <a href="https://huggingface.co/datasets/sindhuhegde/grw"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-GRW-FFD21E.svg" alt="GRW Dataset on Hugging Face"></a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-yellow.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10">
</p>

<p align="center">
  <img src="assets/teaser.gif" width="600" alt="Semantic co-speech gestures recognized and localized in real-world videos"/>
</p>

Our aim is to recognise and localize **semantic gestures** in real-world videos. These gestures are visually depictive and semantically linked to specific spoken words. We introduce a new large-scale benchmark, **GRW** (Gesture Recognition in-the-Wild), which provides word-level annotations and gesture boundaries for semantic gestures occurring in unconstrained real-world settings.

---

## 🚀 News

- **[2026.08.15]** 🏋️ **Training code** released - Both models: (i) semantic gesture classification, and (ii) word recognition and localization, can now be trained from scratch
- **[2026.08.13]** 🔥 **Inference code** released - It is now possible to obtain results for the two tasks: (i) semantic gesture classification, and (ii) word recognition and localization, for any real-world video
- **[2026.08.08]** 🧬 **Pre-trained** checkpoints released
- **[2026.08.06]** 📋 Paper released on [arXiv](https://arxiv.org/abs/2605.31589)
- **[2026.06.17]** 🏆 Our paper is accepted at **ECCV 2026**!!!

---

## ⚙️ Installation

Clone the repository
```bash
git clone https://github.com/Sindhu-Hegde/grw.git
```

Install the required packages (it is recommended to create a new environment)
```bash
python3.10 -m venv env_grw
source env_grw/bin/activate
pip install -r requirements.txt
```

FFmpeg is also needed, install if not already present using: `sudo apt-get install ffmpeg`

**Note:** The code has been tested with `Python 3.10.12` and `ffmpeg 4.4.2`

---

## 📊 The GRW Dataset

We present the GRW (Gesture Recognition in-the-Wild) dataset: A new word-level gesture dataset containing 155 unique semantic gesture words.

Click on the [visualization page](https://www.robots.ox.ac.uk/~vgg/research/grw/dataset) to see sample videos from our dataset.

### Download instructions

The csv files inside `files/` are the train and test datasets for the two tasks. These csv files contain the video-ids along with other annotations. They can be read by:

```python
import pandas as pd

# Semantic gesture classification
df_semantic_train = pd.read_csv("files/semantic_classification_train.csv")	# Semantic Gesture Classification train set
df_semantic_test = pd.read_csv("files/semantic_classification_test.csv")	# Semantic Gesture Classification test set

# Gesture word recognition and localization
df_recog_localize_train = pd.read_csv("files/recognition_localization_train.csv")	# Word Recognition and Localization train set
df_recog_localize_test = pd.read_csv("files/recognition_localization_test.csv")	# Word Recognition and Localization test set
```

To download and pre-process the videos to obtain gesture crops, run the following commands:

```bash
cd preprocess

# Download the videos from YouTube-ids and timestamps
python download_videos.py --input_csv=<csv-file> --result_dir=<raw-video-root>

# Crop the videos with the bounding-box co-ordinates provided in the csv files
python crop_videos.py --input_csv=<csv-file> --video_dir=<raw-video-root> --output_dir=<preprocessed-video-root>
```

**Note:** Due to new YouTube policies, downloading videos (especially for train sets) might take a long time. Thus, we also provide the SHuBERT features (which is taken as input by our gesture models). Users can directly download the SHuBERT features as shown [below](#download-shubert-features) and start the training/evaluation.

<details>
<summary><b>Folder structure after download and pre-processing</b></summary>

```text
raw_video_root (path of the downloaded raw videos)
├── *.mp4 (raw uncropped videos)
```

```text
preprocessed_video_root (path of the pre-processed gesture videos)
├── word folders
│   ├── *.mp4 (extracted person-specific gesture video)
```

</details>

#### Download SHuBERT features

We provide pre-extracted SHuBERT features which are needed for training and evaluating our models. To download these features, run the following:

| Feature set | Download Link |
|:--:|:--:|
| Semantic Classification - Train | [Link](https://thor.robots.ox.ac.uk/grw/shubert-features/semantic_classification_train.tar.gz) |
| Semantic Classification - Test | [Link](https://thor.robots.ox.ac.uk/grw/shubert-features/semantic_classification_test.tar.gz) |
| Semantic Classification - Test unseen words | [Link](https://thor.robots.ox.ac.uk/grw/shubert-features/semantic_classification_test_unseen_words.tar.gz) |
| Word Recognition & Localization - Train | [Link](https://thor.robots.ox.ac.uk/grw/shubert-features/recognition_localization_train.tar.gz) |
| Word Recognition & Localization - Test | [Link](https://thor.robots.ox.ac.uk/grw/shubert-features/recognition_localization_test.tar.gz) |

<details>
<summary><b>Verify downloads (checksums)</b></summary>

Download the checksum file from [here](https://thor.robots.ox.ac.uk/grw/SHA512SUMS)
After downloading all the files, they can be verified by the following:

```bash
# Run the following command
sha512sum -c SHA512SUMS

# The output should look like:
recognition_localization_test.tar.gz: OK
recognition_localization_train.tar.gz: OK
semantic_classification_test.tar.gz: OK
semantic_classification_train.tar.gz: OK
semantic_classification_test_unseen_words.tar.gz: OK

# Alternatively, if only single file has been downloaded, it can be verified using:
grep "recognition_localization_test.tar.gz" SHA512SUMS | sha512sum -c

# The output should look like:
recognition_localization_test.tar.gz: OK
```

</details>

Untar the downloaded files for training and evaluation:
```bash
for f in *.tar.gz; do
    tar -xzf "$f"
done
```

<details>
<summary><b>Folder structure of the extracted features</b></summary>

```text
shubert_features (path of the extracted shubert features)
├── semantic_classification
│   ├── split (<train>/<test>/<test_unseen_words>)
│   │   ├── *.npy
├── recognition_localization
│   ├── split (<train>/<test>)
│   │   ├── *.npy
```

</details>

**Note:** The download and processing of the gesture videos can be SKIPPED if SHuBERT features are downloaded directly.

---

## 💾 Checkpoints

Download the trained models and save in `checkpoints` folder
```bash
mkdir checkpoints
cd checkpoints

#### Semantic Classification model
wget https://thor.robots.ox.ac.uk/grw/checkpoints/semantic_classification.pth

#### Word Recognition and Localization model
wget https://thor.robots.ox.ac.uk/grw/checkpoints/recognition_localization.pth

#### SHuBERT models (needed for preprocessing)
# [1] YOLO
wget https://thor.robots.ox.ac.uk/grw/checkpoints/shubert/yolov8n.pt

# [2] Dino fine-tuned for hands keypoint model
wget https://thor.robots.ox.ac.uk/grw/checkpoints/shubert/hand_dinov2.pth

# [3] Hand keypoint model
wget https://thor.robots.ox.ac.uk/grw/checkpoints/shubert/hand_landmarker.task

# [4] SHuBERT model
wget https://thor.robots.ox.ac.uk/grw/checkpoints/shubert/shubert.pt

cd ..
```

Alternatively, these checkpoints can also be downloaded directly from the links below:

| Model | Download Link |
|:--:|:--:|
| Semantic Classification model | [Link](https://thor.robots.ox.ac.uk/grw/checkpoints/semantic_classification.pth) |
| Word Recognition and Localization model | [Link](https://thor.robots.ox.ac.uk/grw/checkpoints/recognition_localization.pth) |
| SHuBERT | 1. [YOLO](https://thor.robots.ox.ac.uk/grw/checkpoints/shubert/yolov8n.pt)<br>2. [Dino fine-tuned for hands](https://thor.robots.ox.ac.uk/grw/checkpoints/shubert/hand_dinov2.pth)<br>3. [Hand keypoint model](https://thor.robots.ox.ac.uk/grw/checkpoints/shubert/hand_landmarker.task)<br>4. [SHuBERT](https://thor.robots.ox.ac.uk/grw/checkpoints/shubert/shubert.pt) |

---

## 🎯 Inference on a real-world video

### Step-1: Pre-process the video

The first step is to preprocess the video and obtain gesture crops. Run the following command to pre-process the video:

```bash
python preprocess/inference_preprocess.py --video_file <path-to-video-file>
```

The processed gesture video tracks are saved in: `<results/video_file/preprocessed/*.avi>`. The default save directory is `results`, this can be changed by specifying `--preprocessed_root` in the above command. Once the gesture tracks are extracted, the below script can be used to extract SHuBERT features.

### Step-2: Extract SHuBERT features

To extract SHuBERT features, `fairseq` needs to be installed:
```bash
cd models/fairseq
pip install -e .
cd ../..
```

After installing, proceed to extract the features
```bash
python preprocess/inference_shubert.py \
  --input_video=<preprocessed-video>  \
  --yolo_ckpt=checkpoints/yolov8n.pt \
  --hand_kp_ckpt=checkpoints/hand_landmarker.task \
  --dino_ckpt=checkpoints/hand_dinov2.pth \
  --shubert_ckpt=checkpoints/shubert.pt
```

The SHuBERT features and metadata like hand videos are saved in: `results/preprocessed_video/preprocessed_video_shubert.npy`. The default save directory is `results`, this can be changed by specifying `--output_dir` in the above command. Once the features are extracted, the below script can be used to obtain predictions for the semantic classification, word recognition and localization tasks.

### Step-3: Inference: (i) Semantic classification and (ii) Word recognition & localization

```bash
python inference.py \
  --input_feature=<extracted-SHuBERT-feature> \
  --model_semantic_classification=checkpoints/semantic_classification.pth \
  --model_recognition_localization=checkpoints/recognition_localization.pth
```

The outputs for the downstream tasks are displayed on the terminal.
For semantic classification, the default threshold set is `0.7`. This can be modified using `--semantic_thr` flag.

For a quick test, the following pre-extracted features are available in `samples` folder:
- `bye.mp4`, `bye_shubert.npy`
- `below.mp4`, `below_shubert.npy`

**Note:** Steps 1 & 2 need to be skipped for these examples, since they are already pre-processed.

Example run:
```bash
python inference.py \
  --input_feature=samples/bye_shubert.npy \
  --model_semantic_classification=checkpoints/semantic_classification.pth \
  --model_recognition_localization=checkpoints/recognition_localization.pth
```

On running the above command, the following outputs are displayed:

```text
Loaded checkpoint from: checkpoints/semantic_classification.pth
Loaded checkpoint from: checkpoints/recognition_localization.pth
Loaded SHuBERT visual features:  torch.Size([100, 12, 768])
----------------------------------
----------- RESULTS --------------
----------------------------------

Semantic classification prediction:
  'Semantic' class

Word classification top-5 predictions:
  1. bye
  2. below
  3. shake
  4. press
  5. hello

Gesture localization (boundary) prediction:
  Gesture start frame: 61 | Gesture end frame: 91
```

---

## 📈 Evaluation

To reproduce the scores reported in paper, follow the steps illustrated below:

### Step-1: Download the SHuBERT test features

Use the links [above](#download-shubert-features) to download and unzip the SHuBERT test features

### Step-2: Compute the metrics

#### Task-1: Semantic gesture classification

```bash
python evaluate_semantic_classification.py \
  --test_csv=files/semantic_classification_test.csv \
  --checkpoint_path=checkpoints/semantic_classification.pth \
  --feature_dir=shubert_features/semantic_classification/test
```

The results obtained are displayed below:

<div align="center">

*Semantic gesture classification on the GRW test set*

| Accuracy | Precision | Recall | High-confidence Accuracy |
|:--:|:--:|:--:|:--:|
| **75.83** | **79.91** | **69.00** | **93.20** |

</div>

#### Task-2: Word recognition and localization

```bash
python evaluate_recognition_localization.py \
  --test_csv=files/recognition_localization_test.csv \
  --checkpoint_path=checkpoints/recognition_localization.pth \
  --feature_dir=shubert_features/recognition_localization/test
```

The results obtained are displayed below:

<div align="center">

*Word recognition and localization on the GRW test set*

| Acc@1 | Acc@5 | Acc@10 | mIoU |
|:--:|:--:|:--:|:--:|
| **18.35** | **37.30** | **51.70** | **0.67** |

</div>

The predictions from the recognition and localization model are saved in: `preds.csv`

---

## 🏋️ Training

For training, pre-extracted SHuBERT features are required. So make sure the train features have either been downloaded and untarred as described [above](#download-shubert-features) or have been extracted from the gesture videos.

Only the train csv is needed: a small validation set is held out from it automatically (500 clips by default, controlled by `--val_size`), so no separate validation csv is required. The split is stratified, seeded by `--seed`, and disjoint from the data the model is trained on. Checkpoints are written to `--checkpoint_dir` after every epoch, and training can be resumed from any of them with `--checkpoint_path`.

### Task-1: Semantic gesture classification

Trains the binary classifier that decides whether the gesture co-occurring with a target word is semantic.

```bash
python train_semantic_classifier.py \
  --train_csv=<path-to-train-csv> \
  --checkpoint_dir=<path-to-save-checkpoints> \
  --feature_dir=<path-to-shubert-features>
```

Example run:

```bash
python train_semantic_classifier.py \
  --train_csv=files/semantic_classification_train.csv \
  --checkpoint_dir=checkpoints/semantic_classification \
  --feature_dir=shubert_features/semantic_classification/train
```

The two classes are heavily imbalanced, so the sampler and the loss are both class-weighted automatically from the train split. Each epoch prints the running loss and accuracy, followed by the loss and accuracy on the held-out validation set.

### Task-2: Word recognition and localization

Trains the joint model that recognizes which of the 155 gesture words is being performed and localizes its temporal boundary.

```bash
python train_recognition_localization.py \
  --train_csv=<path-to-train-csv> \
  --checkpoint_dir=<path-to-save-checkpoints> \
  --feature_dir=<path-to-shubert-features>
```

Example run:

```bash
python train_recognition_localization.py \
  --train_csv=files/recognition_localization_train.csv \
  --checkpoint_dir=checkpoints/recognition_localization \
  --feature_dir=shubert_features/recognition_localization/train
```

The recognition and localization heads are optimized jointly. Each epoch prints the two losses along with the word accuracy and the mIoU of the predicted boundary. Pass `--apply_augmentations` to enable temporal augmentation, which shifts the features in time and moves the ground-truth boundaries with them.

### Useful flags

| Flag | Default | Description |
|:--|:--:|:--|
| `--val_size` | 500 | Number of clips held out from the train csv for validation |
| `--seed` | 42 | Random seed for the train/validation split |
| `--num_epochs` | 20 | Number of epochs to train for |
| `--batch_size` | 128 | Batch size |
| `--num_workers` | 8 | Dataloader workers |
| `--lr` | 1e-4 | Learning rate |
| `--checkpoint_path` | `None` | Resume training from this checkpoint |
| `--apply_augmentations` | off | Temporal augmentation (Task-2 only) |

---

## 📖 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{hegde_eccv_2026,
  title={Recognizing Co-Speech Gestures in-the-Wild},
  author={Sindhu B Hegde and K R Prajwal and Andrew Zisserman},
  year={2026},
  booktitle={European Conference on Computer Vision (ECCV)}
}
```
