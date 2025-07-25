import os
from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
import ast
import string
import copy
from tqdm.notebook import tqdm
from moviepy.editor import VideoFileClip
import torch
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from features.feature_config import Config as feat_cfg

device = feat_cfg.DEVICE


def sample_frames(frames, num_frames):
    n = len(frames)
    idxs = np.linspace(0, n - 1, num_frames, dtype=int)
    return [frames[i] for i in idxs]


def get_vision_models(device):
    """
    Load a pre-trained Video Swin Base model (Swin3D_B) and its preprocess transforms.
    Returns a backbone with the classification head removed.
    """
    # 1. Choose the Kinetics-400 pretrained weights
    weights = Swin3D_B_Weights.DEFAULT  # equivalent to KINETICS400_V1 :contentReference[oaicite:0]{index=0}

    # 2. Build the model, disabling the head by setting num_classes=0
    #    (this drops the linear classifier so forward() yields feature maps)
    model = swin3d_b(weights=weights).to(device)
    model.eval()
    
    feature_extractor = model
    feature_extractor.head = torch.nn.Identity()
    # 3. Get the inference transforms: resize→central crop→normalize→permute to (C, T, H, W)
    preprocess = weights.transforms()  # :contentReference[oaicite:1]{index=1}

    return feature_extractor, preprocess


def extract_swin_features(
    episode_path, tr, model, preprocess, device, save_dir, season_num
):
    """
    Identical loop as before, but uses the Video Swin backbone.
    """

    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    extracted_features = []

    if len(start_times) > feat_cfg.SKIP_LENGTH:
        print(f"Skipping {os.path.basename(episode_path)}: too many ({len(start_times)}) chunks")
        return

    num_frames = 32  # or whatever you sampled before; often 8

    for start in tqdm(start_times, desc=f"Swin3D_B – {os.path.basename(episode_path)}"):
        sub = clip.subclip(start, start + tr)
        frames = [frame for frame in sub.iter_frames()]  # list of H×W×C arrays
        frames = sample_frames(frames, num_frames)       # your sampling fn

        # ---- Prep for Video SwinTransformer ----
        # Convert to a torch.Tensor of shape (T, C, H, W)
        vid = torch.tensor(np.stack(frames), dtype=torch.float32)  # (T, H, W, C)
        vid = vid.permute(0, 3, 1, 2)                              # (T, C, H, W)

        # Run the torchvision preprocessing (resize / crop / norm / permute)
        inp = preprocess(vid)        # → (C, T, H', W')
        inp = inp.unsqueeze(0).to(device)  # → (1, C, T, H', W')

        #  Extract features
        with torch.no_grad():
            ext_feats = model(inp)
        
        extracted_features.append(ext_feats[0].cpu().numpy())

    #  Save as before
    extracted_features = np.array(extracted_features).astype('float32')
    season_folder = os.path.join(save_dir, str(season_num))
    os.makedirs(season_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(episode_path))[0] + '_features.npy'
    np.save(os.path.join(season_folder, filename), extracted_features)


def save_swin_features():
    model, preprocess = get_vision_models(device)

    save_dir = "data/swin"
    os.makedirs(save_dir, exist_ok=True)
    tr = feat_cfg.TR

    for movie in feat_cfg.ALL_MOVIES:
        if movie in feat_cfg.FRIENDS_SEASONS:
            stimuli_root = "../../stimuli/movies/friends"
            season_dir = os.path.join(stimuli_root, f"s{movie}")

        elif movie in feat_cfg.MOVIE10_MOVIES:
            stimuli_root = "../../stimuli/movies/movie10"
            season_dir = os.path.join(stimuli_root, f"{movie}")

        elif movie in feat_cfg.OOD_MOVIES:
            stimuli_root = "../../stimuli/movies/ood"
            season_dir = os.path.join(stimuli_root, f"{movie}")

        episode_paths = sorted(glob.glob(os.path.join(season_dir, "*.mkv")))

        for episode_path in episode_paths:
            extract_swin_features(
                episode_path, tr, model, preprocess,
                device,
                save_dir, movie
            )
