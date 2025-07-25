import os
import glob
import numpy as np
import torch
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from features.feature_config import Config as feat_cfg

device = feat_cfg.DEVICE


def sample_frames(frames, num_frames):
    n = len(frames)
    idxs = np.linspace(0, n - 1, num_frames, dtype=int)
    return [frames[i] for i in idxs]


def get_clip_model(device):
    """
    Load OpenAI's CLIP ViT-B/32 model and processor.
    Returns:
      - processor: handles resizing & normalization
      - model: CLIPModel on `device`
    """
    model_id = "openai/clip-vit-large-patch14"    
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return processor, model


def extract_clip_features(episode_path, chunk_duration, processor,
                          model, device, save_dir, season_name):
    """
    Split video into `chunk_duration`-second clips, sample frames,
    run CLIP image encoder on each frame, average to one vector per chunk,
    and save.
    """
    clip = VideoFileClip(episode_path)
    # start times for each chunk
    starts = np.arange(0, clip.duration, chunk_duration)[:-1]
    if len(starts) > feat_cfg.SKIP_LENGTH:
        return

    num_frames = 8  # how many frames to sample per chunk
    all_feats = []

    for start in tqdm(starts, desc=f"{season_name} – {os.path.basename(episode_path)}"):
        sub = clip.subclip(start, start + chunk_duration)
        # extract raw frames as RGB uint8 arrays
        frames = [f for f in sub.iter_frames()]
        # uniformly sample
        frames = sample_frames(frames, num_frames)

        inputs = processor(images=frames, return_tensors="pt").to(device)

        with torch.no_grad():
            img_feats = model.get_image_features(**inputs) 
        img_feats = img_feats.cpu().numpy()

        chunk_feat = img_feats.mean(axis=0)
        all_feats.append(chunk_feat.astype('float32'))

    all_feats = np.stack(all_feats, axis=0)
    season_dir = Path(save_dir) / season_name
    season_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(episode_path).stem + "_features.npy"
    np.save(season_dir / fname, all_feats)


def save_clip_features():
    processor, clip_model = get_clip_model(device)

    tr = feat_cfg.TR
    save_dir = "data/clip"
    os.makedirs(save_dir, exist_ok=True)

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
            extract_clip_features(episode_path, tr,
                                processor, clip_model, device,
                                save_dir, movie)
