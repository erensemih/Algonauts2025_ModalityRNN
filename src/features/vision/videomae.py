import os
import glob
import numpy as np
from tqdm.notebook import tqdm
from moviepy.editor import VideoFileClip
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel
from features.feature_config import Config as feat_cfg

device = feat_cfg.DEVICE


def sample_frames(frames, num_frames):
    n = len(frames)
    idxs = np.linspace(0, n - 1, num_frames, dtype=int)
    return [frames[i] for i in idxs]


def get_vision_model(device):

    # Load the model
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    # 2. Load the base VideoMAE (no classification head)
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
    model.eval()
    return model, processor


def extract_videomae_features(episode_path, tr, processor, feature_extractor,
                            device, save_dir, season_num):
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    extracted_features = []
    
    if len(start_times) > feat_cfg.SKIP_LENGTH:
        print(f"Skipping {os.path.basename(episode_path)}: too many ({len(start_times)}) chunks")
        return
    
    num_frames = feature_extractor.config.num_frames  # typically 8


    for i, start in enumerate(tqdm(start_times, desc=f"Season {season_num} - {os.path.basename(episode_path)}")):
        clip_chunk = clip.subclip(start, start + tr)
        frames = [frame for frame in clip_chunk.iter_frames()]

        frames = sample_frames(frames, num_frames)

        inputs = processor(frames, return_tensors="pt").to(device)

        with torch.no_grad():
            ext_feats = feature_extractor(**inputs)

        # `outputs.last_hidden_state` has shape:
        #   (batch_size, num_frames × num_patches, hidden_size)
        ext_feats = ext_feats.last_hidden_state.cpu().numpy()
        ext_feats = np.mean(ext_feats, axis=1)[0]
        
        extracted_features.append(ext_feats)
        
    extracted_features = np.array(extracted_features, dtype='float32')

    season_folder = os.path.join(save_dir, f"{season_num}")
    os.makedirs(season_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(episode_path))[0] + '_features.npy'
    np.save(os.path.join(season_folder, filename), extracted_features)


def save_videomae_features():
    model, processor = get_vision_model(device)
    save_dir = "data/videomae"
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
            extract_videomae_features( 
                episode_path, tr, processor, model,
                device,
                save_dir, movie
            )
