from tqdm.notebook import tqdm
from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale
from moviepy.editor import VideoFileClip
import os
from pathlib import Path
import glob
import numpy as np
import copy
import torch
from feature_config import Config as feat_cfg

device = feat_cfg.DEVICE


def define_frames_transform():
    """Defines the preprocessing pipeline for the video frames. Note that this
    transform is specific to the slow_r50 model."""
    transform = Compose(
                        [
                            UniformTemporalSubsample(32),
                            Lambda(lambda x: x/255.0),
                            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
                            ShortSideScale(size=256),
                            CenterCrop(256)
                        ]
                       )
    return transform


def get_vision_models(device):
    """
    Load a pre-trained SlowFast R101 video model and return both:
      1) a feature-extractor (with the final proj layer removed)
      2) the original full model.

    Parameters
    ----------
    device : torch.device
        The device on which the model will run ('cpu' or 'cuda').

    Returns
    -------
    feature_extractor : torch.nn.Module
        The model up through the penultimate layer (for feature extraction).
    original_model : torch.nn.Module
        The full pre-trained SlowFast R101 model.
    """
    # Load the pretrained model
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "slowfast_r101",
        pretrained=True
    )
    model = model.to(device).eval()

    # Keep a copy of the full model
    original_model = copy.deepcopy(model)

    # Strip off the final projection to create a pure feature extractor
    feature_extractor = model
    feature_extractor.blocks[-1].proj = torch.nn.Identity()

    return feature_extractor, original_model


def extract_slowfast_features(episode_path, tr,
                            transform, device, save_dir, season_num):
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    extracted_features = []
    print(clip.duration)
    
    if len(start_times) > feat_cfg.SKIP_LENGTH:
        return

    for i, start in enumerate(tqdm(start_times, desc=f"Season {season_num} - {os.path.basename(episode_path)}")):
        clip_chunk = clip.subclip(start, start + tr)
        frames = [frame for frame in clip_chunk.iter_frames()]
        frames_array = np.transpose(np.array(frames), (3, 0, 1, 2))
        
        inputs = torch.from_numpy(frames_array).float()
        inputs = transform(inputs).unsqueeze(0).to(device)
        slow_inputs = inputs[:, :, ::4, :, :]  
        



        # 4. Forward pass as a LIST [slow, fast]
        with torch.no_grad():
            ext_feats = feature_extractor([slow_inputs, inputs]) 
        extracted_features.append(np.reshape(ext_feats.cpu().numpy(), -1))

    extracted_features = np.array(extracted_features, dtype='float32')
    print(extracted_features.shape)

    season_folder = os.path.join(save_dir, f"{season_num}")
    os.makedirs(season_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(episode_path))[0] + '_features.npy'
    np.save(os.path.join(season_folder, filename), extracted_features)


def save_slowfast_features():
    transform = define_frames_transform()
    feature_extractor, model = get_vision_models(device)
        
    save_dir = "../final_features/slowfast"
    tr = feat_cfg.TR

    for movie in feat_cfg.ALL_MOVIES:
        if movie in feat_cfg.FRIENDS_SEASONS:
            stimuli_root = "../stimuli/movies/friends"
            season_dir = os.path.join(stimuli_root, f"s{movie}")

        elif movie in feat_cfg.MOVIE10_MOVIES:
            stimuli_root = "../stimuli/movies/movie10"
            season_dir = os.path.join(stimuli_root, f"{movie}")

        elif movie in feat_cfg.OOD_MOVIES:
            stimuli_root = "../stimuli/movies/ood"
            season_dir = os.path.join(stimuli_root, f"{movie}")

        episode_paths = sorted(glob.glob(os.path.join(season_dir, "*.mkv")))

        for episode_path in episode_paths:
            extract_slowfast_features(episode_path, tr,
                                transform, device,
                                save_dir, movie)













stimuli_root = "../stimuli/movies/ood"

for season_num in ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]:
    season_dir = os.path.join(stimuli_root, f"{season_num}")
    episode_paths = sorted(glob.glob(os.path.join(season_dir, "*.mkv")))

    for episode_path in episode_paths:
        extract_visual_features(
            episode_path, tr, transform,
            device,
            save_dir_features, season_num
        )

stimuli_root = "../stimuli/movies/movie10"

for season_num in ["bourne", "figures", "life", "wolf"]:
    season_dir = os.path.join(stimuli_root, f"{season_num}")
    episode_paths = sorted(glob.glob(os.path.join(season_dir, "*.mkv")))

    for episode_path in episode_paths:
        extract_visual_features(
            episode_path, tr, transform,
            device,
            save_dir_features, season_num
        )
        
stimuli_root = "../stimuli/movies/friends"

for season_num in range(1, 8):
    season_dir = os.path.join(stimuli_root, f"s{season_num}")
    episode_paths = sorted(glob.glob(os.path.join(season_dir, "*.mkv")))

    for episode_path in episode_paths:
        extract_visual_features(
            episode_path, tr, transform,
            device,
            save_dir_features, season_num
        )
