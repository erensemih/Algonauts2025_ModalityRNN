import os
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from transformers import ClapProcessor, ClapAudioModelWithProjection
from tqdm import tqdm
import glob

from features.feature_config import Config as feat_cfg

device = feat_cfg.DEVICE


def extract_audio_features_clap(
    episode_path: str,
    tr: float,
    device: str,
    save_dir_features: str,
    model_name: str = "laion/clap-htsat-fused",
) -> np.ndarray:

    clip = VideoFileClip(episode_path)
    start_times = [t for t in np.arange(0, clip.duration, tr)][:-1]
    if len(start_times) > feat_cfg.SKIP_LENGTH:
        return
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapAudioModelWithProjection.from_pretrained(model_name).to(device)
    model.eval()

    target_sr = processor.feature_extractor.sampling_rate
    audio_features = []

    with tqdm(total=len(start_times), desc="Extracting CLAP audio embeddings") as pbar:
        for start in start_times:
            audio_chunk = clip.subclip(start, start + tr).audio

            # build explicit time array (so MoviePy doesn't chunk under the hood)
            tt = np.arange(0, tr, 1/target_sr)
            audio_array = audio_chunk.to_soundarray(tt=tt)
            mono_audio = np.mean(audio_array, axis=1)
            inputs = processor(audios=mono_audio, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            emb = outputs.audio_embeds.cpu().squeeze(0).numpy()
            audio_features.append(emb)
            pbar.update(1)

    audio_features = np.vstack(audio_features).astype("float32")

    os.makedirs(save_dir_features, exist_ok=True)
    base = os.path.splitext(os.path.basename(episode_path))[0]
    save_path = os.path.join(save_dir_features, f"{base}_features.npy")
    np.save(save_path, audio_features)


def save_clap_features():
    save_dir_features_root = "data/clap"
    os.makedirs(save_dir_features_root, exist_ok=True)

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
            save_dir_features = os.path.join(save_dir_features_root, f"{movie}")
            extract_audio_features_clap(
                episode_path=episode_path,
                tr=tr,
                device=device,
                save_dir_features=save_dir_features
            )
