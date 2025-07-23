from torchaudio.pipelines import HUBERT_LARGE
import os
import glob
import numpy as np
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from feature_config import Config as feat_cfg

device = feat_cfg.DEVICE

bundle = HUBERT_LARGE
model = bundle.get_model().to(device)
model.eval()

HUBERT_SAMPLE_RATE = bundle.sample_rate  # should be 16000


def extract_audio_features(
    episode_path: str,
    tr: float,
    save_dir: str,
    season_num,
):
    """
    Splits the episode into tr-second chunks, extracts audio,
    resamples to 16 kHz, feeds through HuBERT-Base, and saves
    a (num_chunks x 768) float32 .npy file.
    """
    # Load video & compute chunk start times
    clip = VideoFileClip(episode_path)
    total_duration = clip.duration
    start_times = [t for t in np.arange(0, total_duration, tr) if t + tr <= total_duration]
    if len(start_times) > feat_cfg.SKIP_LENGTH:
        return

    print(f"{os.path.basename(episode_path)}: {total_duration:.1f}s → {len(start_times)} chunks")

    all_embeddings = []
    for start in tqdm(start_times, desc=f"Season {season_num}"):
        # Extract tr-second audio clip
        audio_clip = clip.audio.subclip(start, start + tr)
        wav_np = audio_clip.to_soundarray(fps=HUBERT_SAMPLE_RATE)
        # Stereo → mono
        if wav_np.ndim == 2 and wav_np.shape[1] > 1:
            wav_np = wav_np.mean(axis=1, keepdims=True)
        wav_np = wav_np[:, 0]

        audio_tensor = torch.from_numpy(wav_np).float().to(device).unsqueeze(0)

        with torch.no_grad():
            hubert_out = model(audio_tensor)
            last_layer = hubert_out[0] 

        embedding = last_layer.mean(dim=1).squeeze(0).cpu().numpy()
        all_embeddings.append(embedding)

    features = np.vstack(all_embeddings).astype('float32')
    print(" → features shape:", features.shape)

    season_folder = os.path.join(save_dir, str(season_num))
    os.makedirs(season_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(episode_path))[0]
    out_path = os.path.join(season_folder, f"{base_name}_features.npy")
    np.save(out_path, features)

def save_hubert_features():
    tr = feat_cfg.TR
    save_dir_features_root = "../final_features/hubert"

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
            save_dir_features = os.path.join(save_dir_features_root, f"{movie}")
            extract_audio_features(
                episode_path=episode_path,
                tr=tr,
                save_dir=save_dir_features_root,
                season_num=movie
            )
