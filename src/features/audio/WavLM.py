import os
import numpy as np
import torch
import torchaudio
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
import glob
from features.feature_config import Config as feat_cfg

device = feat_cfg.DEVICE
bundle = torchaudio.pipelines.WAVLM_LARGE
model = bundle.get_model().to(device)
SAMPLE_RATE = bundle.sample_rate


def extract_audio_features(
    episode_path: str,
    tr: float,
    save_dir: str,
    season_num: int,
):
    """
    Splits the episode into tr‐second chunks, extracts audio,
    resamples to 16kHz, feeds through Wav2Vec2, and returns a
    (num_chunks x embedding_dim) float32 array.
    """

    # 2.1) Load the video with MoviePy:
    clip = VideoFileClip(episode_path)
    total_duration = clip.duration
    # Create start times [0, tr, 2*tr, ...] but only keep those where start + tr <= total_duration
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    
    if len(start_times) > 1000:
        print(f"Skipping {os.path.basename(episode_path)}: too many ({len(start_times)}) chunks")
        return

    all_embeddings = []  # will hold one 1D vector per chunk

    for i, start in enumerate(tqdm(start_times, desc=f"Season {season_num} – {os.path.basename(episode_path)}")):
        # ------------------------------------------------------------
        # 2.2) Extract exactly tr seconds of audio from the video:
        # ------------------------------------------------------------
        audio_clip = clip.audio.subclip(start, start + tr)

        # 2.3) Convert to NumPy at arbitrary frame rate (we’ll pick W2V_SAMPLE_RATE now, 
        #      but MoviePy will do an internal conversion if needed):
        wav_np = audio_clip.to_soundarray(fps=SAMPLE_RATE)  # shape (n_samples, n_channels)
        # If it’s stereo (n_channels=2), convert to mono by averaging:
        if wav_np.ndim == 2 and wav_np.shape[1] > 1:
            wav_np = wav_np.mean(axis=1, keepdims=True)  # (n_samples, 1)
        # Now wav_np is (n_samples, 1). We only need a 1-D float array for W2V:
        wav_np = wav_np[:, 0]  # shape (n_samples,)
        audio_tensor = torch.from_numpy(wav_np).float().to(device).unsqueeze(0)

        # 2.5) Forward pass through Wav2Vec2 under no_grad:
        with torch.no_grad():
            w2v_out = model(audio_tensor)  
            last_layer = w2v_out[0] 
        embeddings = last_layer.mean(dim=1)  # shape (1, 768)
        embeddings_np = embeddings.squeeze(0).cpu().numpy()  # shape (768,)
        all_embeddings.append(embeddings_np)
    audio_features = np.vstack(all_embeddings).astype('float32')
    season_folder = os.path.join(save_dir, f"{season_num}")
    os.makedirs(season_folder, exist_ok=True)
    fname = os.path.splitext(os.path.basename(episode_path))[0] + "_features.npy"
    np.save(os.path.join(season_folder, fname), audio_features)


def save_wavlm_features():
    save_dir_features_root = "data/WavLM"
    os.makedirs(save_dir_features_root, exist_ok=True)
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
            save_dir_features = os.path.join(save_dir_features_root, f"{movie}")
            extract_audio_features(
                episode_path=episode_path,
                tr=tr,
                save_dir=save_dir_features_root,
                season_num=movie
            )
