import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data.data_config import Config as data_cfg

from data.utils import pad_or_crop
from data.fmri import load_fmri


class AllModalityDataset(Dataset):
    def __init__(self, subjects, movies, train=None, with_target=None, fixed_length=data_cfg.FIXED_LENGTH):
        self.fixed_length = fixed_length
        self.with_target = with_target
        self.train = train
        self.lag = data_cfg.LAG
        self.subjects = subjects

        if with_target:
            self.fmri = {}
            for subject in subjects:
                fmri = load_fmri(subject=subject)
                fmri = {f"{subject}_{k}": v for k, v in fmri.items()}
                self.fmri.update(fmri)
            fmri_keys = {key.split("_", 1)[1] for key in self.fmri}

        else:
            self.fmri_samples = {}
            for subject in subjects:
                samples_dir = os.path.join(
                                            '../fmri', f'sub-0{subject}', 'target_sample_number',
                                            f'sub-0{subject}_ood_fmri_samples.npy')
                fmri_samples = np.load(samples_dir, allow_pickle=True).item()
                fmri_samples = {f"{subject}_{k}": v for k, v in fmri_samples.items()}
                self.fmri_samples.update(fmri_samples)
            fmri_keys = {key.split("_", 1)[1] for key in self.fmri_samples}

        self.features = {}

        for movies in movies:
            visual_folder1 = f'data/slowfast/{movies}'
            visual_folder2 = f'data/swin/{movies}'
            visual_folder3 = f'data/videomae/{movies}'
            visual_folder4 = f'data/clip/{movies}'

            audio_folder1 = f'data/hubert/{movies}' 
            audio_folder2 = f'data/WavLM/{movies}' 
            audio_folder3 = f'data/clap/{movies}' 

            language_folder1 = f"data/bert/{movies}"
            language_folder2 = f"data/longformer/{movies}"

            movie_fmri_keys = {
                                key
                                for key in fmri_keys
                                if (key.startswith("s") and int(key[2]) == movies)
                                or (key[:-1] == movies)
                                or (key[:-2] == movies)
                                }

            for key in movie_fmri_keys:
                if key in ['mononoke1', 'passepartout1', 'chaplin2', 'planetearth1', 'wot1', 'pulpfiction2', 'passepartout2', 'planetearth2', 'mononoke2', 'wot2', 'chaplin1', 'pulpfiction1']:
                    visual_path1 = visual_folder1 + f"/task-{key}_video_features.npy"
                    visual_path2 = visual_folder2 + f"/task-{key}_video_features.npy"
                    visual_path3 = visual_folder3 + f"/task-{key}_video_features.npy"
                    visual_path4 = visual_folder4 + f"/task-{key}_video_features.npy"

                    audio_path1 =  audio_folder1 + f"/task-{key}_video_features.npy"
                    audio_path2 =  audio_folder2 + f"/task-{key}_video_features.npy"
                    audio_path3 =  audio_folder3 + f"/task-{key}_video_features.npy"

                    language_path1 = language_folder1 + f"/ood_{key}_all_hidden_avg.npy"
                    language_path2 = language_folder1 + f"/ood_{key}_last20_hidden_avg.npy"
                    language_path3 = language_folder2 + f"/ood_{key}_pooler_output.npy"
                    language_path4 = language_folder2 + f"/ood_{key}_current_hidden_avg.npy"

                elif key[0] == "s":
                    visual_path1 =  visual_folder1 + f"/friends_{key}_features.npy"
                    visual_path2 =  visual_folder2 + f"/friends_{key}_features.npy"
                    visual_path3 =  visual_folder3 + f"/friends_{key}_features.npy"
                    visual_path4 =  visual_folder4 + f"/friends_{key}_features.npy"

                    audio_path1 =  audio_folder1 + f"/friends_{key}_features.npy"
                    audio_path2 =  audio_folder2 + f"/friends_{key}_features.npy"
                    audio_path3 =  audio_folder3 + f"/friends_{key}_features.npy"

                    language_path1 = language_folder1 + f"/friends_{key}_all_hidden_avg.npy"
                    language_path2 = language_folder1 + f"/friends_{key}_last20_hidden_avg.npy"
                    language_path3 = language_folder2 + f"/friends_{key}_pooler_output.npy"
                    language_path4 = language_folder2 + f"/friends_{key}_current_hidden_avg.npy"

                else:
                    visual_path1 = visual_folder1 + f"/{key}_features.npy"
                    visual_path2 = visual_folder2 + f"/{key}_features.npy"
                    visual_path3 = visual_folder3 + f"/{key}_features.npy"
                    visual_path4 = visual_folder4 + f"/{key}_features.npy"

                    audio_path1 = audio_folder1 + f"/{key}_features.npy"
                    audio_path2 = audio_folder2 + f"/{key}_features.npy"
                    audio_path3 = audio_folder3 + f"/{key}_features.npy"

                    language_path1 = language_folder1 + f"/movie10_{key}_all_hidden_avg.npy"
                    language_path2 = language_folder1 + f"/movie10_{key}_last20_hidden_avg.npy"
                    language_path3 = language_folder2 + f"/movie10_{key}_pooler_output.npy"
                    language_path4 = language_folder2 + f"/movie10_{key}_current_hidden_avg.npy"

                visual_feat1 = np.load(visual_path1)
                visual_feat2 = np.load(visual_path2)
                visual_feat3 = np.load(visual_path3)
                visual_feat4 = np.load(visual_path4)
                audio_feat1 = np.load(audio_path1)
                audio_feat2 = np.load(audio_path2)
                audio_feat3 = np.load(audio_path3)
                language_feat1 = np.load(language_path1)
                language_feat2 = np.load(language_path2)
                language_feat3 = np.load(language_path3)
                language_feat4 = np.load(language_path4)

                for subject in subjects:
                    if self.with_target:
                        try:
                            T = self.fmri[f"{subject}_{key}"].shape[0]
                        except:
                            continue
                    else:
                        try:
                            T = self.fmri_samples[f"{subject}_{key}"]
                        except:
                            continue
 
                    feats = [
                        visual_feat1, visual_feat2, visual_feat3, visual_feat4,
                        audio_feat1, audio_feat2, audio_feat3,
                        language_feat1,
                        language_feat2,
                        language_feat3,
                        language_feat4,
                            ]

                    padded = [pad_or_crop(feat, T) for feat in feats]

                    # 4) concatenate them along the feature‐dimension
                    combined = np.concatenate(padded, axis=1)  # shape: (T, total_feat_dim)

                    self.features[f"{subject}_{key}"] = combined

        feats_by_mod = [
                        [visual_feat1, visual_feat2, visual_feat3, visual_feat4],
                        [audio_feat1, audio_feat2, audio_feat3],
                        [language_feat1,
                         language_feat2,
                         language_feat3,
                         language_feat4,
                         ]
                       ]

        # 2) concatenate each modality and unpack
        vis_feat_block, aud_feat_block, lang_feat_block = [
            np.concatenate(group, axis=-1)
            for group in feats_by_mod
        ]

        self.modality_dims = [
            vis_feat_block.shape[-1],
            aud_feat_block.shape[-1],
            lang_feat_block.shape[-1]
        ]

        if with_target:
            self.keys = sorted(set(self.features.keys()) & set(self.fmri.keys()))
        else:
            self.keys = sorted(set(self.features.keys()))
        assert self.keys, "No matching keys between features and fMRI data"

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        subject = key.split("_")[0]
        feats = self.features[key]  # (T, F)

        if self.with_target:
            fmri = self.fmri[key]       # (T, V)
            fmri = fmri[self.lag:, :]
            T_fmri = fmri.shape[0]

        else:
            T_fmri = self.fmri_samples[key]  - (self.lag)

        feats = feats[:T_fmri, :]

        mask = torch.zeros(self.fixed_length, dtype=torch.float32)
        mask[:feats.shape[0]] = 1.0

        if feats.shape[0] < self.fixed_length:
            pad = np.zeros((self.fixed_length - feats.shape[0], feats.shape[1]), dtype=feats.dtype)
            feats = np.vstack([feats, pad])
            if self.with_target:
                pad = np.zeros((self.fixed_length - fmri.shape[0], fmri.shape[1]), dtype=fmri.dtype)
                fmri = np.vstack([fmri, pad])
        else:
            hata

        feats = torch.from_numpy(feats.T).float()

        if self.with_target:
            fmri = torch.from_numpy(fmri.T).float()

            return feats, int(subject), fmri, key, mask
        else:
            return feats, int(subject), key, mask
        
    def get_modality_dims(self):
        return self.modality_dims
