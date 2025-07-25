import os
import glob
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from features.feature_config import Config as feat_cfg

device = feat_cfg.DEVICE

def get_language_model(device):
    """
    Load a pre-trained bert-base-uncased language model and its corresponding
    tokenizer.

    Parameters
    ----------
    device : torch.device
        Device on which the model will run (e.g., 'cpu' or 'cuda').

    Returns
    -------
    model : object
        Pre-trained language model.
    tokenizer : object
        Tokenizer corresponding to the language model.

    """
    model_name = 'allenai/longformer-base-4096'
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=False,
        return_dict=True
    ).to(device)
    model.eval()

    return model, tokenizer


def extract_language_features(
    episode_path,
    num_used_tokens,
    model,
    tokenizer,
    save_dir,
    layer_idx=-1,
    token_buffer=None
):
    """
    Computes per-TR features:
      - pooler_output: the [CLS] token embedding
      - all_hidden_avg: average over all tokens in context
      - current_hidden_avg: average over tokens in current TR
      - last10_hidden_avg: average over the last 10 tokens in context
      - last20_hidden_avg: average over the last 20 tokens in context
      - silence_flag: 1 if current TR has no tokens

    Saves five .npy files per episode.
    """

    # Load and preprocess
    df = pd.read_csv(episode_path, sep='\t')
    df['text_per_tr'] = df['text_per_tr'].fillna('').astype(str)

    tokens = [] if token_buffer is None else token_buffer.copy()

    pooler_outputs    = []
    all_hidden_list   = []
    curr_hidden_list  = []
    last10_list       = []
    last20_list       = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text_per_tr']
        new_ids = tokenizer.encode(text, add_special_tokens=False)
        n_curr = len(new_ids)

        # update rolling context
        tokens.extend(new_ids)
        tokens = tokens[-num_used_tokens:]

        # build inputs
        input_ids = [tokenizer.cls_token_id] + tokens + [tokenizer.sep_token_id]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_tensor, output_hidden_states=True)
        hs = outputs.hidden_states[layer_idx].squeeze(0)  # [seq_len, hidden_size]

        # 1) pooler
        pooler_outputs.append(outputs.pooler_output[0].cpu().numpy())

        # 2) all-context average (exclude CLS/SEP)
        if hs.size(0) > 2:
            all_avg = hs[1:-1].mean(dim=0)
        else:
            all_avg = hs[0]
        all_hidden_list.append(all_avg.cpu().numpy())

        # 3) current-TR average
        if n_curr > 0:
            cur_states = hs[-(1 + n_curr):-1]
            curr_avg = cur_states.mean(dim=0)
        else:
            curr_avg = torch.zeros(hs.size(1), device=hs.device)
        curr_hidden_list.append(curr_avg.cpu().numpy())

        # 4) last-10 tokens average
        num = 10
        if hs.size(0) > num + 2:
            last10_states = hs[-(1 + num):-1]
            last10_avg = last10_states.mean(dim=0)
        elif hs.size(0) > 2:
            # fewer than 10 tokens: average all tokens
            last10_avg = hs[1:-1].mean(dim=0)
        else:
            last10_avg = torch.zeros(hs.size(1), device=hs.device)
        last10_list.append(last10_avg.cpu().numpy())

        # 5) last-20 tokens average
        num = 20
        if hs.size(0) > num + 2:
            last20_states = hs[-(1 + num):-1]
            last20_avg = last20_states.mean(dim=0)
        elif hs.size(0) > 2:
            last20_avg = hs[1:-1].mean(dim=0)
        else:
            last20_avg = torch.zeros(hs.size(1), device=hs.device)
        last20_list.append(last20_avg.cpu().numpy())

    # Stack and save
    episode_name = os.path.splitext(os.path.basename(episode_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"{episode_name}_pooler_output.npy"),
            np.stack(pooler_outputs, axis=0))
    np.save(os.path.join(save_dir, f"{episode_name}_all_hidden_avg.npy"),
            np.stack(all_hidden_list, axis=0))
    np.save(os.path.join(save_dir, f"{episode_name}_current_hidden_avg.npy"),
            np.stack(curr_hidden_list, axis=0))
    np.save(os.path.join(save_dir, f"{episode_name}_last10_hidden_avg.npy"),
            np.stack(last10_list, axis=0))
    np.save(os.path.join(save_dir, f"{episode_name}_last20_hidden_avg.npy"),
            np.stack(last20_list, axis=0))

    return tokens


def save_longformer_features():
    model, tokenizer = get_language_model(device)
    tokenizer.truncation_side = "left"

    for movie in feat_cfg.ALL_MOVIES:
        if movie in feat_cfg.NO_LANGUAGE_MOVIES:
            continue

        elif movie in feat_cfg.FRIENDS_SEASONS:
            stimuli_root = "../../stimuli/transcripts/friends"
            season_dir = os.path.join(stimuli_root, f"s{movie}")

        elif movie in feat_cfg.MOVIE10_MOVIES:
            stimuli_root = "../../stimuli/transcripts/movie10"
            season_dir = os.path.join(stimuli_root, f"{movie}")

        elif movie in feat_cfg.OOD_MOVIES:
            stimuli_root = "../../stimuli/transcripts/ood"
            season_dir = os.path.join(stimuli_root, f"{movie}")
 
        save_dir = f"data/longformer/{movie}"
        os.makedirs(save_dir, exist_ok=True)
        episode_paths = sorted(glob.glob(os.path.join(season_dir, "*.tsv")))

        token_buffer = []
        for episode_path in episode_paths:
            token_buffer = extract_language_features(episode_path, feat_cfg.LONGFORMER_WINDOW, model,
                                                        tokenizer, save_dir, layer_idx = -1, 
                                                        token_buffer = token_buffer)
