from training.models import LSTMRegressor
from training.training_config import Config as train_cfg
from torch.utils.data import DataLoader
import torch
import os
from training.train_utils.metrics import PearsonCorrLoss
import torch.nn as nn
from training.trainer import train_one_epoch
from torch.optim import Adam
import copy


def get_config_for_seed(seed: int):
    """
    Given an integer seed, returns a tuple (criterion, model_type) according
    to the following mapping:
      0  <= seed < 20 : PearsonCorrLoss(), 'lstm_lstm'
      20 <= seed < 40 : PearsonCorrLoss(), 'lstm_gru'
      40 <= seed < 60 : nn.MSELoss(),      'lstm_lstm'
    Raises ValueError if seed is outside [0,60).
    """
    if not (0 <= seed < 60):
        raise ValueError(f"seed must be in [0,60), got {seed}")

    if seed < 20:
        return PearsonCorrLoss(), "lstm_lstm"
    elif seed < 40:
        return PearsonCorrLoss(), "lstm_gru"
    elif seed < 60:
        return nn.MSELoss(), "lstm_lstm"
    else:
        return nn.MSELoss(), "lstm_gru"
    

def init_model(dataset, model_type) -> LSTMRegressor:
    """
    Create an LSTMRegressor and load the first checkpoint to initialize weights.

    Args:
        dataset: Dataset instance to infer input dims.
        model_paths: List of paths to model checkpoints.
        modality_bidir: Whether to use bidirectional LSTM on modalities.
        post_bidir: Whether to use bidirectional LSTM on post layers.

    Returns:
        LSTMRegressor ready for inference.
    """
    model = LSTMRegressor(
        modality_dims=dataset.get_modality_dims(),
        hidden_size=train_cfg.HIDDEN_SIZE,
        num_layers=train_cfg.NUM_LAYERS,
        post_hidden=train_cfg.HIDDEN_SIZE,
        post_layers=train_cfg.POST_LAYERS,
        out_features=train_cfg.OUT_CH,
        dropout=train_cfg.DROPOUT,
        model_type=model_type
    ).to(train_cfg.DEVICE)

    return model


def train_models(all_modality_train_dataset, no_language_train_dataset, criterion, model_type, seed):

    os.makedirs(train_cfg.ALL_MODALITY_MODEL_PATHS, exist_ok=True)
    os.makedirs(train_cfg.NO_LANGUAGE_MODEL_PATHS, exist_ok=True)

    all_modality_train_dataloader = DataLoader(all_modality_train_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
    no_language_train_dataloader = DataLoader(no_language_train_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)

    all_modality_model = init_model(all_modality_train_dataset, model_type)
    no_language_model = init_model(no_language_train_dataset, model_type)

    all_modality_optimizer = Adam(all_modality_model.parameters(), lr=train_cfg.LR)
    no_language_optimizer = Adam(no_language_model.parameters(), lr=train_cfg.LR)

    for epoch in range(train_cfg.NUM_EPOCHS):
        all_modality_model = train_one_epoch(all_modality_model, all_modality_optimizer, criterion, all_modality_train_dataloader)
        no_language_model = train_one_epoch(no_language_model, no_language_optimizer, criterion, no_language_train_dataloader)
        if epoch in train_cfg.CHECKPOINT_EPOCHS:
            all_modality_model_state = copy.deepcopy(all_modality_model.state_dict())
            no_language_model_state = copy.deepcopy(no_language_model.state_dict())

            all_modality_ckpt = os.path.join(
                train_cfg.ALL_MODALITY_MODEL_PATHS,
                f"all_modality_epoch_{seed}_{epoch}.mdl"
            )
            no_language_ckpt = os.path.join(
                train_cfg.ALL_MODALITY_MODEL_PATHS,
                f"no_language_epoch_{seed}_{epoch}.mdl"
            )

            torch.save(all_modality_model_state, all_modality_ckpt)
            torch.save(no_language_model_state, no_language_ckpt)