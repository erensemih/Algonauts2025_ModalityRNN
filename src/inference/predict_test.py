import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import zipfile

from data.all_modality_dataset import AllModalityDataset
from data.no_language_dataset import NoLanguageDataset
from inference_config import Config as inf_cfg
from training.models import LSTMRegressor
from training.train_utils.utils import init_model


def build_dataloader(dataset_cls, subject: str, movies: list, batch_size: int = 1) -> DataLoader:
    """
    Instantiate dataset and wrap it in a DataLoader.

    Args:
        dataset_cls: Dataset class (AllModalityDataset or NoLanguageDataset).
        subject: Subject identifier.
        seasons: List of movie seasons.
        batch_size: Batch size for DataLoader.

    Returns:
        DataLoader for inference (no shuffling).
    """
    dataset = dataset_cls(subjects=[subject], seasons=movies, with_target=False, train=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def predict_on_loader(
    dataloader: DataLoader,
    model: LSTMRegressor,
    model_paths: list,
    lag: int,
    device: torch.device,
    key_parser=None
) -> dict:
    """
    Run inference over a dataloader by averaging predictions from multiple checkpoints.

    Args:
        dataloader: DataLoader for test data.
        model: Initialized LSTMRegressor.
        model_paths: List of checkpoint paths.
        lag: Number of zero frames to prepend.
        device: Device for computation.
        key_parser: Optional function to extract unique key from epi_key string.

    Returns:
        Dictionary mapping episode keys to prediction arrays.
    """
    submission = {}
    for feat, subject, epi_key, mask in dataloader:
        raw_key = epi_key[0]
        key = key_parser(raw_key) if key_parser else raw_key

        feat = feat.to(device)
        mask = mask.squeeze(0).bool()
        preds = None

        # Accumulate outputs over all checkpoints
        for ckpt_path in model_paths:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
            model.eval()

            with torch.no_grad():
                out = model(feat, subject)

            arr = out.cpu().permute(0, 2, 1).numpy()[0].astype(np.float32)
            arr = arr[mask]
            preds = arr if preds is None else preds + arr

        # Prepend zeros for lag frames
        zeros = np.zeros((lag, preds.shape[1]), dtype=preds.dtype)
        preds_full = np.vstack((zeros, preds))

        # Validate length
        expected_len = dataloader.dataset.fmri_samples[raw_key]
        assert preds_full.shape[0] == expected_len

        submission[key] = preds_full

    return submission


def predict_movies(subject: str) -> dict:
    """
    Generate predictions for both all-modality and no-language models.

    Args:
        subject: Subject identifier.
        movies: List of movie seasons to run inference on.

    Returns:
        Dictionary mapping episode identifiers to prediction arrays.
    """
    # Prepare dataloaders
    all_mod_loader = build_dataloader(AllModalityDataset, subject, inf_cfg.ALL_MODALITY_SUBMISSION_MOVIES)
    no_lang_loader = build_dataloader(NoLanguageDataset, subject, inf_cfg.NO_LANGUAGE_SUBMISSION_MOVIES)

    # Initialize models
    all_mod_model = init_model(all_mod_loader.dataset.get_modality_dims())
    no_lang_model = init_model(no_lang_loader.dataset.get_modality_dims())

    # Helper to extract numeric key
    key_parser = lambda k: k.split("_")[1]

    # Run inference
    subject_predictions = {}
    subject_predictions.update(
        predict_on_loader(
            all_mod_loader,
            all_mod_model,
            inf_cfg.ALL_MODALITY_MODEL_PATHS,
            all_mod_loader.dataset.lag,
            inf_cfg.DEVICE,
            key_parser
        )
    )
    subject_predictions.update(
        predict_on_loader(
            no_lang_loader,
            no_lang_model,
            inf_cfg.NO_LANGUAGE_MODEL_PATHS,
            no_lang_loader.dataset.lag,
            inf_cfg.DEVICE,
            key_parser
        )
    )

    return subject_predictions


def save_submission(submission):
    # Save the predicted fMRI dictionary as a .npy file
    output_file = inf_cfg.SUBMISSION_NUMPY_SAVE_DIR + "/submission.npy"
    np.save(output_file, submission)
    print(f"Formatted predictions saved to: {output_file}")

    # Zip the saved file for submission
    zip_file = inf_cfg.SUBMISSION_NUMPY_SAVE_DIR + "/submission.zip"
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))