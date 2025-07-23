# inference_config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    SUBJECTS = [1,2,3,5]
    ALL_MODALITY_SUBMISSION_MOVIES: list = [7, "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]
    NO_LANGUAGE_SUBMISSION_MOVIES: list = ["chaplin"]
    MOVIES = ALL_MODALITY_SUBMISSION_MOVIES + NO_LANGUAGE_SUBMISSION_MOVIES
    ALL_MODALITY_MODEL_PATHS = "model_checkpoints/all_modality_models"
    NO_LANGUAGE_MODEL_PATHS = "model_checkpoints/no_language_models"
    SUBMISSION_NUMPY_SAVE_DIR = "submission_files"
    DEVICE: str = "cuda"


# Single config object to import elsewhere
config = Config()
