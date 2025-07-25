# inference_config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    SUBJECTS: tuple = (1, 2, 3, 5)
    ALL_MODALITY_SUBMISSION_MOVIES: tuple = ("mononoke", "passepartout", "planetearth", "pulpfiction", "wot")
    NO_LANGUAGE_SUBMISSION_MOVIES: tuple = ("chaplin",)
    ALL_MODALITY_MODEL_PATHS: str = "model_checkpoints/all_modality_models"
    NO_LANGUAGE_MODEL_PATHS: str = "model_checkpoints/no_language_models"
    SUBMISSION_NUMPY_SAVE_DIR: str = "submission_files"
    DEVICE: str = "cuda"


# Single config object to import elsewhere
config = Config()
