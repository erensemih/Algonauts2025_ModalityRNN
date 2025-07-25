# training_config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Data identifiers
    #MOVIES: tuple = (1, 2, 3, 4, 5, 6, "wolf", "bourne", "figures", "life")
    MOVIES: tuple = ("life",)
    SUBJECTS: tuple = (1, 2, 3, 5)

    # Keys to remove from train dataset
    REMOVE_KEYS: tuple = (
        "1_s06e18b",
        "1_s06e19a",
        "1_s06e19b",
        "2_s05e13a",
        "2_bourne01",
    )
    # Training hyperparameters
    BATCH_SIZE: int = 4
    LR: float = 1e-3
    NUM_EPOCHS: int = 5
    CHECKPOINT_EPOCHS: tuple = (3, 4)

    # Model architecture
    HIDDEN_SIZE: int = 768
    NUM_LAYERS: int = 1
    POST_HIDDEN: int = 768
    POST_LAYERS: int = 1
    DROPOUT: float = 0.0
    MODALITY_BIDIRECTIONAL: bool = True
    POST_BIDIRECTIONAL: bool = True
    OUT_CH: int = 1000

    # Random seeds to try
    SEEDS: tuple = tuple(range(1))

    DEVICE: str = "cuda"
    ALL_MODALITY_MODEL_PATHS: str = "model_checkpoints/all_modality_models"
    NO_LANGUAGE_MODEL_PATHS: str = "model_checkpoints/no_language_models"

# Single config object to import elsewhere
config = Config()
