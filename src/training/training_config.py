# training_config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Data identifiers
    MOVIES: list = [i for i in range(1, 7)] + ["wolf", "bourne", "figures", "life"]
    SUBJECTS: list = [1, 2, 3, 5]

    # Keys to remove from train dataset
    REMOVE_KEYS: list = [
        "1_s06e18b",
        "1_s06e19a",
        "1_s06e19b",
        "2_s05e13a",
        "2_bourne01",
    ]

    # Training hyperparameters
    BATCH_SIZE: int = 4
    LR: float = 1e-3
    NUM_EPOCHS: int = 5
    CHECKPOINT_EPOCHS: list = [3, 4]  # epochs at which to snapshot the model

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
    SEEDS: list = list(range(10))

    DEVICE: str = "cuda"
    CHECKPOINT_DIR: str = "model_checkpoints"

# Single config object to import elsewhere
config = Config()
