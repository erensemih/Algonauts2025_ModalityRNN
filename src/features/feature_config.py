# feature_config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:

    DEVICE: str = "cuda"
    SKIP_LENGTH: int = 1000
    TR: float = 1.49
    FRIENDS_SEASONS: list = [1, 2, 3, 4, 5, 6, 7]
    MOVIE10_MOVIES: list = ["wolf", "bourne", "life", "figures"]
    OOD_MOVIES: list = ["chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]
    ALL_MOVIES: list = FRIENDS_SEASONS + MOVIE10_MOVIES + OOD_MOVIES
    NO_LANGUAGE_MOVIES: list = ["chaplin"]

    BERT_WINDOW: int = 256
    LONGFORMER_WINDOW: int = 4000

# Single config object to import elsewhere
config = Config()
