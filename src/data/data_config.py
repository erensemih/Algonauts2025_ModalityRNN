# data_config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:

    LAG: int = 3
    FIXED_LENGTH: int = 600


# Single config object to import elsewhere
config = Config()
