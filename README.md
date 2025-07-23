## Model Architecture
![Model Architecture](figures/pipeline-1.png)

## Requirements

Before running the pipeline, please install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

After installing the requirements, you can run the main script to execute the entire pipeline:

```bash
python main.py
```

## Pipeline Details

When you run main.py, it will first extract visual, audio, and language features for the movies defined in the ALL_MOVIES parameter in src/features/feature_config.py. By default for quick trials, this includes only 'Friends' Seasons 6; to process all movies, update the ALL_MOVIES list accordingly.

To ensure all movies with extracted features are included in model training, set the MOVIES parameter in src/training/training_config.py to your desired list of movies.

Once these configurations are set, main.py will proceed through feature extraction followed by model training and evaluation.