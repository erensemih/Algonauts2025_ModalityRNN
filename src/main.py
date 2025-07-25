from data.all_modality_dataset import AllModalityDataset
from data.no_language_dataset import NoLanguageDataset
from training.training_config import Config as train_cfg
from inference.inference_config import Config as inf_cfg
from inference.predict_test import predict_movies, save_submission
from training.train_utils.utils import train_models, get_config_for_seed
from features.utils import extract_visual_features, extract_audio_features, extract_language_features


extract_visual_features()
extract_audio_features()
extract_language_features()

all_modality_train_dataset = AllModalityDataset(train_cfg.SUBJECTS, movies=train_cfg.MOVIES,  with_target=True, train=True)
no_language_train_dataset = NoLanguageDataset(train_cfg.SUBJECTS, movies=train_cfg.MOVIES,  with_target=True, train=True)
    
for remove_key in train_cfg.REMOVE_KEYS:
    try:
        all_modality_train_dataset.keys.remove(remove_key)
        no_language_train_dataset.keys.remove(remove_key)

    except ValueError:
        pass

for seed in train_cfg.SEEDS:
    print("SEED : ", seed)
    criterion, model_type = get_config_for_seed(seed)
    train_models(all_modality_train_dataset, no_language_train_dataset, criterion, model_type, seed)

submission = {}
for subject in inf_cfg.SUBJECTS:
    submission[f"sub-0{subject}"] = (predict_movies(subject=subject))


save_submission(submission)
