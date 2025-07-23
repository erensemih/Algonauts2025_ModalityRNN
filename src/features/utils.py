from features.audio.clap import save_clap_features
from features.audio.hubert import save_hubert_features
from features.audio.WavLM import save_wavlm_features
from features.vision.clip import save_clip_features
from features.vision.slowfast import save_slowfast_features
from features.vision.swin import save_swin_features
from features.vision.videomae import save_videomae_features
from features.language.bert import save_bert_features
from features.language.longformer import save_longformer_features


def extract_visual_features():
    save_slowfast_features()
    save_clip_features()
    save_swin_features()
    save_videomae_features()


def extract_audio_features():
    save_clap_features()
    save_hubert_features()
    save_wavlm_features()


def extract_language_features():
    save_bert_features()
    save_longformer_features()
