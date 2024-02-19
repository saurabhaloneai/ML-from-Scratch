import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from pathlib import Path 


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
        
def get_tokenizer(config, ds , lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizers = Tokenizer(WordLevel(unk_token=[config['unk_token']]))    