import os
import torch
import random

from torch.utils import data
from transformers import AutoTokenizer, AutoConfig

from .augment import Augmenter

from transformers.utils import is_offline_mode
from transformers.utils.hub import TRANSFORMERS_CACHE
import os
import glob

def load_tokenizer_once(model_name):
    """
    Pure offline load if files exist in cache.
    Only download once if absolutely necessary.
    """

    # 1. Check manually if a tokenizer directory already exists
    cached_dirs = glob.glob(os.path.join(TRANSFORMERS_CACHE, f"models--{model_name.replace('/', '--')}*", "*"))
    if cached_dirs:
        try:
            return AutoTokenizer.from_pretrained(cached_dirs[0], local_files_only=True)
        except:
            pass  # fallback to normal flow below

    # 2. Force Transformers into offline-only mode
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        # Try offline load (no internet calls)
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except:
        # 3. Model really not present → temporarily allow download
        print(f"[INFO] {model_name} not found locally. Downloading once...")

        # Disable offline mode temporarily
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

        tok = AutoTokenizer.from_pretrained(model_name)

        # After down


class DMDataset(data.Dataset):
    """EM dataset"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da=None):

        model_name = lm_mp[lm]

        # Use the safe loader (local first, download once if missing)
        self.tokenizer = load_tokenizer_once(model_name)

        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.size = size

        for line in open(path):
            LL = line.strip().split('\t')
            self.pairs.append(tuple(LL[:-1]))
            self.labels.append(int(LL[-1]))

        self.pairs = self.pairs[:size]
        self.labels = self.labels[:size]
        self.da = da
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        if len(self.pairs[idx]) == 2:
            left = self.pairs[idx][0]
            right = self.pairs[idx][1]

            if self.da is not None:
                left = self.augmenter.augment_sent(left, self.da)
                right = self.augmenter.augment_sent(right, self.da)

            x1 = self.tokenizer.encode(text=left,
                                       max_length=self.max_len,
                                       truncation=True)

            x2 = self.tokenizer.encode(text=right,
                                       max_length=self.max_len,
                                       truncation=True)

            x12 = self.tokenizer.encode(text=left,
                                        text_pair=right,
                                        max_length=self.max_len,
                                        truncation=True)

            return x1, x2, x12, self.labels[idx]

        else:
            left = self.pairs[idx][0]
            if self.da is not None:
                left = self.augmenter.augment_sent(left, self.da)
            x = self.tokenizer.encode(text=left,
                                      max_length=self.max_len,
                                      truncation=True)
            return x, self.labels[idx]


    @staticmethod
    def pad(batch):
        if len(batch[0]) == 4:
            x1, x2, x12, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]

            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]

            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(x12), \
                   torch.LongTensor(y)
        else:
            x1, y = zip(*batch)
            maxlen = max([len(x) for x in x1])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            return torch.LongTensor(x1), torch.LongTensor(y)
