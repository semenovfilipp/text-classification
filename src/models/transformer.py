import numpy as np
from torch.utils.data import Dataset

from src.utils.metrics import compute_metrics


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def positive_proba(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e[:, 1] / e.sum(axis=1)


def hf_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_metrics(labels, positive_proba(logits))
