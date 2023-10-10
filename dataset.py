import random
import pandas as pd

from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import torch


class MindDataset(Dataset):
    def __init__(
        self,
        file_path,
        news_dict,
        vocab,
        title_size,
        max_his_size,
        mode = 'train',
    ):
        self.file_path = file_path
        self.news_dict = news_dict
        self.vocab = vocab
        self.title_size = title_size
        self.max_his_size = max_his_size
        self.mode = mode

        self.samples = []
        self.impid2idx = {}
        self.pad_id = 0
        self.unk_id = len(vocab) + 1

        self.gene_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def imps_len(self):
        return len(self.impid2idx)

    def gene_samples(self):
        """
        Generate samples from impressions
        """
        column_names = ['impid', 'uid', 'time', 'history', 'imps']
        raw_data = pd.read_csv(
            self.file_path, sep='\t', 
            header=None, 
            names=column_names,
        )
        raw_data['history'] = raw_data['history'].fillna('')
        idx = 0
        for _, row in tqdm(raw_data.iterrows()):
            history = row['history'].split()
            imps = row['imps'].split()
            idx_list = []
            for imp in imps:
                # Hint 4: Class Imbalance. Too many negative samples!
                if self.mode == 'train':
                    imp = imp.split('-')
                    self.samples.append({
                        'impid': row['impid'], 'history': history, 
                        'imp': imp[0], 'label': imp[1]
                    })
                elif self.mode == 'test':
                    self.samples.append({
                        'impid': row['impid'], 'history': history, 
                        'imp': imp
                    })
                idx_list.append(idx)
                idx += 1
            self.impid2idx[row['impid']] = idx_list

    def train_val_split(self, val_imps_len):
        """ 
        Split dataset by impressions
        """
        if self.mode == 'test':
            return
        
        val_imps = random.sample(self.impid2idx.keys(), val_imps_len)
        val_imps = set(val_imps)
        train_indices = []
        val_indices = []
        for impid, idx in self.impid2idx.items():
            if impid in val_imps:
                val_indices.extend(idx)
            else:
                train_indices.extend(idx)
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        return train_dataset, val_dataset

    def encode(self, tokens, max_length):
        """
        Converts a sequence of tokens in a sequence of ids, using the vocabulary.
        """
        ids = []
        for token in tokens[:max_length]:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.unk_id)
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids.extend([self.pad_id] * pad_len)
        return ids
    
    def collate_fn(self, batch):
        batch_impid = [x['impid'] for x in batch]
        batch_history = [x['history'] for x in batch]
        batch_imp = [x['imp'] for x in batch]
        
        for i, history in enumerate(batch_history):
            if len(history) == 0:
                history = [[self.pad_id] * self.title_size]
            else:
                history = history[-self.max_his_size :]
                history = [
                    self.news_dict[nid]['title'] for nid in history
                ]
                history = [
                    self.encode(title, self.title_size) for title in history
                ]
            batch_history[i] = history

        batch_imp = [
            self.news_dict[nid]['title'] for nid in batch_imp
        ]
        batch_imp = [
            self.encode(title, self.title_size) for title in batch_imp
        ]

        batch_impid = torch.LongTensor(batch_impid)
        batch_history = [
            torch.LongTensor(history) for history in batch_history
        ]
        batch_imp = torch.LongTensor(batch_imp)

        if self.mode == 'train':
            batch_label = [int(x['label']) for x in batch]
            batch_label = torch.LongTensor(batch_label)
            return batch_impid, batch_history, batch_imp, batch_label
        elif self.mode == 'test':
            return batch_impid, batch_history, batch_imp
