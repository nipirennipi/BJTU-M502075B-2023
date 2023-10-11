import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from datetime import datetime
from torch.optim import Adam 
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import get_args
from dataset import MindDataset
from model import NewsRecBaseModel
from utils import init_seed, read_news, load_word_vectors, green_print
from metrics import *


@torch.no_grad()
def predict(args, model, test_loader):
    model.eval()
    test_loader = tqdm(test_loader, ncols=args.ncols)

    impid_list, score_list = [], []

    for step, (
        batch_impid, 
        batch_history, 
        batch_imp, 
    ) in enumerate(test_loader):
        batch_impid = batch_impid.to(args.device)
        batch_history = [
            history.to(args.device) for history in batch_history
        ]
        batch_imp = batch_imp.to(args.device)

        batch_score = model(
            batch_history, batch_imp
        )

        impid_list.extend(batch_impid.tolist())
        score_list.extend(batch_score.tolist())

    impres = {}
    for impid, score in zip(impid_list, score_list):
        if impid not in impres:
            impres[impid] = {}
            impres[impid]['score'] = []
        impres[impid]['score'].append(score)
    
    preds = []
    for impid in sorted(impres.keys()):
        pred = np.argsort(np.argsort(impres[impid]['score'])) + 1
        preds.append(pred)
    
    return preds


def main():
    args = get_args()

    green_print('### 1. Build vocabulary')
    news_dict, vocab = read_news(
        file_path=os.path.join(args.data_path, 'news.txt'), 
        filter_num=args.filter_num,
    )

    green_print('### 2. Load testset')
    test_dataset = MindDataset(
        file_path=os.path.join(args.data_path, 'test_behaviors_1.txt'),
        news_dict=news_dict,
        vocab=vocab,
        title_size=args.title_size,
        max_his_size=args.max_his_size,
        mode='test',
    )
    imps_len = test_dataset.imps_len()
    print(f'# test impressions: {imps_len}')

    test_kwargs = {
        'batch_size': args.infer_batch_size, 
        'shuffle': False,
        'collate_fn': test_dataset.collate_fn
    }
    test_loader = DataLoader(test_dataset, **test_kwargs)

    green_print('### 3. Load model and checkpoint')
    model = NewsRecBaseModel(
        vector_dim=args.vector_dim,
        news_dim=args.news_dim,
        window_size=args.window_size,
        vocab=vocab,
    )
    model.to(args.device)

    save_path = os.path.join(args.ckpt_path, args.ckpt_name)
    print(f'load from {save_path}')
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint)

    green_print('### 4. Start testing')
    print(f'time: {datetime.now()}')
    preds = predict(args, model, test_loader)

    green_print('### 5. Save prediction')
    save_file = './prediction.txt'
    with open(save_file, 'w') as f:
        for pred in preds:
            pred = ' '.join(map(str, pred))
            f.write(pred + '\n')

    print(f'save at {save_file}')


if __name__ == '__main__':
    main()
