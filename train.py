import os
import pprint
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


def train(args, model, optimizer, train_loader):
    model.train()
    train_loader = tqdm(train_loader, ncols=args.ncols)

    logloss = 0.
    for step, (
        batch_impid,
        batch_history,
        batch_imp,
        batch_label,
    ) in enumerate(train_loader):
        batch_impid = batch_impid.to(args.device)
        batch_history = [
            history.to(args.device) for history in batch_history
        ]
        batch_imp = batch_imp.to(args.device)
        batch_label = batch_label.to(args.device)

        batch_loss, batch_score = model(
            batch_history, batch_imp, batch_label
        )

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logloss += batch_loss.item()

    logloss = logloss / step
    return logloss


@torch.no_grad()
def eval(args, model, val_loader):
    model.eval()
    val_loader = tqdm(val_loader, ncols=args.ncols)

    logloss = 0.
    impid_list, label_list, score_list = [], [], []

    for step, (
        batch_impid, 
        batch_history, 
        batch_imp, 
        batch_label,
    ) in enumerate(val_loader):
        batch_impid = batch_impid.to(args.device)
        batch_history = [
            history.to(args.device) for history in batch_history
        ]
        batch_imp = batch_imp.to(args.device)
        batch_label = batch_label.to(args.device)

        batch_loss, batch_score = model(
            batch_history, batch_imp, batch_label
        )

        logloss += batch_loss.item()
        impid_list.extend(batch_impid.tolist())
        label_list.extend(batch_label.tolist())
        score_list.extend(batch_score.tolist())

    logloss = logloss / step

    impres = {}
    for impid, label, score in zip(impid_list, label_list, score_list):
        if impid not in impres:
            impres[impid] = {}
            impres[impid]['label'] = []
            impres[impid]['score'] = []
        impres[impid]['label'].append(label)
        impres[impid]['score'].append(score)

    auc_list, mrr_list, ndcg5_list, ndcg10_list = [], [], [], []
    for impid in impres.keys():
        label = impres[impid]['label']
        score = impres[impid]['score']

        imp_auc = roc_auc_score(label, score)
        imp_mrr = mrr_score(label, score)
        imp_ndcg5 = ndcg_score(label, score, k=5)
        imp_ndcg10 = ndcg_score(label, score, k=10)

        auc_list.append(imp_auc)
        mrr_list.append(imp_mrr)
        ndcg5_list.append(imp_ndcg5)
        ndcg10_list.append(imp_ndcg10)

    auc = np.mean(auc_list)
    mrr = np.mean(mrr_list)
    ndcg5 = np.mean(ndcg5_list)
    ndcg10 = np.mean(ndcg10_list)
    return logloss, auc, mrr, ndcg5, ndcg10


def main():
    args = get_args()
    green_print('### arguments:')
    pprint.pprint(args.__dict__, width=1)
    init_seed(args.seed)        

    green_print('### 1. Build vocabulary and load pre-trained vectors')
    news_dict, vocab = read_news(
        file_path=os.path.join(args.data_path, 'news.txt'), 
        filter_num=args.filter_num,
    )

    word_vectors = load_word_vectors(
        vectors_path=os.path.join(
            args.vectors_path, 'glove.840B.300d.txt'
        ),
        vocab=vocab,
    )

    print(f"vocab size: {len(vocab)}")
    print(f"unknow words: {len(vocab) - len(word_vectors)}")

    green_print('### 2. Load data and split')
    mind_dataset = MindDataset(
        file_path=os.path.join(args.data_path, 'train_behaviors.txt'),
        news_dict=news_dict,
        vocab=vocab,
        title_size=args.title_size,
        max_his_size=args.max_his_size,
        mode='train',
    )
    imps_len = mind_dataset.imps_len()
    val_imps_len = int(imps_len * args.val_ratio)
    train_imps_len = imps_len - val_imps_len
    print(
        f'# total impressions: {imps_len:>6}\n' \
        f'# train impressions: {train_imps_len:>6} | {1 - args.val_ratio:6.2%}\n' \
        f'# valid impressions: {val_imps_len:>6} | {args.val_ratio:6.2%}' \
    )
    
    train_dataset, val_dataset = mind_dataset.train_val_split(val_imps_len)

    train_kwargs = {
        'batch_size': args.train_batch_size, 
        'shuffle': True, 
        'collate_fn': mind_dataset.collate_fn
    }
    val_kwargs = {
        'batch_size': args.infer_batch_size, 
        'shuffle': False,
        'collate_fn': mind_dataset.collate_fn
    }
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    green_print('### 3. Load model and optimizer')
    model = NewsRecBaseModel(
        vector_dim=args.vector_dim,
        news_dim=args.news_dim,
        window_size=args.window_size,
        vocab=vocab,
        word_vectors=word_vectors,
    )
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    print('done.')

    green_print('### 4. Start training')
    print(f'time: {datetime.now()}')
    for epoch in range(args.epochs):
        print('-' * 88)
        print(f'epoch: {epoch}')
        train_logloss = train(args, model, optimizer, train_loader)
        print(f'train info || logloss: {train_logloss:.4f}')
        val_logloss, auc, mrr, ndcg5, ndcg10 = eval(args, model, val_loader)
        print(
            f'valid info || logloss: {val_logloss:.4f} | auc: {auc:.4f} ' \
            f'| mrr: {mrr:.4f} | ndcg@5: {ndcg5:.4f} | ndcg@10: {ndcg10:.4f}' \
        )

    green_print('### 5. Save model')
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    save_path = os.path.join(args.ckpt_path, args.ckpt_name)
    torch.save(model.state_dict(), save_path)

    print(f'save at {save_path}')


if __name__ == '__main__':
    main()
