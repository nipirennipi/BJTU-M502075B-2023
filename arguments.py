import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int, 
        default=23, 
        help='Random seed.',
    )

    parser.add_argument(
        '--data_path',
        type=str, 
        default='./data', 
        help='Path of data set.',
    )

    parser.add_argument(
        '--vectors_path',
        type=str, 
        default='./data', 
        help='Path of pre-trained word vectors.',
    )

    parser.add_argument(
        '--vector_dim',
        type=int,
        default=300,
        help='Dimensions of pre-trained word vectors.',
    )

    parser.add_argument(
        '--filter_num',
        type=int, 
        default=3, 
        help='Filter words that appear less frequently than <filter_num>.',
    )

    parser.add_argument(
        '--title_size',
        type=int,
        default=20,
        help='Pad or truncate the news title length to <title_size>',
    )

    parser.add_argument(
        '--max_his_size',
        type=int,
        default=50,
        help='Maximum length of the history interaction. (truncate old if necessary)',
    )

    parser.add_argument(
        '--val_ratio',
        type=float, 
        default=0.05, 
        help='Split <val_ratio> from training set as the validation set.',
    )

    parser.add_argument(
        '--news_dim',
        type=int,
        default=128,
        help='Dimensions of news representations.',
    )

    parser.add_argument(
        '--window_size',
        type=int,
        default=3,
        help='Window size of CNN filters.',
    )

    parser.add_argument(
        '--device',
        type=str, 
        default=('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    parser.add_argument(
        '--epochs',
        type=int, 
        default=5,
    )

    parser.add_argument(
        '--train_batch_size',
        type=int, 
        default=64, 
        help='Batch size during training.',
    )

    parser.add_argument(
        '--infer_batch_size',
        type=int, 
        default=256,
        help='Batch size during inference.',
    )

    parser.add_argument(
        '--learning_rate',
        type=float, 
        default=0.0001,
    )

    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='./checkpoint', 
        help='Path of checkpoint.',
    )

    parser.add_argument(
        '--ckpt_name',
        type=str,
        default='model_checkpoint.pth',
    )

    parser.add_argument(
        '--ncols',
        type=int,
        default=80,
        help='Parameters of tqdm: the width of the entire output message.',
    )

    args = parser.parse_args()
    return args
