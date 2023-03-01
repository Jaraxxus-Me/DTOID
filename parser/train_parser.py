import argparse

def train_parser():
    parser = argparse.ArgumentParser(description='Video Auto-encoder RealEstate10K Testing Options')

    # experiment specifics
    parser.add_argument('--dataset', default='OWID',
                        help='Name of the dataset')
    parser.add_argument('--data_path', default='data/OWID/P2',
                        help='Path of the dataset')
    parser.add_argument('--support_path', default='data/OWID/P1',
                        help='Path of the dataset')
    parser.add_argument('--savepath', type=str, default='log/train3',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint file to resume')
    parser.add_argument('--worker', type=int, default=8,
                        help='number of dataloader threads')
    parser.add_argument('--bs', type=int, default=16,
                        help='number of batch size')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs')
    parser.add_argument('--valid_freq', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='number of epochs')
    return parser
