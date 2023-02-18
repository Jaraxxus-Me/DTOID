import argparse

def test_parser():
    parser = argparse.ArgumentParser(description='Video Auto-encoder RealEstate10K Testing Options')

    # experiment specifics
    parser.add_argument('--dataset', default='BOP',
                        help='Name of the dataset')
    parser.add_argument('--data_path', default='data/BOP/ycbv/test',
                        help='Path of the dataset')
    parser.add_argument('--support_path', default='data/BOP/ycbv/test_video',
                        help='Path of the dataset')
    parser.add_argument('--savepath', type=str, default='log/test0',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default='pretrained/model.pth.tar',
                        help='Checkpoint file to resume')
    parser.add_argument('--worker', type=int, default=8,
                        help='number of dataloader threads')

    return parser
