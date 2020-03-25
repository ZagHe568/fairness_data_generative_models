import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_features', type=int, default=102)
    parser.add_argument('--data_A_path', type=str, default='data/male_adult_dataset')
    parser.add_argument('--data_B_path', type=str, default='data/female_adult_dataset')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--gpu', type=str, default='')

    return parser.parse_args()