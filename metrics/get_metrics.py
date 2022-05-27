import argparse

parser = argparse.ArgumentParser(description='Metrics parser')
parser.add_argument('dataset', choices=['kitti', 'tum'],
                    required=True,
                    help="Dataset can be `kitti` or `tum`")

subparsers = parser.add_subparsers(help='sub help')

parser_kitti = subparsers.add_parser('kitti', help='kitti help')
parser_kitti.add_argument('bar', type=int, help='bar help')
print(parser.parse_args())
