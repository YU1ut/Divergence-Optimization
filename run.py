import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='PyTorch MixMatch Example')
parser.add_argument('--dataset', default='office', type=str, choices=['office', 'officehome', 'visda'],
                    help='type of dataset')
parser.add_argument('--noise-type', default='pairflip', type=str, choices=['pairflip', 'symmetric'],
                    help='type of label noise')
parser.add_argument('--percent', default=0.2, type=float, choices=[0.2, 0.45],
                    help='Percentage of noise')
parser.add_argument('--gpu', default='0', type=str,
                    help='id for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()


if args.dataset == 'office':
    domains = ['amazon', 'webcam', 'dslr']
    for source in domains:
        for target in domains:
            if source == target:
                continue
            subprocess.run(f"python train.py --config configs/{args.dataset}-train-config_opda.yaml --source ./txt/source_{source}_opda.txt --target ./txt/target_{target}_opda.txt --gpu {args.gpu}", shell=True)

elif args.dataset == 'officehome':
    domains = ['Art', 'Clipart', 'Product', 'Real']
    for source in domains:
        for target in domains:
            if source == target:
                continue
            subprocess.run(f"python train.py --config configs/{args.dataset}-train-config_opda.yaml --source ./txt/source_{source}_opda.txt --target ./txt/target_{target}_opda.txt --gpu {args.gpu}", shell=True)

elif args.dataset == 'visda':
    subprocess.run(f"python train.py --config configs/{args.dataset}-train-config_opda.yaml --source ./txt/source_visda_opda.txt --target ./txt/target_visda_opda.txt --gpu {args.gpu}", shell=True)

else:
    raise ValueError("Wrong dataset!")