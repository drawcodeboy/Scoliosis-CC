from models import load_encoder, ContrastiveNetwork, InstanceLoss, ClusterLoss
from dataloader import load_dataset
from utils.engine import train_one_epoch
from utils.save_ckpt import *

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

import argparse
import time
import sys

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # GPU
    parser.add_argument("--use-cuda", action="store_true")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default='image') # 'image', 'mask', 'seq'
    
    # Model 
    parser.add_argument("--encoder", type=str, default='ResNet') # Encoder (ResNet, LSTMNet)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--instance-temperature", type=float, default=0.5)
    parser.add_argument("--cluster-temperature", type=float, default=1.0)
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4*3.)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0)
    
    # Save
    parser.add_argument("--save-weights-dir", type=str, default="saved/weights")
    parser.add_argument("--save-losses-dir", type=str, default="saved/losses")
    
    return parser

def print_setup(device, args):
    print("=======================[Settings]========================")
    print(f"\n  [GPU]")
    print(f"  |-[device]: {device}")
    print(f"\n  [MODEL]")
    print(f"  |-[encoder]: {args.encoder}")
    print(f"  |-[feature dim]: {args.feature_dim}")
    print(f"  |-[instance temperature]: {args.instance_temperature}")
    print(f"  |-[cluster temperature]: {args.cluster_temperature}")
    print(f"  |-[n clusters]: {args.n_clusters}")
    print(f"\n  [DATA]")
    print(f"  |-[dataset]: {args.dataset}")
    print(f"\n  [HYPERPARAMETERS]")
    print(f"  |-[epochs]: {args.epochs}")
    print(f"  |-[lr]: {args.lr:06f}")
    print(f"  |-[weight decay]: {args.weight_decay}")
    print(f"  |-[batch size]: {args.batch_size}")
    print(f"\n [SAVE PATHS]")
    print(f"  |-[SAVE WEIGHTS DIR]: {args.save_weights_dir}")
    print(f"  |-[SAVE LOSSES DIR]: {args.save_losses_dir}")
    print("\n=======================================================")
    
    print("Proceed? [Y/N]: ", end="")
    proceed = input().lower()
    
    if proceed == 'n':
        sys.exit()

def collate_fn(batch):
    longest_size = max([sample[0][0].shape[0] for sample in batch])
    
    paths = [sample[1] for sample in batch]
        
    x_i_batch, x_j_batch = [], []
        
    for sample in batch: 
        if longest_size != sample[0][0].shape[0]:
            zero_temp = torch.zeros(longest_size - sample[0][0].shape[0], 1)
        
            x_i_batch.append(torch.cat([zero_temp, sample[0][0]], dim=0))
            x_j_batch.append(torch.cat([zero_temp, sample[0][1]], dim=0))
        else:
            x_i_batch.append(sample[0][0])
            x_j_batch.append(sample[0][1])
    
    x_i_batch = torch.stack(x_i_batch, dim=0)
    x_j_batch = torch.stack(x_j_batch, dim=0)
    
    return (x_i_batch, x_j_batch), paths

def main(args):
    device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
        
    print_setup(device, args)
    
    # Load Models
    
    encoder = load_encoder(args.encoder)
    model = ContrastiveNetwork(encoder, 
                               feature_dim=args.feature_dim,
                               class_num=args.n_clusters).to(device)
    
    # Load Dataset
    
    ds = load_dataset(dataset=args.dataset)
    dl = DataLoader(ds, 
                    shuffle=True, 
                    batch_size=args.batch_size,
                    collate_fn=collate_fn if args.dataset == 'seq' else None)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Loss_fn
    loss_fns = {
        'instance': InstanceLoss(args.batch_size, args.instance_temperature, device).to(device),
        'cluster': ClusterLoss(args.n_clusters, args.cluster_temperature, device).to(device),
    }
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer,
                                mode='min',
                                factor=0.5,
                                patience=5,
                                min_lr=1e-7)
    
    # Training
    total_train_loss = []
    
    for current_epoch in range(0, args.epochs):
        current_epoch += 1
        print("=======================================================")
        print(f"Epoch: [{current_epoch:03d}/{args.epochs:03d}]", end="\n\n")
        
        start_time = int(time.time())
        train_loss = train_one_epoch(model, dl, optimizer, loss_fns, scheduler, device)
        elapsed_time = int(time.time() - start_time)
        print(f"Training Time: {elapsed_time//60:02d}m {elapsed_time%60:02d}s", end="\n\n")
        
        total_train_loss.append(train_loss)
        
        if current_epoch % 25 == 0:
            save_model_ckpt(model, args.dataset, args.n_clusters, current_epoch, args.save_weights_dir)
            
        save_loss_ckpt(total_train_loss, args.dataset, args.n_clusters, args.save_losses_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Contrastive Clustering", parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    main(args)