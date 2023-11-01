import os
import argparse
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append("/home/rishabrs/cistup/contrastive_learning_pyssl/models")

import torch
import torch.nn as nn
import torchvision

from torch.utils.tensorboard import SummaryWriter

import pyssl.builders as builders
from utils import DatasetSelector

# Take in command line arguments
parser = argparse.ArgumentParser(description='SimCLR Training')
parser.add_argument('dataset', type=str,
                    help='dataset type')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--latent-dim', default=1024, type=int, metavar='N',
                    help='Size of Latent dimension vectors')
parser.add_argument('--learning-rate', default=1e-3, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--batch-size', default=1024, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--checkpoint-dir', default='../../checkpoints/simclr', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--start-at-checkpoint', default=False, type=bool,
                    help='Value = True to start at checkpoint')
parser.add_argument('--start-epoch', type=int,
                    help='Epoch to resume training using checkpoint')


def main():
    args = parser.parse_args()

    # use cuda if gpu is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Choose datasets between mnist, fmnist, cifar10 and intel classification datasets 
    dataset_selector = DatasetSelector(args.dataset)
    dataset = dataset_selector.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # initialize backbone (resnet50) for simclr
    backbone = torchvision.models.resnet50(weights=None)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Linear(num_features, args.latent_dim)
    feature_size = backbone.fc.out_features
    model = builders.SimCLR(backbone, feature_size, image_size=32)
    model = model.to(device)

    num_batches = len(dataloader)
    num_epochs = args.epochs 
    start_epoch = 1
    writer = SummaryWriter("runs/simclr")
    running_loss = 0
    step = 0
    checkpoint_step = 50

    # loss optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    # If we want to resume at a checkpoint
    if(args.start_at_checkpoint):
        start_epoch = getattr(args, "start-epoch")  # The start epoch of the model you want to load.
        resume_epoch = start_epoch-1
        step = start_epoch
        
        print(os.path.join(
                args.checkpoint_dir, f'SimCLR-{start_epoch}.pth'))
        
        # Load the model weights from the checkpoint.
        model_checkpoint = torch.load(os.path.join(
                args.checkpoint_dir, f'SimCLR-{resume_epoch}.pth'))

        model.load_state_dict(model_checkpoint['model_state_dict'])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])

    # Start training loop
    for epoch in range(num_epochs - start_epoch+1):

         # Initialize the progress bar
        pbar = tqdm(dataloader)
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)

            # zero the parameter gradients
            model.zero_grad()

            # compute loss
            loss = model(images)
            running_loss += loss.item()
            
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()

            # update the progress bar with the loss and accuracy
            pbar.set_postfix(loss=loss.item())
                
        # Log metrics to Tensorboard.
        writer.add_scalar("Training loss", running_loss/num_batches, global_step=step)
        step += 1
        running_loss = 0.0
        
        if ((epoch+start_epoch) % checkpoint_step) == 0:
            # Save model checkpoints
            model_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }

            file_name = f'SimCLR-{args.dataset}-latent_dim{args.latent_dim}-epoch{epoch}.pth'
            torch.save(model_checkpoint, os.path.join(
                args.checkpoint_dir, file_name))


if __name__ == '__main__':
    main()
