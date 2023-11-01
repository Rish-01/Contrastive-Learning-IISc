import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append("/home/rishabrs/cistup/contrastive_learning_pyssl/models")

import torch
import torch.nn as nn
import torchvision

import pyssl.builders as builders
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import DatasetSelector

# Take in command line arguments
parser = argparse.ArgumentParser(description='SimCLR Training')
parser.add_argument('dataset', type=str,
                    help='dataset type')
parser.add_argument('file_name', type=str,
                    help='Name of .pth file of model for inference')
parser.add_argument('--latent-dim', default=1024, type=int, metavar='N',
                    help='Size of Latent dimension vectors')
parser.add_argument('--batch-size', default=1024, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--checkpoints-dir', default='../../checkpoints/simclr', type=Path,
                    metavar='DIR', help='path to model checkpoint directory')
parser.add_argument('--plot-dir', default='../../tsne_plots/simclr', type=Path,
                    help='path to tsne plot directory')
parser.add_argument('--inference-epoch', default=1000, type=int, metavar='N',
                    help='Inference model trained for these many epochs')

def save_tsne_plot():
    '''
    inference_epoch - epoch number of the model whose weights you want to use for inference.
    ''' 
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize backbone (resnet50) for simclr
    backbone = torchvision.models.resnet50(weights=None)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Linear(num_features, args.latent_dim)
    feature_size = backbone.fc.out_features
    model = builders.SimCLR(backbone, feature_size, image_size=32)
    model = model.to(device)

    # Choose datasets between mnist, fmnist, cifar10 and intel classification datasets 
    dataset_selector = DatasetSelector(args.dataset)
    dataset = dataset_selector.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Load the model weights for inference.
    model_checkpoint = torch.load(os.path.join(
                args.checkpoints_dir, args.file_name))

    model.load_state_dict(model_checkpoint['model_state_dict'])

    # Calculate the embeddings by passing inputs through the trained encoder.
    fig, ax = plt.subplots()
    
    label_appended_embeddings = []
    label_appended_tsne_embeddings = []
    model.eval()
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        images = images.to(device)        
        with torch.no_grad():
            backbone_embeddings = model.backbone(images)
    
        tsne = TSNE(n_components=2, random_state=42)
        backbone_embeddings = backbone_embeddings.cpu()
        tsne_embeddings = tsne.fit_transform(backbone_embeddings)

        # Concat the labels into the embeddings to create an embedding of shape(batch-size, feature-size + 1)
        labels = labels.view(-1, 1)
        batch_label_appended_embeddings = torch.cat((backbone_embeddings, labels + 1), dim=-1)
        label_appended_embeddings.append(batch_label_appended_embeddings)

        # Accumulate tsne embeddings over all batches of shape(batch-size, 2 + 1)
        tsne_embeddings_tensor = torch.tensor(tsne_embeddings)
        tsne_embeddings_tensor = torch.cat((tsne_embeddings_tensor, labels), dim=-1)
        label_appended_tsne_embeddings.append(tsne_embeddings_tensor)

        sc = ax.scatter(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], c=labels, s=0.7)
        ax.legend(*sc.legend_elements(), title='clusters', bbox_to_anchor = (1 , 1))
    
    label_appended_embeddings = torch.cat(label_appended_embeddings, dim=0)
    print(label_appended_embeddings.shape)
    label_appended_tsne_embeddings = torch.cat(label_appended_tsne_embeddings, dim=0)
    print(label_appended_tsne_embeddings.shape)

    # Save the embeddings as a .npy file
    np.save(f'../../embeddings/simclr/embeddings_{args.file_name}.npy', label_appended_embeddings.numpy())
    np.save(f'../../embeddings/simclr/tsne_embeddings_{args.file_name}.npy', label_appended_tsne_embeddings.numpy())

    # Save TSNE plot
    fig.savefig(os.path.join(
                args.plot_dir, f'tsne-{args.dataset}-{args.inference_epoch}'))


if __name__ == "__main__":
    args = parser.parse_args()
    save_tsne_plot()