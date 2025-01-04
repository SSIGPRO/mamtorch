import os
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from transformers import ViTForImageClassification
from utils import set_seed, generate_paths, get_layers, add_mam_to_ViT
import copy
import numpy as np
from pruning_utils import LMP, GMP, LGP, GGP, create_pruning_plot
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pruning script with MAM layers configuration')
    parser.add_argument('--mam_layers_num', type=int, default=-10,
                      help='Number of MAM layers (positive: start from input, negative: start from output)')
    parser.add_argument('--full_pruning', action='store_true', help='Enable full pruning mode')
    return parser.parse_args()

def main():
    args = parse_args()
    mam_layers_num = args.mam_layers_num
    
    output_path = f"./pruned_results"
    os.makedirs(output_path, exist_ok=True)

    # Hyperparameters
    batch_size = 128

    # Pruning points
    pruning_points = [0, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999, 0.99999999, 0.999999999]
    starting_model_mac = f"./vit_mac_imagenet1k/model.pt"
    starting_model_mam = f"./vit_mam_imagenet1k/model.pt"

    # CONSTANTS
    NUM_CLASSES = 1000  # number of classes for classification

    # System-setup
    num_workers = 36
    seed = 42

    set_seed(seed)

    # Define transformations for datasets
    transform_mean = [0.485, 0.456, 0.406]
    transform_std = [0.229, 0.224, 0.225]

    transform_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((224, 224)),
        v2.Normalize(transform_mean, transform_std)
    ]
    transform = v2.Compose(transform_list)

    testset = torchvision.datasets.ImageNet(root='./data', split='val', transform=transform)
    valset, testset = torch.utils.data.random_split(testset, [0.5, 0.5], generator=torch.Generator().manual_seed(42))
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print("validation set length:", len(val_loader))
    print("test set length:", len(test_loader))

    # LGP
    print(f'\nPruning MAM!\n')
    lgp_mam = LGP(starting_model=starting_model_mam, test_loader=test_loader, 
                val_loader=val_loader, use_mam=True, 
                mam_layers_num=mam_layers_num, pruning_points=pruning_points, 
                num_classes=NUM_CLASSES, full_pruning=args.full_pruning
                )
    
    print(f'\nPruning MAC!\n')
    lgp_mac = LGP(starting_model=starting_model_mac, test_loader=test_loader, 
                val_loader=val_loader, use_mam=False, 
                mam_layers_num=mam_layers_num, pruning_points=pruning_points, 
                num_classes=NUM_CLASSES, full_pruning=args.full_pruning
                )
    
    create_pruning_plot(pruning_points, lgp_mac, lgp_mam, mam_layers_num, pruning_technique='LGP', output_path=output_path)


    # GGP
    print(f'\nPruning MAM!\n')
    ggp_mam = GGP(starting_model=starting_model_mam, test_loader=test_loader, 
                val_loader=val_loader, use_mam=True, 
                mam_layers_num=mam_layers_num, pruning_points=pruning_points, 
                num_classes=NUM_CLASSES, full_pruning=args.full_pruning
                )
    
    print(f'\nPruning MAC!\n')
    ggp_mac = GGP(starting_model=starting_model_mac, test_loader=test_loader, 
                val_loader=val_loader, use_mam=False, 
                mam_layers_num=mam_layers_num, pruning_points=pruning_points, 
                num_classes=NUM_CLASSES, full_pruning=args.full_pruning
                )
    
    create_pruning_plot(pruning_points, ggp_mac, ggp_mam, mam_layers_num, pruning_technique='GGP', output_path=output_path)

    # LMP
    print(f'\nPruning MAM!\n')
    lmp_mam = LMP(starting_model=starting_model_mam, test_loader=test_loader, 
                use_mam=True, mam_layers_num=mam_layers_num, 
                pruning_points=pruning_points, num_classes=NUM_CLASSES, full_pruning=args.full_pruning
                )
    
    print(f'\nPruning MAC!\n')
    lmp_mac = LMP(starting_model=starting_model_mac, test_loader=test_loader, 
                use_mam=False, mam_layers_num=mam_layers_num, 
                pruning_points=pruning_points, num_classes=NUM_CLASSES, full_pruning=args.full_pruning
                )
    
    create_pruning_plot(pruning_points, lmp_mac, lmp_mam, mam_layers_num, pruning_technique='LMP', output_path=output_path)

    # GMP
    print(f'\nPruning MAM!\n')
    gmp_mam = GMP(starting_model=starting_model_mam, test_loader=test_loader, 
                use_mam=True, mam_layers_num=mam_layers_num, 
                pruning_points=pruning_points, num_classes=NUM_CLASSES, full_pruning=args.full_pruning
                )
    
    print(f'\nPruning MAC!\n')
    gmp_mac = GMP(starting_model=starting_model_mac, test_loader=test_loader, 
                use_mam=False, mam_layers_num=mam_layers_num, 
                pruning_points=pruning_points, num_classes=NUM_CLASSES, full_pruning=args.full_pruning
                )
    
    create_pruning_plot(pruning_points, gmp_mac, gmp_mam, mam_layers_num, pruning_technique='GMP', output_path=output_path)


if __name__ == "__main__":
    main()