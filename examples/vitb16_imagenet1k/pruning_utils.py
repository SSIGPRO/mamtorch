import os
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from transformers import ViTForImageClassification
from utils import set_seed, generate_paths, get_layers, add_mam_to_ViT
import copy
import numpy as np
import matplotlib.pyplot as plt


def load_model(starting_model, use_mam, mam_layers_num, num_classes, verbose=False): 
    # Define ViT model
    model_checkpoint = 'google/vit-base-patch16-224-in21k'
    model = ViTForImageClassification.from_pretrained(
        model_checkpoint,
        ignore_mismatched_sizes=True,
        num_labels=num_classes,
    )
    layers_str = generate_paths(use_mam, use_mam, False, layers_number=mam_layers_num)
    model = add_mam_to_ViT(model, layers_str)
    mam_layers = get_layers(model, layers_str)

    if starting_model != "":
        model.load_state_dict(torch.load(starting_model, weights_only=True)) 

    # Put model on GPU and set eval mode
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if verbose:
        # Print model to check for structure (has mam been substituted correctly?)
        print(model)
    return model, device


# TEST THE MODEL (NO GRAD)
def evaluate(model, dataloader_in_use, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader_in_use, 0):
            # prepare input data
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
        
            # forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            # evaluate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
            accuracy = (100 * correct / total)
            print(f'[{i+1:4d}] acc: {accuracy:3.3f}%', end="\r")
    print(f'[{i+1:4d}] acc: {accuracy:3.3f}%')
    return accuracy


# TEST THE MODEL (WITH GRAD)
def evaluate_with_grad(model, dataloader_in_use, device):
    print('Evaluating with grad')
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i, data in enumerate(dataloader_in_use, 0):
        # prepare input data
        inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
    
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        _, predicted = torch.max(outputs.logits, 1)
        # evaluate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
        accuracy = (100 * correct / total)
        print(f'[{i+1:4d}] acc: {accuracy:3.3f}%', end="\r")
    model.eval()
    print('Evaluation with grad finished')


def LMP(starting_model, test_loader, use_mam, mam_layers_num, pruning_points, num_classes, full_pruning):
    print(f'Loading original non-pruned model')
    model, device = load_model(starting_model, use_mam, mam_layers_num, num_classes, verbose=False)
    print(f'Model ready!\n')

    print("\nLAYERWISE MAGNITUDE PRUNING:\n")

    all_acc = []
    model.eval()
    if full_pruning:
        mam_layers_num = -12
    with torch.no_grad():
        for p in pruning_points:
            print(f'Percentage pruned: {p*100:.2f}%')
            for i in range(-1, mam_layers_num - 1, -1):
                intermediate = model.vit.encoder.layer[i].intermediate.dense.weight
                output = model.vit.encoder.layer[i].output.dense.weight
    
                flattened_weights = intermediate.flatten()
                threshold = torch.quantile(flattened_weights.abs(), p)
                intermediate[intermediate.abs() <= threshold] = 0
    
                flattened_weights = output.flatten()
                threshold = torch.quantile(flattened_weights.abs(), p)
                output[output.abs() <= threshold] = 0
    
                model.vit.encoder.layer[i].intermediate.dense.weight
                # Replace the weight matrix
                model.vit.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(intermediate)
                model.vit.encoder.layer[i].output.dense.weight = torch.nn.Parameter(output)
            accuracy = evaluate(model, test_loader, device)
            all_acc.append(accuracy)
    return np.array(all_acc)


def GMP(starting_model, test_loader, use_mam, mam_layers_num, pruning_points, num_classes, full_pruning):
    print(f'Loading original non-pruned model')
    model, device = load_model(starting_model, use_mam, mam_layers_num, num_classes, verbose=False)
    print(f'Model ready!\n')

    print("\nGLOBAL MAGNITUDE PRUNING:\n")
    
    all_acc = []
    model.eval()
    if full_pruning:
        mam_layers_num = -12
    with torch.no_grad():
        for p in pruning_points:
            print(f'Percentage pruned: {p*100:.2f}%')

            concatenated_tensor = torch.tensor([], device=device)

            for i in range(-1, mam_layers_num - 1, -1):
                intermediate = model.vit.encoder.layer[i].intermediate.dense.weight.flatten()
                output = model.vit.encoder.layer[i].output.dense.weight.flatten()
                concatenated_tensor = torch.cat([concatenated_tensor, intermediate, output])

            # Compute the GLOBAL pruning threshold
            print('Computing global threshold...')
            concatenated_tensor = concatenated_tensor.abs()
            concatenated_tensor, _ = torch.sort(concatenated_tensor)
            index_tensor = int(p * len(concatenated_tensor))
            index_tensor = max(0, min(index_tensor, len(concatenated_tensor) - 1))
            threshold = concatenated_tensor[index_tensor]
            print(f'Global threshold: {threshold:3f}')

            #Prune matrices according to threshold
            with torch.no_grad():
                for i in range(-1, mam_layers_num - 1, -1):
                    intermediate = model.vit.encoder.layer[i].intermediate.dense.weight
                    intermediate[intermediate.abs() <= threshold] = 0

                    output = model.vit.encoder.layer[i].output.dense.weight
                    output[output.abs() <= threshold] = 0

                    model.vit.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(intermediate)
                    model.vit.encoder.layer[i].output.dense.weight = torch.nn.Parameter(output)

            accuracy = evaluate(model, test_loader, device)
            print()
            all_acc.append(accuracy)
    return np.array(all_acc)


def LGP(starting_model, test_loader, val_loader, use_mam, mam_layers_num, pruning_points, num_classes, full_pruning):
    print(f'Loading original non-pruned model')
    model, device = load_model(starting_model, use_mam, mam_layers_num, num_classes, verbose=False)
    model_with_gradient = copy.deepcopy(model)
    model_with_gradient.zero_grad()
    evaluate_with_grad(model_with_gradient, val_loader, device)
    print(f'Model ready!\n')

    print("\nLAYERWISE GRADIENT PRUNING:\n")
    
    all_acc = []
    model.eval()
    if full_pruning:
        mam_layers_num = -12

    for p in pruning_points:
        print(f'Percentage pruned: {p*100:.2f}%')
        with torch.no_grad():
            for i in range(-1, mam_layers_num - 1, -1):
                intermediate = model_with_gradient.vit.encoder.layer[i].intermediate.dense.weight
                grad_intermediate = model_with_gradient.vit.encoder.layer[i].intermediate.dense.weight.grad
                weight_grad_intermediate = intermediate * grad_intermediate
    
                flattened_weights = weight_grad_intermediate.flatten()
                threshold = torch.quantile(flattened_weights.abs(), p)
                intermediate[weight_grad_intermediate.abs() <= threshold] = 0
    
                output = model_with_gradient.vit.encoder.layer[i].output.dense.weight
                grad_output = model_with_gradient.vit.encoder.layer[i].output.dense.weight.grad
                weight_grad_output = output * grad_output
    
                flattened_weights = weight_grad_output.flatten()
                threshold = torch.quantile(flattened_weights.abs(), p)
                output[weight_grad_output.abs() <= threshold] = 0
                # Replace the weight matrix
                model.vit.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(intermediate)
                model.vit.encoder.layer[i].output.dense.weight = torch.nn.Parameter(output)
            print(f'Accuracy: ')
            accuracy = evaluate(model, test_loader, device)
            print()
            all_acc.append(accuracy)
    return np.array(all_acc)


def GGP(starting_model, test_loader, val_loader, use_mam, mam_layers_num, pruning_points, num_classes, full_pruning):
    print(f'Loading original non-pruned model')
    model, device = load_model(starting_model, use_mam, mam_layers_num, num_classes, verbose=False)
    model_with_gradient = copy.deepcopy(model)
    model_with_gradient.zero_grad()
    evaluate_with_grad(model_with_gradient, val_loader, device)
    print(f'Model ready!\n')
    
    all_acc = []
    model.eval()
    if full_pruning:
        mam_layers_num = -12

    for p in pruning_points:
        print(f'Percentage pruned: {p*100:.2f}%')
        #Find global threshold
        concatenated_tensor = torch.tensor([], device=device)
        with torch.no_grad():
            for i in range(-1, mam_layers_num - 1, -1):
                intermediate = model_with_gradient.vit.encoder.layer[i].intermediate.dense.weight
                grad_intermediate = model_with_gradient.vit.encoder.layer[i].intermediate.dense.weight.grad
                weight_grad_intermediate = intermediate * grad_intermediate
                weight_grad_intermediate = weight_grad_intermediate.flatten()
    
                output = model_with_gradient.vit.encoder.layer[i].output.dense.weight
                grad_output = model_with_gradient.vit.encoder.layer[i].output.dense.weight.grad
                weight_grad_output = output * grad_output
                weight_grad_output = weight_grad_output.flatten()
                
                concatenated_tensor = torch.cat([concatenated_tensor, weight_grad_intermediate, weight_grad_output])
    
        # Compute the GLOBAL pruning threshold
        print('Computing global threshold...')
        concatenated_tensor = concatenated_tensor.abs()
        concatenated_tensor, _ = torch.sort(concatenated_tensor)
        index_tensor = int(p * len(concatenated_tensor))
        index_tensor = max(0, min(index_tensor, len(concatenated_tensor) - 1))
        threshold = concatenated_tensor[index_tensor]
        print(f'Global threshold: {threshold:3f}')
    
        #Prune matrices according to threshold
        with torch.no_grad():
            for i in range(-1, mam_layers_num - 1, -1):
                intermediate = model_with_gradient.vit.encoder.layer[i].intermediate.dense.weight
                grad_intermediate = model_with_gradient.vit.encoder.layer[i].intermediate.dense.weight.grad
                weight_grad_intermediate = intermediate * grad_intermediate
                
                intermediate[weight_grad_intermediate.abs() <= threshold] = 0
    
                output = model_with_gradient.vit.encoder.layer[i].output.dense.weight
                grad_output = model_with_gradient.vit.encoder.layer[i].output.dense.weight.grad
                weight_grad_output = output * grad_output
                
                output[weight_grad_output.abs() <= threshold] = 0
    
                model.vit.encoder.layer[i].intermediate.dense.weight = torch.nn.Parameter(intermediate)
                model.vit.encoder.layer[i].output.dense.weight = torch.nn.Parameter(output)
            print(f'Accuracy: ')
            accuracy = evaluate(model, test_loader, device)
            print()
            all_acc.append(accuracy)
    return np.array(all_acc) 


def create_pruning_plot(x, y, z, mam_layers_num, pruning_technique, output_path):
    # Create the plot
    xticks = np.array([100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001])
    x, y, z = 100-100*np.array(x), np.array(y), np.array(z)
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, marker='o', label=f'ViT-MAC{mam_layers_num}')  # Plot with markers and a label
    plt.plot(x, z, marker='x', label=f'ViT-MAM{mam_layers_num}')  # Plot with markers and a label

    # Add title and labels
    plt.title(f'{pruning_technique} Pruning of ViT-ImageNet1K', fontsize=14)
    plt.xlabel('Remaining parameters (%)', fontsize=12)  # Replace with your specific label
    plt.ylabel('Accuracy', fontsize=12)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    # Set x-axis to logarithmic
    plt.xlim([xticks[-1], xticks[0]])
    plt.xscale('log')
    plt.xticks(xticks, [str(num) for num in xticks])
    plt.gca().invert_xaxis()

    output_pdf = output_path + f"/{pruning_technique}{mam_layers_num}.pdf"
    output_csv = output_path + f"/{pruning_technique}{mam_layers_num}.csv"
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")

    # Save to CSV
    data = np.column_stack((x, y, z))  # Combine the arrays into columns
    header = 'x,y,z'  # Column names
    np.savetxt(output_csv, data, delimiter=',', header=header, comments='')

    # Show the plot
    plt.show()


def calculate_absolute_pruning_percentage(percentages, num_blocks):
    """
    Calculate the pruning percentage for given parameters.

    Args:
        percentages (list): Array of percentages to compute the real percentages for.
        num_blocks (int): Number of blocks (-12 to -1).

    Returns:
        list: Real percentages corresponding to the input percentages.
    """
    # Against 12 blocks (each block contains 7087872, so 7087872*12=85054464)
    # The whole model contains 86567656 parameter
    # Each block contain 7087872, so the 12 blocks contain 85054464 parameters
    # Each block contains 2 fc that we prune so 4722432, in total over the 12 blocks we prune up to 56669184 parameters

    # Validate the number of blocks
    assert num_blocks in range(-12, 0), "Error: num_blocks must be in range [-12, -1]."

    # Calculate maximum pruning percentage and block percentage
    nonfc_percentage = 100 - (56669184/86567656 * 100.0)
    fc_block_percentage = 4722432/86567656 * 100.0
    
    # Calculate the pruning percentage for the given num_blocks
    max_percentage = nonfc_percentage + (12 + num_blocks) * fc_block_percentage

    # Calculate the real percentages for the input array of remaining parameters
    real_percentages = [(p/100)*(100-max_percentage) + max_percentage for p in percentages]
    return real_percentages