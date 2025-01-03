#!/usr/bin/env python -u
# coding: utf-8

import os
import subprocess
import configparser
import argparse
from datetime import datetime
import copy
from types import SimpleNamespace

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.transforms import v2

from transformers import ViTForImageClassification

from utils import set_seed, generate_paths, generate_noskip_paths, \
    get_layers, add_mam_to_ViT, add_empty_to_ViT, apply_noskip, generate_newskip_paths, \
    apply_newskip, combine_config, convert_string

# Default configuration
config = configparser.ConfigParser()
config['System-setup'] = {
    'num_workers': 36,
    'seed': 42,
}
config['Model'] = {
    'model_name': f"vit_mam_imagenet1k",
    'starting_model': "", # pre-trained model name
    'teacher_model': "", # teacher model name
    'from_teacher': False,
    'remove_mlp_rescon': False, # remove intermediate and output mlp skip connection
    'rewire_mlp_rescon': False, # rewire int. and out. mlp skip connection with the value from att. and att. out skip connection
}
config['Task'] = {
    'num_classes': 1000,
    'dataset': 'imagenet1k', # or 'cifar100'
    'dataset_root': './data',
}
config['Mam-ification'] = {
    'attention_to_mam': False,
    'attention_output_to_mam': False,
    'intermediate_to_mam': True,
    'output_to_mam': True,
    'mam_blocks_num': -10,
    'mam_blocks_num_betazero': 0,
}
config['Empty blocks'] = {
    'convert_mam_to_empty': False,
}
config['Mam setup'] = {
    'mam_layer_splits': 1,
    'vcon_epochs': 12,
    'vcon_type': 'linear',
    'vcon_update_only_mam': False,
    'beta_zero_gain': 1.0,
}
config['Freezed layers'] = {
    'freeze_patch_embeddings': False,
    'freeze_attention': False,
    'freeze_attention_output': False,
}
config['Batch and epochs'] = {
    'batch_size': 128,
    'batch_accumulation': 1,
    'starting_epoch': 0,
    'total_epochs': 50,
}
config['Learning rate scheduling'] = {
    'learning_rate': 1e-4,
    'lr_decay_factor': 0.3,
    'lr_patience': 3,
    'lr_minimum': 1e-7,
    'lr_warmup_epochs': 12,
    'lr_warmup_start': 1e-4,
    'lr_cosine_decay': False
}
config['Regularizations'] = {
    'weight_decay': 0,
    'clipping_global_norm': False,
    'wsatstd': '', # weight magnitude saturation
    'attention_output_dropout_prob': 0,
    'hidden_dropout_prob': 0,
    'gradient_dropout_embeddings': 0,
    'gradient_dropout_attention_qkv': 0,
    'gradient_dropout_attention_output': 0,
}
config['Optimizer'] = {
    'actloss': 'cosine',
    'outloss': 'cross',
}
config['Dataugmentation'] = {
    'use_cutmix_mixup': True,
    'use_mixup_only': True,
    'mixup_alpha': 1.0,
    'use_autoaugment': False,
    'use_randaugment': True,
}
config['Metrics'] = {
    'store_attention_output_norm': False,
    'store_output_norm': False,
    'store_attention_input_gradient_norm': False,
    'store_intermediate_input_gradient_norm': False,
}

# Get from argparse configuration variables

parser = argparse.ArgumentParser(description="Update config with command-line arguments.")
for key in config:
    for subkey in config[key]:
        parser.add_argument(f"--{subkey}", type=str, help=f"Set the value for {subkey}")

parser.add_argument("-c", "--config_load_path", type=str, help=f"File path of an existent config file")
parser.add_argument("-f", "--force", action='store_true', help=f"File path of an existent config file")

args = parser.parse_args()

# Load from the [optional] specified config file (and overwrite default config)

if args.config_load_path is not None:
    if os.path.exists(args.config_load_path):
        config = configparser.ConfigParser()
        config.read(args.config_load_path)
        print(f"Config loaded from {args.config_load_path}.")
    else:
        print(f"Config file {args.config_load_path} does not exist.")

# Update config with args

for key in config:
    for subkey in config[key]:
        value = getattr(args, subkey)  # Get the argument value
        if value is not None:  # Only update if a value was provided
            config[key][subkey] = value

# File paths

model_name = config["Model"]["model_name"]
os.makedirs(model_name, exist_ok=True)
config_path = os.path.join(model_name, 'config.ini')
model_path = os.path.join(model_name, 'model.pt')
log_path = os.path.join(model_name, 'log.txt')
def backup_path(epoch):
    return os.path.join(model_name, f'backup_e{epoch}.pt')
tb_path = model_name

# Store configuration in the model folder

write_option = 'w' if args.force else 'x'
with open(config_path, write_option) as configfile:
    config.write(configfile)

# Extract configuration to variables

config = SimpleNamespace(**combine_config(config))
for c in vars(config):
    setattr(config, c, convert_string(getattr(config, c)))

# Set seed for execution

set_seed(config.seed)

# Define transformations for the training and test sets

if config.dataset == 'cifar100':
    transform_mean = (0.5074,0.4867,0.4411)
    transform_std = (0.2011,0.1987,0.2025)
elif config.dataset == 'imagenet1k':
    transform_mean = (0.485, 0.456, 0.406)
    transform_std = (0.229, 0.224, 0.225)

transform_train_list = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
]

if config.use_autoaugment:
    transform_train_list.append(v2.AutoAugment())
if config.use_randaugment:
    transform_train_list.append(v2.RandAugment())
transform_train_list.append(v2.Normalize(transform_mean, transform_std))

transform_train = v2.Compose(transform_train_list)

transform_test = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224)),
    v2.Normalize(transform_mean, transform_std),
])

mixup = v2.MixUp(alpha=config.mixup_alpha, num_classes=config.num_classes)
if config.use_mixup_only:
    cutmix_or_mixup = mixup
else:
    cutmix = v2.CutMix(num_classes=config.num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

if config.dataset == 'cifar100':
    # Download CIFAR-100 dataset and apply transformations
    trainset = torchvision.datasets.CIFAR100(root=config.dataset_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=config.dataset_root, train=False, download=True, transform=transform_test)
    valset, testset = torch.utils.data.random_split(testset, [0.5, 0.5], generator=torch.Generator().manual_seed(42))
elif config.dataset == 'imagenet1k':
    # Download ImageNet1k dataset and apply transformations
    trainset = torchvision.datasets.ImageNet(root=config.dataset_root, split='train', transform=transform_train)
    testset = torchvision.datasets.ImageNet(root=config.dataset_root, split='val', transform=transform_test)
    valset, testset = torch.utils.data.random_split(testset, [0.5, 0.5], generator=torch.Generator().manual_seed(42))    

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
print("training set length:", len(trainloader))

valloader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

print("validation set length:", len(valloader))
print("test set length:", len(testloader))

# Instantiate ViT model
model_checkpoint = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True,
    num_labels=config.num_classes,
)

# instantiate teacher model
if config.from_teacher:
    teacher = copy.deepcopy(model)
    teacher.load_state_dict(torch.load(config.teacher_model, weights_only=True))

# remove skip connections from student model
if config.rewire_mlp_rescon:
    newskip_layers_str = generate_newskip_paths(layers_number=config.mam_blocks_num)
    model = apply_newskip(model, newskip_layers_str)
if config.remove_mlp_rescon:
    noskip_layers_str = generate_noskip_paths(layers_number=config.mam_blocks_num)
    model = apply_noskip(model, noskip_layers_str)

# substitute MAM layers (or empty blocks)
vcon_steps = config.vcon_epochs*len(trainloader)//config.batch_accumulation
mam_layers_str = generate_paths(config.attention_to_mam, config.attention_output_to_mam, config.intermediate_to_mam, config.output_to_mam, layers_number=config.mam_blocks_num)
if config.convert_mam_to_empty:
    model = add_empty_to_ViT(model, mam_layers_str, vcon_steps, config.vcon_type)
else:
    model = add_mam_to_ViT(model, mam_layers_str, vcon_steps, config.mam_layer_splits, config.vcon_type, config.vcon_update_only_mam, config.beta_zero_gain)
mam_layers = get_layers(model, mam_layers_str)
if config.mam_blocks_num_betazero > 0:
    for i in range(config.mam_blocks_num_betazero):
        model.vit.encoder.layer[i].attention.attention.dense.beta = 0
        model.vit.encoder.layer[i].attention.output.dense.beta = 0
        model.vit.encoder.layer[i].intermediate.dense.beta = 0
        model.vit.encoder.layer[i].output.dense.beta = 0
elif config.mam_blocks_num_betazero < 0:
    for i in range(12+config.mam_blocks_num_betazero, 12):
        model.vit.encoder.layer[i].attention.attention.dense.beta = 0
        model.vit.encoder.layer[i].attention.output.dense.beta = 0
        model.vit.encoder.layer[i].intermediate.dense.beta = 0
        model.vit.encoder.layer[i].output.dense.beta = 0

if config.starting_model != "":
    model.load_state_dict(torch.load(config.starting_model, weights_only=True)) 

# put models on GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.to(device)
if config.from_teacher:
    teacher.to(device)

# define forward hooks
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

if config.from_teacher:
    for i in range(12):
        teacher.vit.encoder.layer[i].output.dense.register_forward_hook(get_activation(f'teacher.output{i}'))
        model.vit.encoder.layer[i].output.dense.register_forward_hook(get_activation(f'student.output{i}'))

if config.store_attention_output_norm:
    for i in range(12):
        model.vit.encoder.layer[i].attention.output.dense.register_forward_hook(get_activation(f"model.attention_output{i}"))

if config.store_output_norm:
    for i in range(12):
        model.vit.encoder.layer[i].output.dense.register_forward_hook(get_activation(f"model.output{i}"))

if config.store_attention_input_gradient_norm:
    model.vit.embeddings.register_forward_hook(get_activation(f"model.attention{0}.in"))
    for i in range(1, 12):
        model.vit.encoder.layer[i-1].output.register_forward_hook(get_activation(f"model.attention{i}.in"))

if config.store_intermediate_input_gradient_norm:
    for i in range(12):
        model.vit.encoder.layer[i].attention.output.register_forward_hook(get_activation(f"model.intermediate{i}.in"))

# freeze patch embeddings
for params in model.vit.embeddings.parameters():
    params.requires_grad = not config.freeze_patch_embeddings

# freeze attention layers and set attention output dropout
for l in model.vit.encoder.layer:
    for params in l.attention.attention.query.parameters():
        params.requires_grad = not config.freeze_attention
    for params in l.attention.attention.key.parameters():
        params.requires_grad = not config.freeze_attention
    for params in l.attention.attention.value.parameters():
        params.requires_grad = not config.freeze_attention
    for params in l.attention.output.dense.parameters():
        params.requires_grad = not config.freeze_attention_output
    l.attention.output.dropout.p = config.attention_output_dropout_prob
    
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
# Define loss function and optimizer
if config.actloss == "mse":
    criterion_d = nn.MSELoss()
elif config.actloss == "cosine":
    criterion_d = nn.CosineEmbeddingLoss()
criterion = nn.CrossEntropyLoss()
criterion_kld = nn.KLDivLoss(reduction="batchmean")
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# Train the model

print(model)

writer = SummaryWriter(log_dir=tb_path)

start_time = datetime.now()
last_time = start_time

progressless_epochs = 0
best_val_loss = 100000
total_train_steps = config.total_epochs*len(trainloader)

optimizer.zero_grad()
if config.from_teacher:
    teacher.eval()

for epoch in range(config.total_epochs):
    # learning rate warmup (linear increase from lr_warmup_start to learning_rate for lr_warmup_epochs epochs)
    if epoch < config.lr_warmup_epochs:
        optimizer.param_groups[0]['lr'] = epoch*(config.learning_rate-config.lr_warmup_start)/config.lr_warmup_epochs+config.lr_warmup_start
    
    # prepare lists to store activations norms
    if config.store_attention_output_norm:
        for i in range(12):
            train_attention_output_norm_mean = [0.0 for i in range(12)]
            train_attention_output_norm_var = [0.0 for i in range(12)]
            val_attention_output_norm_mean = [0.0 for i in range(12)]
            val_attention_output_norm_var = [0.0 for i in range(12)]

    if config.store_output_norm:
        for i in range(12):
            train_output_norm_mean = [0.0 for i in range(12)]
            train_output_norm_var = [0.0 for i in range(12)]
            val_output_norm_mean = [0.0 for i in range(12)]
            val_output_norm_var = [0.0 for i in range(12)]

    if config.store_attention_input_gradient_norm:
        for i in range(12):
            train_attention_input_gradient_norm_mean = [0.0 for i in range(12)]
            train_attention_input_gradient_norm_var = [0.0 for i in range(12)]

    if config.store_intermediate_input_gradient_norm:
        for i in range(12):
            train_intermediate_input_gradient_norm_mean = [0.0 for i in range(12)]
            train_intermediate_input_gradient_norm_var = [0.0 for i in range(12)]

    # train for one epoch
    running_loss = 0.0
    running_activations_loss = 0.0
    running_output_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for i, data in enumerate(trainloader, 0):
        # evaluate current training step
        train_step = epoch*len(trainloader) + i

        # prepare input data
        inputs, labels = data[0].to(device), data[1].to(device)
        if config.use_cutmix_mixup:
            inputs, labels = cutmix_or_mixup(inputs, labels)
        
        # forward pass
        if config.from_teacher:
            outputs = model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            activations_loss = 0
            for j in range(10):
                if config.actloss == "cosine":
                    activations_loss += criterion_d(activations[f"student.output{j}"].flatten(start_dim=1), activations[f"teacher.output{j}"].flatten(start_dim=1), torch.ones(inputs.size(0), device=device))
                else:
                    activations_loss += criterion_d(activations[f"student.output{j}"], activations[f"teacher.output{j}"])
            
            # output loss
            if config.outloss == "kld":
                output_loss = criterion_kld(torch.log_softmax(outputs.logits, axis=1), torch.softmax(teacher_outputs.logits, axis=1))
            else:
                output_loss = criterion(outputs.logits, torch.softmax(teacher_outputs.logits, axis=1))
            output_loss += criterion(outputs.logits, labels)
            loss = activations_loss + output_loss
        else:
            outputs = model(inputs)
            output_loss = criterion(outputs.logits, labels)
            loss = output_loss

        # get predictions
        if config.use_cutmix_mixup:
            _, maxlabels = torch.max(labels, 1)
        _, predicted = torch.max(outputs.logits, 1)

        # apply l2 weight decay
        if config.weight_decay != 0:
            for l in mam_layers:
                loss += config.weight_decay*torch.norm(l.weight)

        # prepare tensor with gradients for storing
        if config.store_attention_input_gradient_norm:
            for j in range(12):
                activations[f"model.attention{j}.in"].retain_grad()
        if config.store_intermediate_input_gradient_norm:
            for j in range(12):
                activations[f"model.intermediate{j}.in"].retain_grad()

        # backward pass
        loss.backward()

        # gradient dropout
        if config.gradient_dropout_embeddings > 0.0:
            for j in range(12):
                mask = (torch.rand_like(model.vit.embeddings.patch_embeddings.projection.weight.grad) > config.gradient_dropout_embeddings).int()
                model.vit.embeddings.patch_embeddings.projection.weight.grad *= mask

        if config.gradient_dropout_attention_qkv > 0.0:
            for j in range(12):
                mask = (torch.rand_like(model.vit.encoder.layer[j].attention.attention.query.weight.grad) > config.gradient_dropout_attention_qkv).int()
                model.vit.encoder.layer[j].attention.attention.query.weight.grad *= mask

        if config.gradient_dropout_attention_output > 0.0:
            for j in range(12):
                mask = (torch.rand_like(model.vit.encoder.layer[j].attention.output.dense.weight.grad) > config.gradient_dropout_attention_output).int()
                model.vit.encoder.layer[j].attention.output.dense.weight.grad *= mask

        # evaluate norms of the gradients
        if config.store_attention_input_gradient_norm:
            for j in range(12):
                norm = torch.norm(activations[f"model.attention{j}.in"].grad, p=2, dim=-1)
                train_attention_input_gradient_norm_mean[j] = (train_attention_input_gradient_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                train_attention_input_gradient_norm_var[j] = (train_attention_input_gradient_norm_var[j]*i+torch.mean(torch.square(norm-train_attention_input_gradient_norm_mean[j])).detach().cpu().numpy())/(i+1)
                del norm
        if config.store_intermediate_input_gradient_norm:
            for j in range(12):
                norm = torch.norm(activations[f"model.intermediate{j}.in"].grad, p=2, dim=-1)
                train_intermediate_input_gradient_norm_mean[j] = (train_intermediate_input_gradient_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                train_intermediate_input_gradient_norm_var[j] = (train_intermediate_input_gradient_norm_var[j]*i+torch.mean(torch.square(norm-train_intermediate_input_gradient_norm_var[j])).detach().cpu().numpy())/(i+1)
                del norm

        # optimize
        if (i+1) % config.batch_accumulation == 0:
            if config.clipping_global_norm:
                # apply gradient clipping with maximum norm = 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            optimizer.zero_grad()

        # apply weight magnitude saturation
        if config.wsatstd != "":
            for l in mam_layers:
                wsat = config.wsatstd*torch.var(l.weight.data).square_()
                l.output.dense.weight.data = torch.clip(l.data, -wsat, wsat)

        # evaluate norm of the activations
        if config.store_attention_output_norm:
            for j in range(12):
                norm = torch.norm(activations[f"model.attention_output{j}"], p=2, dim=-1)
                train_attention_output_norm_mean[j] = (train_attention_output_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                train_attention_output_norm_var[j] = (train_attention_output_norm_var[j]*i+torch.mean(torch.square(norm-train_attention_output_norm_mean[j])).detach().cpu().numpy())/(i+1)
                del norm
        if config.store_output_norm:
            for j in range(12):
                norm = torch.norm(activations[f"model.output{j}"], p=2, dim=-1)
                train_output_norm_mean[j] = (train_output_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                train_output_norm_var[j] = (train_output_norm_var[j]*i+torch.mean(torch.square(norm-train_output_norm_mean[j])).detach().cpu().numpy())/(i+1)
                del norm

        # evaluate current loss and accuracy
        running_loss += loss.item()
        if config.from_teacher:
            running_activations_loss += activations_loss.item()
        running_output_loss += output_loss.item()
        total += labels.size(0)
        if config.use_cutmix_mixup:
            correct += (predicted == maxlabels).sum().item()
        else:
            correct += (predicted == labels).sum().item()
        
        train_accuracy = (100 * correct / total)
        train_loss = running_loss * config.batch_size / total
        if config.from_teacher:
            train_activations_loss = running_activations_loss * config.batch_size / total
        train_output_loss = running_output_loss * config.batch_size / total

        # apply vanishing contributions (only when update is performed)
        if (i+1) % config.batch_accumulation == 0:
            if vcon_steps > 0:
                for l in mam_layers:
                    l.vcon_step()

        # apply learning rate cosine decay
        if epoch > config.lr_warmup_epochs and config.lr_cosine_decay:
            optimizer.param_groups[0]['lr'] = config.lr_minimum + 0.5 * (config.learning_rate - config.lr_minimum) * (1 + np.cos(np.pi * train_step/total_train_steps))

        if config.from_teacher:
            print(f'[{epoch+1}, {i+1:4d}] activations_loss: {train_activations_loss:1.3e} output_loss: {train_output_loss:1.3e} total_loss: {train_loss:1.3e} acc: {train_accuracy:3.2f}%', end="\r")
        else:
            print(f'[{epoch+1}, {i+1:4d}] output_loss: {train_output_loss:1.3e} acc: {train_accuracy:3.2f}%', end="\r")
    if config.from_teacher:
        print(f'[{epoch+1}, {i+1:4d}] activations_loss: {train_activations_loss:1.3e} output_loss: {train_output_loss:1.3e} total_loss: {train_loss:1.3e} acc: {train_accuracy:3.2f}%')
    else:
        print(f'[{epoch+1}, {i+1:4d}] output_loss: {train_output_loss:1.3e} acc: {train_accuracy:3.2f}%')
            
    # validation

    running_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            _, predicted = torch.max(outputs.logits, 1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            validation_accuracy = (100 * correct / total)
            validation_loss = running_loss * config.batch_size / total

            # evaluate norm of the activations
            if config.store_attention_output_norm:
                for j in range(12):
                    norm = torch.norm(activations[f"model.attention_output{j}"], p=2, dim=-1)
                    val_attention_output_norm_mean[j] = (val_attention_output_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                    val_attention_output_norm_var[j] = (val_attention_output_norm_var[j]*i+torch.mean(torch.square(norm-val_attention_output_norm_mean[j])).detach().cpu().numpy())/(i+1)
                    del norm
            if config.store_output_norm:
                for j in range(12):
                    norm = torch.norm(activations[f"model.output{j}"], p=2, dim=-1)
                    val_output_norm_mean[j] = (val_output_norm_mean[j]*i+torch.mean(norm).detach().cpu().numpy())/(i+1)
                    val_output_norm_var[j] = (val_output_norm_var[j]*i+torch.mean(torch.square(norm-val_output_norm_mean[j])).detach().cpu().numpy())/(i+1)
                    del norm

            print(f'[{epoch+1}, {i+1:4d}] loss: {validation_loss:1.3e} acc: {validation_accuracy:3.2f}%', end="\r")
        print(f'[{epoch+1}, {i+1:4d}] loss: {validation_loss:1.3e} acc: {validation_accuracy:3.2f}%')
    
    # store info on tensorboard
    writer.add_scalar('Loss/train', train_loss, config.starting_epoch+epoch+1)
    writer.add_scalar('Accuracy/train', train_accuracy, config.starting_epoch+epoch+1)
    writer.add_scalar('Loss/validation', validation_loss, config.starting_epoch+epoch+1)
    writer.add_scalar('Accuracy/validation', validation_accuracy, config.starting_epoch+epoch+1)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], config.starting_epoch+epoch+1)
    if config.store_attention_output_norm:
        for j in range(12):
            writer.add_scalar(f'Attention output act norm mean (train)/layer{j}', train_attention_output_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Attention output act norm var (train)/layer{j}', train_attention_output_norm_var[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Attention output act norm mean (val)/layer{j}', val_attention_output_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Attention output act norm var (val)/layer{j}', val_attention_output_norm_var[j], config.starting_epoch+epoch+1)
    if config.store_output_norm:
        for j in range(12):
            writer.add_scalar(f'Output act norm mean (train)/layer{j}', train_output_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Output act norm var (train)/layer{j}', train_output_norm_var[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Output act norm mean (val)/layer{j}', val_output_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Output act norm var (val)/layer{j}', val_output_norm_var[j], config.starting_epoch+epoch+1)
    if config.store_attention_input_gradient_norm:
        for j in range(12):
            writer.add_scalar(f'Attention in grad norm mean (train)/layer{j}', train_attention_input_gradient_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Attention in grad norm var (train)/layer{j}', train_attention_input_gradient_norm_var[j], config.starting_epoch+epoch+1)
    if config.store_intermediate_input_gradient_norm:
        for j in range(12):
            writer.add_scalar(f'Intermediate in grad norm mean (train)/layer{j}', train_intermediate_input_gradient_norm_mean[j], config.starting_epoch+epoch+1)
            writer.add_scalar(f'Intermediate in grad norm var (train)/layer{j}', train_intermediate_input_gradient_norm_var[j], config.starting_epoch+epoch+1)

    if mam_layers != []:
        beta_value = mam_layers[0].beta
        writer.add_scalar('Beta', beta_value, epoch+1) 
    
    # update learning rate on plateau
    if epoch > config.lr_warmup_epochs:
        if validation_loss > best_val_loss:
            progressless_epochs += 1
        else:
            best_val_loss = validation_loss
        if progressless_epochs > config.lr_patience:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*config.lr_decay_factor
            if optimizer.param_groups[0]['lr'] < config.lr_minimum:
                optimizer.param_groups[0]['lr'] = config.lr_minimum
            progressless_epochs = 0
    
    print(f'Epoch time: {datetime.now()-last_time}', end=" ")
    print(f'Running time: {datetime.now()-start_time}')
    last_time = datetime.now()

    # backup model/store final model
    torch.save(model.state_dict(), model_path)
    
stop_time = datetime.now()
    
print('Training Complete')
print(f'Total time: {str(stop_time-start_time)}')


# Test the model
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {(100 * correct / total)}%')