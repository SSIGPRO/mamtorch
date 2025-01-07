## Reproducing ViT-B/16 + ImageNet1k results from "A Multiply-And-Max/min Neuron Paradigm for Aggressively Prunable Deep Neural Networks"

Launch the training script to train ViT-B/16 + ImageNet1k containing MAM layers

    python vit_training.py

To train also a vanilla MAC-based version of ViT-B/16 + ImageNet1k, run

    python vit_training.py --model_name=vit_mac_imagenet1k --intermediate_to_mam=False --output_to_mam=False

To prune and compare the MAM- and MAC-based models, run

    python create_pruning_plots.py
