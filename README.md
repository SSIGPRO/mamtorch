# mamtorch
Pytorch Multiply-And-Max/min repository

## Setup instructions

In order to use the MAM GPU kernels, launch "python kernelsetup.py install" inside the mamtorch folder. Ninja installation highly recommended (pip install ninja).

To install the python package, run "pip install ." in the root folder of this repository.

Torch version: 2.0.1+cu117

## Original paper:
Prono, Luciano; Bich, Philippe; Mangia, Mauro; Pareschi, Fabio; Rovatti, Riccardo; Setti, Gianluca (2023). A Multiply-And-Max/min Neuron Paradigm for Aggressively Prunable Deep Neural Networks. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.22561567.v1