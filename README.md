# mamtorch
Pytorch Multiply-And-Max/min repository

## Setup instructions

**Make sure your torch version is compiled with the same cuda version of your machine and nvcc, otherwise you will not be able to compile the library.**

Ninja installation is highly recommended

    pip install ninja

To compile and install mamtorch library, run

    pip install git+https://github.com/SSIGPRO/mamtorch

To compile and install the latest work-in-progress library, run

    pip install git+https://github.com/SSIGPRO/mamtorch@develop

Otherwise, you can download the repository and run

    pip install .

in the repository's root folder.

Note: installation may take some time since nvcc must compile the kernels for the machine in use.

Latest tested torch version: 2.5.0+cu126

## Original paper:
Prono, Luciano; Bich, Philippe; Boretti, Chiara; Mangia, Mauro; Pareschi, Fabio; Rovatti, Riccardo; Setti, Gianluca (2023). A Multiply-And-Max/min Neuron Paradigm for Aggressively Prunable Deep Neural Networks. In: Transactions on Neural Networks and Learning Systems, 2025 (DOI: 10.1109/TNNLS.2025.3527644)
## Acknowledgment

This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-Generation EU (Piano Nazionale di Ripresa e Resilienza (PNRR) – Missione 4 Componente 2, Investimento 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them.
