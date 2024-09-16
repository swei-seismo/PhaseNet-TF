# PhaseNet-TF
PhaseNet-TF: Advanced Seismic Arrival Time Detection via Deep Neural Networks in the Spectrogram Domain, Leveraging Cutting-Edge Image Segmentation Approaches

## Train

To train the model with Slurm on MSU ICER, use `python -m src.train --multirun hydra/launcher=submitit_slurm experiment=production_default`