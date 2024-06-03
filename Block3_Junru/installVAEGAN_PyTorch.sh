# Install VAEGAN pytorch environment to jupyter
module add python
conda activate /archive/course/SWE22/shared/SWE2023_week2/CondaEnvs/VAE_GAN_PyTorch
python -m ipykernel install --name VAE_GAN_PyTorch --user --display-name "VAE_GAN_PyTorch"
conda deactivate
