# install Day 1 environment into jupyter 
module add python
conda activate /archive/course/SWE22/shared/SWE2023_week2/CondaEnvs/VAE_GAN_env_new
python -m ipykernel install --name VAE_GAN_env_new --user --display-name "VAE_GAN_env_new"
conda deactivate
