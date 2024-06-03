# install Day 1 environment into jupyter 
module add python
conda activate /archive/course/SWE22/shared/SWE2023_week2/CondaEnvs/Nanocourse2022TF_new
python -m ipykernel install --name Nanocourse2022TF_new --user --display-name "Nanocourse2022TF_new"
conda deactivate
