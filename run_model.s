#!/bin/bash
#
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=galaxy_zoo_1
#SBATCH --mail-type=ALL
##SBATCH --mail-user=asd508@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1 -c1

module load anaconda3/5.3.1
conda env create -f requirements.yaml
source activate vision
python GalaxyZooAllQuestions.py --epochs 50 --train_length 50000 --validation_length 10000


