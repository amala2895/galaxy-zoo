#!/bin/bash

#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --time=60:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=galaxy_zoo_augmented
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asd508@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1 -c1

module load anaconda3/5.3.1
source activate vision
python GalaxyZooAllQuestions.py --data data_augmented --model_directory augmented_models --validation_length 32000 --train_length 150000 --outputfile augmented_output.txt --augmentation yes --epochs 100 --batch_size 128
