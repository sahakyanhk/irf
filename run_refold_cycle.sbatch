#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=100g 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100x:1
#SBATCH --time=48:00:00


source /home/saakyanh2/.bashrc
conda activate esm 

PDB=$1
OUT=$2
NAME=`basename $PDB .pdb`


/data/saakyanh2/miniconda3/envs/esm/bin/python inverse_refolding.py -i $PDB -c A -o $OUT/${NAME} -Ni 3000 -Ns 3 -t 0 -l ${NAME}.log -g 0.1

zip $OUT/${NAME}/pdb.zip $OUT/${NAME}/pdb/*pdb
rm -r $OUT/${NAME}/pdb


	