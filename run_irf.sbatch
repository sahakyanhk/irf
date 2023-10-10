#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=100g 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:v100x:1
#SBATCH --time=8:00:00

set -e

PDB=$1
OUT=$2
NAME=`basename $PDB .pdb`


python inverse_refolding.py -i $PDB -c A -o $OUT/${NAME} -Ni 1000 -Ns 3 -t 0 -l ${NAME}.log -g 0.1

zip $OUT/${NAME}/pdb.zip $OUT/${NAME}/pdb/*pdb
rm -r $OUT/${NAME}/pdb


	
