#!/bin/bash
#usage ./run_hhpred.sh input.fasta output_path

set -e


fasta=$1
fasta_path=$PWD/$fasta
name=`basename ${fasta%.*}`
OUTDIR="${2:-output/$name}" #set defoult output if the second argument in null 
Ncpu=24
Niter=3

#dbs
UR30=/fdb/hhsuite/UniRef30_2022_02/UniRef30_2022_02
ECOD70=/data/saakyanh2/dbs/hhsuite/ECOD/ECOD_F70/ECOD_F70_20220613
PDB70=/data/saakyanh2/dbs/hhsuite/pdb70/pdb70
PFAM=/data/saakyanh2/dbs/hhsuite/Pfam35/pfama
CDD=/data/saakyanh2/dbs/hhsuite/NCBI_CD_v3.19/NCBI_CD
COG=/data/saakyanh2/dbs/hhsuite/COG_KOG/COG_KOG
SCOPE=/data/saakyanh2/dbs/hhsuite/scope70/scope70
CATH=/data/saakyanh2/dbs/hhsuite/CATH/CATH_S40


hhblits="hhblits	-cpu $Ncpu \
					-d $UR30 \
					-n $Niter \
					-o /dev/null \
					-e 1e-3 -p 20 \
					-z 1 -Z 250 -b 1 -B 250 \
					-v 1"

hhsearch="hhsearch	-cpu $Ncpu \
					-d $PDB70 -d $PFAM -d $CDD -d $COG -d $SCOPE -d $CATH \
					-p 60 -cov 30 \
					-z 1 -Z 250 -b 1 -B 250 \
					-loc -norealign \
					-seq 1 -ssm 2 -sc 1 \
					-dbstrlen 10000 \
					-maxres 32000 \
					-contxt /data/saakyanh2/dbs/hhsuite/context_data.crf \
					-v 1"


if [ -e $OUTDIR ]; then
	echo -e "\nDirectory $OUTDIR exists, renaming to ${OUTDIR}.bk \n" 
	mv -b $OUTDIR ${OUTDIR}.bk
	mkdir -p $OUTDIR
else
	mkdir -p $OUTDIR
fi

cd $OUTDIR

if (( `grep ">" $fasta_path | wc -l`  == 1 )); then
	echo -e "Query ($fasta_path) is a single protein sequence."
	echo -e "Running $Niter iterations of HHblits against UniRef30 for query MSA generation..."
	$hhblits -i $fasta_path -oa3m ${name}_hhbl.a3m 
	echo -e "done\n"

	echo -e "Predicting the secondary structures..."
	addss.pl ${name}_hhbl.a3m ${name}_hhbl_ss.a3m 2> /dev/null
	echo -e "done\n"

	#convert a3m to hhm to filter the similar sequences 
	echo -e "Running hhsearch..."
	$hhsearch -i ${name}_hhbl_ss.a3m -o ${name}.hhr 
	echo -e "done\n"

else 
	echo -e "\nQuery ($fasta_path) conntains multiple sequences, assuming it is an MSA"	

	echo -e "\nPredicting the secondary structures for $name"
	addss.pl $fasta_path ${name}_ss.a3m 2> /dev/null
	echo -e "done\n"

	echo -e "Running hhsearch..."
	$hhsearch -i ${name}_ss.a3m -o ${name}.hhr 
	echo -e "done\n"
	
fi
echo -e "Results are in the $OUTDIR"

cd ..
