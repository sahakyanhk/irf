import argparse
import numpy as np
import os
import re
from pathlib import Path
from matplotlib import pyplot as plt  

import biotite.structure as bs
import biotite.structure.io.pdb
import biotite.structure.io.xtc

import torch
import esm
import esm.inverse_folding

#https://github.com/dauparas/ProteinMPNN/tree/main try this instead of esm-if


#!!!!! ADD PDB PROCESSING, FILTER ATOMS, RENAME CHAIN TO "A", ADD PDB FETCHING

#backup if output directory exists
def backup_output(outpath):
    print(f'\nSaving output files to {os.getcwd()}/{args.outpath}')
    if os.path.isdir(outpath): 
        backup_list = []
        last_backup = int()
        for dir_name in os.listdir(outpath + "/.."):
            if dir_name.startswith(outpath.split('/')[-1] + '.'):
                backup=(dir_name.split('.')[-1])
                if backup.isdigit(): 
                    backup_list.append(backup)
                    last_backup = int(max(backup_list))
        print(f'\n{outpath} already exists, renameing to {outpath}.{str(last_backup +  1)}') 
        os.replace(outpath, outpath + '.' + str(last_backup +  1))


#fetch pdb if it does not exists
def fetch_pdb(pdbid, chain, outpath):
    print(f'Fetching {pdbid}{chain} from RCSB')
    fetched_pdb_path = f'{outpath}/{pdbid}{chain}.pdb'
    ATOM='"ATOM"'
    chain=f'"{chain}"'
    C0='${}'.format("0")
    C1='${}'.format("1")
    C5='${}'.format("5")
    cmd=f"curl -s https://files.rcsb.org/view/{pdbid}.pdb | grep -m 1 END -B1000000 | awk '{C1}=={ATOM} && {C5}=={chain}' > {fetched_pdb_path}"
    os.system(cmd)
    return fetched_pdb_path


#find the highest possible temperature to generate very diverse sequences
def sample_temperature(pdb, chain, outpath):
    os.makedirs(outpath + '/Ttesting', exist_ok=True)
    with open(outpath + '/Ttesting/temp_sampling.log', 'a') as f:
        f.write(f'Temperature was not set manualy, searching for an optimal temperature.\n')
    coords, native_seq = esm.inverse_folding.util.load_coords(pdb, chain)
    T = 0.1
    mean_lddt = 100
    lddt_cutoff = args.lddt_cutoff
    step = 0
    T_X = []
    pLDDT_Y = []
    pTM_Y = []
    while mean_lddt > lddt_cutoff:
        T_pdb_path = outpath + '/Ttesting/' + 'T_' + str(round(T, 3))
        os.makedirs(T_pdb_path, exist_ok=True)
        lddt_list = []
        ptm_list = []
        step += 1
        for i in range(10):
            sampled_seq = model_if.sample(coords, temperature=T, device=torch.device('cuda'))
            with torch.no_grad():
                output = model_v1.infer(sampled_seq)
            predicted_structure = model_v1.output_to_pdb(output)[0] 
            lddt = output["mean_plddt"][0].tolist()
            ptm = output["ptm"][0].tolist()
            lddt_list.append(lddt)
            ptm_list.append(ptm)
            with open(T_pdb_path + f'/temp_{round(T, 3)}_test_{i}.pdb', 'w') as f:
                f.write(predicted_structure)
                f.close()
        mean_lddt = np.average(lddt_list)
        mean_ptm = np.average(ptm_list)
        delta_plddt = mean_lddt - lddt_cutoff
        T_X.append(T)
        pLDDT_Y.append(mean_lddt)
        pTM_Y.append(mean_ptm)
        Tprev = T
        T = T + 0.01 * delta_plddt
        with open(outpath + '/Ttesting/temp_sampling.log', 'a') as f:
            f.write(f'Step {step}\tT={round(T,3)}\tpLDDT={round(mean_lddt,2)}+/-{round(np.std(lddt_list),2)}\tpTM={round(mean_ptm, 2)}+/-{round(np.std(ptm_list),2)}\tΔpLDDT={round(delta_plddt, 2)}\n')
        print(f'Step {step}\tT={round(T,3)}\tpLDDT={round(mean_lddt,2)}+/-{round(np.std(lddt_list),2)}\tpTM={round(mean_ptm, 2)}+/-{round(np.std(ptm_list),2)}\tΔpLDDT={round(delta_plddt, 2)}')

    #save temp testing plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Temp ')
    ax1.set_ylabel('pLDDT', color='tab:red')
    ax1.plot(T_X, pLDDT_Y, color='tab:red') #do not pring the last element
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('pTM', color='tab:blue') 
    ax2.plot(T_X, pTM_Y, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(outpath + '/Ttesting/temp_sampling.png')

    if step <= 1 and T <=0.001:
        print("!!!WARNING!!! cannot predict reliable structures")
        args.savepdbs = False
        args.gen_warn = False
        args.num_iterations = args.num_samples * args.num_iterations
        args.num_samples = 1 
        args.temperature = 0.3
        return args.temperature
    else:
        print(f"optimal temperature for pLDDT cutoff {lddt_cutoff} is {args.temperature}")
        return round(Tprev, 3)


def extract_lddt(pdb_string):
    lddts = []
    for line in pdb_string.split('\n'):
        if line[13:15] == 'CA':
            lddt = float(line[60:66].strip())
            lddts.append(lddt)
    aver_lddt = sum(lddts) / len(lddts)
    return aver_lddt, lddts


# introduce gaps in sequence regions with plddt < x (percentile)  
def make_gaps(seq, lddt):
    percentile  =  sorted(np.array(lddt))[(round(len(lddt) * args.gaps))] #10 residues with the lowest plddt will be substittued with gaps
    mask = np.array(lddt) < percentile
    subst_list = ['-' if pos else item for pos, item in zip(mask, seq)]
    gapped_seq = ''.join(subst_list)
    return gapped_seq


#needed for make_backbone_traj
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


#make backebone trajectory for clusterization
def make_backbone_traj(outpath):

    pdb_path = args.outpath + '/pdb/'

    basename = (os.path.basename(args.inputpdb).split('.')[0])
    inputprot =  bs.io.load_structure(args.inputpdb)
    backbone = inputprot[((((inputprot.atom_name == "N") | 
                        (inputprot.atom_name == "CA") | 
                        (inputprot.atom_name == "C") | 
                        (inputprot.atom_name == "O")) & 
                        (inputprot.chain_id == "A") & 
                        (inputprot.hetero == False)))]

    bs.io.save_structure(args.outpath + '/' + basename + '_backbone.pdb', backbone)

    pdb_file = bs.io.pdb.PDBFile()
    coord = []

    for file_name in sorted_alphanumeric(os.listdir(pdb_path)):
        pdb_file = bs.io.pdb.PDBFile.read(pdb_path + file_name)
        struct = pdb_file.get_structure()
        struct_bb = struct[0][(struct[0].atom_name == "N") | (struct[0].atom_name == "CA") | (struct[0].atom_name == "C") | (struct[0].atom_name == "O")]
        pdb_file.set_structure(struct_bb)
        coord.append(pdb_file.get_coord()[0])

    traj = bs.from_template(backbone, np.array(coord))
    bs.io.save_structure(args.outpath + '/' + basename + '_traj.pdb', traj)


#do PCA or build a tree using to cluster diverse MSAs 
# def cluster_by_tree:
#     return cluster
    

def inverse_cycle(model_if, model_v1, alphabet, args):  

    #make outputh dirs
    #if args.savepdbs:
    pdb_path = args.outpath + '/pdb/'
    os.makedirs(pdb_path, exist_ok=True)

    #download pdb file
    if args.fetchpdb is not None:
        args.inputpdb = fetch_pdb(args.fetchpdb, args.chain, args.outpath)       

    #set temp
    if args.temperature == 0:
        print(f'Temperature was not set manualy, searching for an optimal temperature.\n')
        args.temperature = sample_temperature(args.inputpdb, args.chain, args.outpath)

    with open(args.outpath + '/' + args.log, 'a') as f:
        f.write(f'''
\nrunning inverce cycle
============input parameters============\n
input pdb\t=\t{args.inputpdb}
chain\t\t=\t{args.chain}
outpath\t\t=\t{args.outpath} # print full path here
temperature\t=\t{args.temperature}
lddt_cutof\t=\t{args.lddt_cutoff}
num_samp\t=\t{args.num_samples}
num_iter\t=\t{args.num_iterations}
output log\t=\t{args.log}
gaps \t\t=\t{args.gaps}
\n
''')

        print(f'''\nrunning inverce cycle 
\n============input parameters============
\npdbfile\t=\t{args.inputpdb}
chain\t\t=\t{args.chain}
outpath\t\t=\t{args.outpath}
temperature\t=\t{args.temperature}
lddt_cutof\t=\t{args.lddt_cutoff}
num_samp\t=\t{args.num_samples}
num_iter\t=\t{args.num_iterations}
output_log\t=\t{args.log}
\n
''')

    coords, native_seq = esm.inverse_folding.util.load_coords(args.inputpdb, args.chain)  
    basename = (os.path.basename(args.inputpdb).split('.')[0])
    with open(f'{args.outpath}/{basename}_seqs.fasta', 'a') as f:
                f.write(f'>{basename} | input pdb sequence \n')
                f.write(native_seq + '\n')

    #main loop    
    if args.num_samples > 1:
        print(f'Running {args.num_iterations} iterations with {args.num_samples} sequences in each\n\nSample\tTemp\tseq_id\tprev_seq_id\tpLDDT\tpTM')

        LOW_LDDT_WARNING = 0
        
        for ni in range(args.num_iterations):
            coords, native_seq = esm.inverse_folding.util.load_coords(args.inputpdb, args.chain)  
            prev_sampled_seq=native_seq
            ns = 0
            while ns < args.num_samples:
                #sample sequence and sequence identities
                sampled_seq = model_if.sample(coords, temperature=args.temperature, device=torch.device('cuda'))
                native_seq_recovery = np.mean([(a==b) for a, b in zip(native_seq, sampled_seq)])
                prev_seq_recovery = np.mean([(a==b) for a, b in zip(prev_sampled_seq, sampled_seq)])

                #predict a structure for the sampled sequence and into a pdb file
                with torch.no_grad():
                    output = model_v1.infer(sampled_seq)
                predicted_structure = model_v1.output_to_pdb(output)[0]
                ptm = output["ptm"][0].tolist()

                lddt, lddt_list = extract_lddt(predicted_structure)
                if lddt > args.lddt_cutoff:
                    ns+=1 
                    LOW_LDDT_WARNING -= 1 
                    #write pdb file
                    if args.savepdbs is True: 
                        with open(pdb_path + f'i{ni+1}_{basename}_{ns}.pdb', 'w') as f:
                            f.write(predicted_structure)

                    #wrtire all sequences in a single file (MSA) 
                    with open(f'{args.outpath}/{basename}_seqs.fasta', 'a') as f:
                        f.write(f'>i{ni+1}_{basename}_{ns} | T: {round(args.temperature,2)} | seq_id: {round(native_seq_recovery, 2)} | prev_seq_id: {round(prev_seq_recovery, 2)} | lDDT {round(lddt, 2)} | pTM {round(ptm, 2)} \n')
                        f.write(sampled_seq + '\n')

                    #write an MSA with gaped sequences 
                    if args.gaps > 0:
                        gapped_sampled_seq = make_gaps(sampled_seq, lddt_list)
                        with open(f'{args.outpath}/{basename}_seqs_gapped.fasta', 'a') as f:
                            f.write(f'>i{ni+1}_{basename}_{ns} | T: {round(args.temperature,2)} | seq_id: {round(native_seq_recovery, 2)} | prev_seq_id: {round(prev_seq_recovery, 2)} | lDDT {round(lddt, 2)} | pTM {round(ptm, 2)} \n')
                            f.write(gapped_sampled_seq + '\n') 

                    #write a log file
                    with open(args.outpath + '/' + args.log, 'a') as f:
                        f.write(f'i{ni+1}_{basename}_{ns}\t{round(args.temperature,2 )}\t{round(native_seq_recovery, 2)}\t{round(prev_seq_recovery, 2)}\t{round(lddt, 2)}\t{round(ptm, 2)}\n')
                    
                    #extract xyz coordinates from the pdb file
                    coords, prev_sampled_seq = esm.inverse_folding.util.load_coords(pdb_path + f'i{ni+1}_{basename}_{ns}.pdb', "A")
                    print(f'i{ni+1}_{basename}_{ns}\t{round(args.temperature, 2)}\t{round(native_seq_recovery, 2)}\t{round(prev_seq_recovery, 2)}\t{round(lddt, 2)}\t{round(ptm, 2)}')
                elif args.gen_warn is True: 
                    LOW_LDDT_WARNING += 1 
                    print(f'#low pLDDT {round(lddt, 1)}')
                    if LOW_LDDT_WARNING > 10 and args.temperature > 0.2:
                        args.temperature -= 0.01
                        args.temperature = round(args.temperature, 2)
                        print(f"too many LOW_LDDT_WARNING, updating temp. New temp is {round(args.temperature, 2)} ")
                        LOW_LDDT_WARNING = 0 
    
        print(f'\n{args.num_iterations * args.num_samples} sequences are generated')

    else:
        print(f'Running {args.num_iterations} iterations\n\nSample\tTemp\tseq_id\tpLDDT\tpTM')
        for ni in range(args.num_iterations):
            #sample sequence and sequence identities
            sampled_seq = model_if.sample(coords, temperature=args.temperature, device=torch.device('cuda'))
            native_seq_recovery = np.mean([(a==b) for a, b in zip(native_seq, sampled_seq)])
            #predict a structure for the sampled sequence and into a pdb file
            with torch.no_grad():
                output = model_v1.infer(sampled_seq)
            predicted_structure = model_v1.output_to_pdb(output)[0]
            ptm = output["ptm"][0].tolist()
            lddt, lddt_list = extract_lddt(predicted_structure)

            with open(pdb_path + f'i{ni+1}_{basename}.pdb', 'w') as f:
                f.write(predicted_structure)

            #wrtire all sequences in a single file (MSA) 
            with open(f'{args.outpath}/{basename}_seqs.fasta', 'a') as f:
                f.write(f'>i{ni+1}_{basename} | T: {round(args.temperature, 2)} | seq_id: {round(native_seq_recovery, 2)} | lDDT {round(lddt, 2)} | pTM {round(ptm, 2)} \n')
                f.write(sampled_seq + '\n')

            #write an MSA with gaped sequences 
            if args.gaps > 0:
                gapped_sampled_seq = make_gaps(sampled_seq, lddt_list)
                with open(f'{args.outpath}/{basename}_seqs_gapped.fasta', 'a') as f:
                    f.write(f'>i{ni+1}_{basename} | T: {round(args.temperature, 2)} | seq_id: {round(native_seq_recovery, 2)} | lDDT {round(lddt, 2)} | pTM {round(ptm, 2)} \n')
                    f.write(gapped_sampled_seq + '\n') 

            #write a log file
            with open(args.outpath + '/' + args.log, 'a') as f:
                f.write(f'i{ni+1}_{basename}\t{round(args.temperature, 2)}\t{round(native_seq_recovery, 2)}\t{round(lddt, 2)}\t{round(ptm, 2)}\n')
            print(f'i{ni+1}_{basename}\t{round(args.temperature, 2)}\t{round(native_seq_recovery, 2)}\t{round(lddt, 2)}\t{round(ptm, 2)}')
        print(f'\n{args.num_iterations} sequences are generated')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Sample sequences based on a given structure.'
    )
    parser.add_argument(
            '-i', '--inputpdb', type=str,
            help='input filepath, either .pdb or .cif',
    )
    parser.add_argument(
            '-f', '--fetchpdb', type=str,
            help='fetch file from RCSB',
    )
    parser.add_argument(
            '-c', '--chain', type=str,
            help='chain ID for the chain of interest', 
            default='A',
    )
    parser.add_argument(
            '-o' ,'--outpath', type=str, 
            help='output filepath for saving sampled sequences',
            default='output',#change to output + input basename
    )
    parser.add_argument(
            '-t', '--temperature', type=float,
            help='IF sampling temperature, if 0 will set automaticaly ',
            default=0.,
    )
    parser.add_argument(
            '-Ns', '--num_samples', type=int,
            help='number of iterations',
            default=3,
    )
    parser.add_argument(
            '-Ni', '--num_iterations', type=int,
            help='number of sequences to sample per iteration',
            default=100,
    )
    parser.add_argument(
            '-cut', '--lddt_cutoff', type=int,
            help='cut-off of structures quality for next cycle',
            default=65,
    )
    parser.add_argument(
            '-spdb', '--savepdbs', type=bool, #test this
            help='save predicted PDB files',
            default=True,
    )
    parser.add_argument(
            '-warn', '--gen_warn', type=bool, #test this
            help='give warning if too many low pLDDT predictions',
            default=True,
    )
    parser.add_argument(
            '-g', '--gaps', type=float, 
            help='substitute a persent residues with lowest pLDDT with gaps [from 0 to 1]',
            default=0.0,
    )   
    parser.add_argument(
            '-l', '--log', type=str,
            help='log output',
            default='output.log',
    )


    args = parser.parse_args()


    #backup if output directory exists
    backup_output(args.outpath)

   
    #
    #check arguments and input paths before loading models 
    #

    #load models
    print('\nloading esm.pretrained.esmfold_v1... \n')
    model_v1 = esm.pretrained.esmfold_v1()
    model_v1 = model_v1.eval().cuda()

    print('loading esm.pretrained.esm_if1_gvp4_t16_142M_UR50... \n')
    model_if, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model_if = model_if.eval().cuda()


    inverse_cycle(model_if, model_v1, alphabet, args)
    
