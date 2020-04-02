#!/usr/bin/env python
############################################################


# http://www.bindingdb.org/bind/downloads/BindingDB_All_terse_2D_2019m4.sdf.zip


############################################################


import os
import sys
import argparse
import math
import warnings
import collections
import dask
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from scipy import spatial
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools
from rdkit import RDLogger
from Bio.PDB import *


############################################################


IC50_cutoff = 10000     # nM
NHGD_cutoff = 4         # A
BATCH_SIZE  = 512       # PDBs
max_ptn_sz  = 10000     # atoms
min_ptn_sz  = 500       # atoms
INFERENCE_ONLY = True   # no affinity data, all labelled no-bind

############################################################


NAME_TO_CHARGE = {
    'H':1,
    'D':1,   #Deuterium?
    'C':6, 
    'N':7, 
    'O':8, 
    'F':9, 
    'G':12,  #Mg
    'P':15, 
    'S':16, 
    'L':17,  #Cl
    'Z':30,  #Zn
    'E':34,  #Se
    'R':35   #Br
}


def atomic_charge(aname):
    # Map first letter of atom name from PDB to an atomic charge.
    atype = aname[0]
    if atype not in NAME_TO_CHARGE:
        print("Error: Unknown atom type: '%s' from '%s'!"%(atype,aname))
        sys.exit()
    return NAME_TO_CHARGE[atype]

    
def write_nhg_file(atoms,edges,nhgfn):
    with open(nhgfn,"w") as nhgf:
        # Write num atoms and num edges
        out = [ len(edges), len(atoms) ]
        out = np.array(out, dtype=np.int32)
        out.tofile(nhgf)
        # Write each atom as a 5-tuple: (t,b,x,y,z)
        out = []
        for atom in atoms:
            apos = atom.get_vector()
            atype = atomic_charge(atom.get_name())
            out += [ atype, 0.0, apos[0], apos[1], apos[2] ]
        out = np.array(out, dtype=np.float32)
        out.tofile(nhgf)
        # Write each edge as a 3-tuple: (d,ndx_i,ndx_j)
        out = []
        for edge in edges:
            out += [ edge[2], edge[0], edge[1] ]
        out = np.array(out, dtype=np.float32)
        out.tofile(nhgf)


def write_lig_file(mol,ligfn):
    with open(ligfn,"w") as ligf:
        atoms = mol.GetAtoms();
        # Write number of atoms
        ligf.write("%d\n"%(len(atoms)))
        # Write atomic charges
        for atom in atoms:
            ligf.write("%s "%(str(atom.GetAtomicNum())))
        ligf.write("\n")
        # Write formal charges
        for atom in atoms:
            ligf.write("%s "%(str(atom.GetFormalCharge())))
        ligf.write("\n")
        # Write bond matrix
        for ndx_a in range(0, len(atoms)):
            for ndx_b in range(0, len(atoms)):
                bond = mol.GetBondBetweenAtoms(ndx_a, ndx_b)
                if bond != None:
                    if bond.GetIsAromatic():
                        ligf.write("4")
                    else:
                        btype = str(bond.GetBondType())
                        if btype == "SINGLE":
                            btype = "1"
                        elif btype == "DOUBLE":
                            btype = "2"
                        elif btype == "TRIPLE":
                            btype = "3"
                        ligf.write(str(btype))
                else:
                    ligf.write("0")                    
                ligf.write(" ")
            ligf.write("\n")
        # Write (empty) distance matrix
        for ndx_a in range(0, len(atoms)):
            for ndx_b in range(0, len(atoms)):
                ligf.write("0 ")
            ligf.write("\n")


def write_map_file(ligands,proteins,outdir="data/"):
    mapfn = outdir + "/map/dataset.map"
    bind_stats = {}
    with open(mapfn,"w") as mapf:
        out = ""
        for pdbid in proteins:
            n_bind   = 0
            n_nobind = 0
            for ligid in proteins[pdbid]:
                # Get the ligand and convert to bind / nobind.
                lig = ligands[ligid]
                ic50_str = str(lig[2])
                if INFERENCE_ONLY != True:
                    bind = True if ic50_str[0] != '>' and float(ic50_str[1:]) <= IC50_cutoff else False
                else:
                    bind = np.random.randint(2, size=1)[0]
                n_bind   += int(bind)
                n_nobind += int(not bind)
                # Write two input files, nhg before lig.
                out += "2 %s %s"%("../nhg/"+str(pdbid)+".nhg", "../lig/lig"+str(ligid)+".lig")
                # Write one output, the bind state as 1 / 0.
                bind_float = 1.0 if bind == True else 0.0
                out += " 1 %f"%(bind_float)
                # Go ahead and add a text tag for bind / nobind.
                bind_tag = "bind" if bind == True else "nobind"
                out += " 1 %s\n"%(bind_tag)
            # Flush the output buffer once per protein.
            mapf.write(out)
            out = ""
            bind_stats[pdbid] = (n_bind, n_nobind)
    return bind_stats


############################################################

def load_csv(csv_file_name):
    print("Loading CSV.")
    # Parse the CSV file.
    rdk_lg = RDLogger.logger()
    rdk_lg.setLevel(RDLogger.CRITICAL)
    with open(csv_file_name,"r") as csvf:
        ligands = [ list(line.split(",")) for line in csvf.read().split("\n") ]
    # Convert to mol objects.
    print("Converting ligands to mol objects.")
    valid_ligands = []
    for ndx,ligand in enumerate(ligands):
        if len(ligand) == 1 and ligand[0] == "":
            continue
        if len(ligand) != 2:
            print(ligand)
            continue
        ligand.append(Chem.MolFromSmiles(ligand[1]))
        valid_ligands.append(ligand)
        if ndx < 10:
            print(ligand)
        elif ndx == 10:
            print("...")
    print("Done creating mol objects.")
    return valid_ligands


def split_sdf(file_name,outdir="data/"):

    if ".sdf" in file_name:
        print("Loading sdf.")

        rdk_lg = RDLogger.logger()
        rdk_lg.setLevel(RDLogger.CRITICAL)
        df = PandasTools.LoadSDF(sdf_file_name,
                             smilesName='SMILES',
                             molColName='Molecule',
                             includeFingerprints=False)

    if ".csv" in file_name:
        print("Loading CSV.")
        # Parse the CSV file.
        rdk_lg = RDLogger.logger()
        rdk_lg.setLevel(RDLogger.CRITICAL)
        with open(file_name,"r") as csvf:
            pdb_list = [ list(line.split(",")) for line in csvf.read().split("\n") ]
        df = pd.DataFrame(columns=pdb_list[0].append('Molecule'))
        for pdb in pdb_list[1:-1]:
            print("pdb=",pdb)
            df = df.append({'PDB ID':pdb}, ignore_index=True)
    print("Raw cols = ", [str(x) for x in df.columns])
    # Select only the needed columns and merge the two PDB cols.
    #df_list=['PDB ID(s) for Ligand-Target Complex','PDB ID(s) of Target Chain','SMILES','IC50 (nM)','Molecule']
    df_list=['PDB ID']
    df_selected = df[df_list].copy()
    #df_selected["PDB IDs"] = df_selected['PDB ID(s) for Ligand-Target Complex'] + ',' + df_selected['PDB ID(s) of Target Chain']
    print("Selected cols = ", [str(x) for x in df_selected.columns])
    #df_selected = df_selected[ ["PDB IDs"] + df_list[2:] ]
    # Drop any rows with missing data.
    df_selected = df_selected.replace('',  np.nan)
    df_selected = df_selected.replace(',', np.nan)
    df_selected = df_selected.dropna()
    r_rows = len(df.index)
    s_rows = len(df_selected.index)
    print("Raw rows = ", r_rows)
    print("Sel rows = ", s_rows)
    print("Keep pct = %.2f%s"%(((float(s_rows)/float(r_rows))*100.0),'%'))
    # Build ligand dictionary and a protein dictionary.
    print("Building protein-ligand dictionary.")
    uligs = {}
    prots_ligs = {}
    for lndx,row in enumerate(df_selected.values):
        print("row[0]=",row[0])
        pdbs = row[0][0].split(',')
        for pdb in pdbs:
            if pdb == '':
                continue
            if pdb not in prots_ligs:
                prots_ligs[pdb] = []
            prots_ligs[pdb] += [ lndx ]
        uligs[ lndx ] = row
    print("Unique proteins = ", len(prots_ligs))
    print("Writing per-ligand output files.")
    # Write out .lig files and return the data dictionaries.
    for key in uligs:
        ndx = str(key)
        lig = uligs[key]
        write_lig_file(lig[3],outdir+"/lig/lig%s.lig"%ndx)
    return uligs, prots_ligs


############################################################
def split_pdb_with_sdf(pdb_id,sdf_file_name,outdir="data/"):
# This function takes in a PDB-list from csv (from rcsb.org)
# alongside an sdf file containing compounds to test against every structure in the pdb-list

    print("Loading sdf from ", sdf_file_name)

    rdk_lg = RDLogger.logger()
    rdk_lg.setLevel(RDLogger.CRITICAL)
    df = PandasTools.LoadSDF(sdf_file_name,
                             smilesName='SMILES',
                             molColName='Molecule',
                             includeFingerprints=False)
    PandasTools.AddMoleculeColumnToFrame(df,'SMILES','PDB ID',includeFingerprints=False)
    # Select only the needed columns and merge the two PDB cols.
    df_sdf_list = ['PDB ID','Molecule','FDA drugnames']
    df_selected = df[df_sdf_list].copy()
    print("Selected SDF cols = ", [str(x) for x in df_selected.columns])

    print("Loading compounds for test against PDB ID = ", pdb_id)
    #with open(pdb_list_file_name,"r") as csvf:
    #    pdb_list = [ list(line.split(",")) for line in csvf.read().split("\n") ]
    #    df = pd.DataFrame(columns=pdb_list[0].append('Molecule'))
    for mol in df_selected['Molecule']:
        print("pdb=",pdb_id, ",Molecule=", mol)
        df = df.append({'PDB ID':pdb_id}, ignore_index=True)
        
    print("Raw PDB file cols = ", [str(x) for x in df.columns])
    # Select only the needed columns and merge the two PDB cols.
    df_selected = df[df_sdf_list].copy()
    #df_selected["PDB IDs"] = df_selected['PDB ID(s) for Ligand-Target Complex'] + ',' + df_selected['PDB ID(s) of Target Chain']
    print("Selected PDB cols = ", [str(x) for x in df_selected.columns])
    #df_selected = df_selected[ ["PDB IDs"] + df_list[2:] ]
    # Drop any rows with missing data.
    df_selected = df_selected.replace('',  np.nan)
    df_selected = df_selected.replace(',', np.nan)
    df_selected = df_selected.dropna()
    r_rows = len(df.index)
    s_rows = len(df_selected.index)
    print("Raw rows = ", r_rows)
    print("Sel rows = ", s_rows)
    print("Keep pct = %.2f%s"%(((float(s_rows)/float(r_rows))*100.0),'%'))
    # Build ligand dictionary and a protein dictionary.
    print("Building protein-ligand dictionary.")
    uligs = {}
    prots_ligs = {}
    for lndx,row in enumerate(df_selected.values):
        print("row=",row)
        pdbs = [pdb_id] #row[0].split(',')
        for pdb in pdbs:
            if pdb == '':
                continue
            if pdb not in prots_ligs:
                prots_ligs[pdb] = []
            prots_ligs[pdb] += [ lndx ]
        uligs[ lndx ] = row
    print("Unique proteins = ", len(prots_ligs))
    print("Writing per-ligand output files.")
    # Write out .lig files and return the data dictionaries.
    for key in uligs:
        ndx = str(key)
        lig = uligs[key]
        write_lig_file(lig[1],outdir+"/lig/lig%s.lig"%ndx)
    return uligs, prots_ligs




############################################################


def convert_pdb(pdbid,outdir="data/"):
    # Print header for each protein and download if needed.
    status = "---------------- %s ----------------\n"%pdbid
    bp_pdbl = PDBList(verbose=False)
    bp_parser = MMCIFParser()
    bp_pdbl.retrieve_pdb_file(pdbid,pdir=outdir+"/pdb/",file_format="mmCif")
    pdb_fn = outdir+"/pdb/"+pdbid.lower()+".cif"
    # Parse the PDB / CIF file into a structure object.
    structure = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            structure = bp_parser.get_structure("",pdb_fn)
        except Exception as ex:
            status += "!! Failed to parse '%s', skipping 'invalid' PDB ('%s').\n"%(pdb_fn,str(ex))
            return (pdbid, False, status)
    # Traverse the PDB structure, looking for the protein chain.
    status += "Models: %d\n"%(len(structure))
    chain_A = []
    for mndx, model in enumerate(structure):
        status += "  models[%d]: %d chains\n"%(mndx,len(model))
        chain = None
        if 'A' in model:
            chain = model['A']
            status += "    Selected chain: 'A'\n"
        else:
            for c in model:
                chain = c
                break
            status += "    Selected chain: first ('%s')\n"%(chain.get_id())
        atoms = []
        for residue in chain:
            resid = residue.get_full_id()
            if resid[3][0] == "W" or (resid[3][0][0] == "H" and resid[3][0][1] == "_"):
                continue                    
            for atom in residue:
                atoms.append( atom )
        chain_A = atoms
        # !!av: For now, just consider the first model.
        break
    status += "    Selected chain len: %d\n"%(len(chain_A))
    # Convert the selected chain into a NHG.
    status += "Building neighborhood graph.\n"   
    try:
        # Get an array of the atom positions.
        positions = []
        for atom in chain_A:
            positions += list(atom.get_vector()) 
        positions = np.array(positions, dtype=np.float32)
        positions = np.reshape(positions, (len(chain_A),3))
        if positions.shape[0] > max_ptn_sz:
            raise Exception('Protein too large (%d)!'%(positions.shape[0]))
        if positions.shape[0] < min_ptn_sz:
            raise Exception('Protein too small (%d)!'%(positions.shape[0]))
        # Find pairwise distances between all atoms.
        distances = spatial.distance.pdist(positions)
        distances = spatial.distance.squareform(distances)
        # Find local neighborhoods by removing long (and self) distances.
        distances[distances == 0.0] = 1.0 + NHGD_cutoff
        neighborhoods = np.nonzero(distances <= NHGD_cutoff)
        # Turn the distances / indicies into an explicit edge list.
        nh_edges = [ (ndx_a, ndx_b, distances[ndx_a][ndx_b]) for ndx_a, ndx_b in zip(*neighborhoods) ]
        # Examine the local neighborhood sizes to check connectivity.
        local_nh_sizes = { ndx:0 for ndx in np.unique(neighborhoods[0]) }
        for ndx in neighborhoods[0]:
            local_nh_sizes[ndx] += 1
        if min(local_nh_sizes.values()) < 3:
            raise Exception('Too few local edges (%d)!'%(min(local_nh_sizes.values())))
        status += "Neighborhood graph info:\n"
        status += "  Total edges: %d"%(len(nh_edges))
        status += "  Local edges:\n"
        status += "    Min edges: %d\n"%(min(local_nh_sizes.values()))
        status += "    Max edges: %d\n"%(max(local_nh_sizes.values()))
    except Exception as ex:
        # Return a failure if anything went wrong.
        status += "!! Neighborhood graph failed for '%s', skipping PDB ('%s').\n"%(pdb_fn,str(ex))
        return (pdbid, False, status)
    # Write the output NHG and return success.
    status += "Writing neighborhood graph for '%s'.\n"%(pdbid)
    write_nhg_file(chain_A,nh_edges,outdir+"/nhg/%s.nhg"%(pdbid))
    status += "Done converting PDB.\n"
    return (pdbid, True, status)


def chunks(lst, size):
    # Turn the list into cunks of size 'size'.
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def convert_pdbs(proteins,outdir="data/",nworkers=16):
    print("Processing PDBs.")
    # Maintain a list of rejected PDBs for later.
    rejected_pdbs = []
    # Split all the PDBs into batches for Dask.
    pdbids = [ str(key) for key in proteins ]
    batches = [ batch for batch in chunks(pdbids, BATCH_SIZE) ]
    for bndx, batch in enumerate(batches):
        print("PDB batch %d of %d (%.2f%s):"%(bndx+1,len(batches),float(bndx)/len(batches)*100.0,"%"))
        # Create lsit of Dask tasks and launch them with threads.
        results = [ dask.delayed(convert_pdb)(pdbid,outdir) for pdbid in batch ]
        with ProgressBar(dt=0.5):
            results = dask.compute(*results, scheduler='threads', num_workers=nworkers)
        # Process the results from the batch.
        for pdbid, flag, status in results:
            print("%s"%(status),end="")
            if not flag:
                rejected_pdbs.append(pdbid)
    # Finished, so print stats and return the rejected list.
    print("--------------------------------------")
    print("Done converting PDBs:")
    print("  Rejected: %d/%d"%(len(rejected_pdbs),len(proteins)))
    print("  Accepted: %.2f%s"%(float(len(proteins)-len(rejected_pdbs))*100.0/float(len(proteins)),'%'))
    return rejected_pdbs


############################################################


if __name__ == "__main__":
    # Parse command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_id',  type=str,   required=True,   help='Name of PDB structure.')
    parser.add_argument('--sdf',     type=str,   required=True,   help='Path to SDF file.')
    parser.add_argument('--out',     type=str,   default="data",  help='Output directory.')
    parser.add_argument('--threads', type=int,   default=16,      help='Number of threads for PDB processing.')
    parser.add_argument('--ic50',    type=float, default=10000.0, help='IC50 max cutoff for bind in nM.')
    parser.add_argument('--nhgr',    type=float, default=4.0,     help='Local NHG radius in A.')
    args = parser.parse_args()
    IC50_cutoff = args.ic50
    NHGD_cutoff = args.nhgr
    # Create any needed subdirectories in advance.
    for subdir in ("pdb", "nhg", "lig", "map"):
        if not os.path.exists(args.out+"/"+subdir):
            os.makedirs(args.out+"/"+subdir)
    # First parse and process the SDF file for ligands and PDB-IDs.
    #ligands, proteins = split_sdf(args.sdf,outdir=args.out)
    ligands, proteins = split_pdb_with_sdf(args.pdb_id, args.sdf, outdir=args.out)
    # Next download and process the PDB files for the proteins.
    rejected = convert_pdbs(proteins,outdir=args.out,nworkers=args.threads)
    # Write out a map file listing all the resultant data items.
    print("Writing map file for the dataset:")
    rejected = { pdbid:ndx for ndx,pdbid in enumerate(rejected) }
    accepted = {}
    for pdbid in proteins:
        if pdbid not in rejected:
            accepted[pdbid] = proteins[pdbid]
    stats = write_map_file(ligands,accepted)
    # Get some stats / distributions.
    bind = 0
    nobind = 0
    for pdb in stats:
        bind += stats[pdb][0]
        nobind += stats[pdb][1]
    print("  Proteins:     %d"%(len(accepted)))
    print("  Ligands:      %d"%(len(ligands)))
    print("  Bind pairs:   %d"%(bind))
    print("  Nobind pairs: %d"%(nobind))
    dist = [ stats[pdb][0]+stats[pdb][1] for pdb in stats ]
    dist.sort()
    max_count = max(dist)
    dist = collections.Counter(dist)
    print("Writing distribution of ligand counts to 'pdb_dist.dat'.")
    with open("pdb_dist.dat","w") as distf:
        for count in range(1,max_count+1):
            num = dist[count] if count in dist else 0
            distf.write("%d %d\n"%(count,num))
    print("Success!")
    
    
############################################################
