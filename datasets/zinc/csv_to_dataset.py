#!/usr/bin/env python
############################################################


# http://zinc15.docking.org/substances.csv?count=all


############################################################


import os
import sys
import argparse
import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools
from rdkit import RDLogger


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


def write_ligands(ligands,outdir="data/"):
    print("Writing %d .lig files."%(len(ligands)))
    for ligand in ligands:
        write_lig_file(ligand[2],outdir+"/lig/"+ligand[0]+".lig")
    print("Done writing %d files."%(len(ligands)))


############################################################


if __name__ == "__main__":
    # Parse command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     type=str,   required=True,   help='Path to CSV file.')
    parser.add_argument('--out',     type=str,   default="data",  help='Output directory.')
    args = parser.parse_args()
    # Create any needed subdirectories in advance.
    for subdir in ("lig",):
        if not os.path.exists(args.out+"/"+subdir):
            os.makedirs(args.out+"/"+subdir)
    # Read input CSV file.
    ligands = load_csv(args.csv)
    # Write out as .lig files.
    write_ligands(ligands,outdir=args.out)
    # Done.
    print("Success!")
    
    
############################################################
