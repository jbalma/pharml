#!/usr/bin/env python
############################################################


# http://zinc15.docking.org/substances.csv?count=all


############################################################


import os
import sys
import argparse


############################################################


def write_map_file(ligands,proteins,outdir="data/",map_name="dataset.map"):
    mapfn = outdir + "/map/" + map_name
    with open(mapfn,"w") as mapf:
        out = ""
        for pdbfn in proteins:
            for ligfn in ligands:
                # Write two input files, nhg before lig.
                out += "2 %s %s"%("../nhg/"+str(pdbfn)+".nhg", "../lig/"+str(ligfn))
                # Write one output, the bind state as 0.
                bind_float = 0.0 
                out += " 1 %f"%(bind_float)
                # Go ahead and add a text tag for bind / nobind.
                bind_tag = "nobind"
                out += " 1 %s\n"%(bind_tag)
            # Flush the output buffer once per protein.
            mapf.write(out)
            out = ""
    return mapfn


############################################################


if __name__ == "__main__":
    # Parse command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--lig_dir', type=str,   required=True,   help='Path to lig dir.')
    parser.add_argument('--pdbs',    type=str,   required=True,   help='List of PDB IDs.')
    parser.add_argument('--out',     type=str,   default="data",  help='Output directory.')
    parser.add_argument('--map_name',type=str,   default="dataset.map",  help='Name of resulting map file.')
    args = parser.parse_args()
    # Create any needed subdirectories in advance.
    for subdir in ("map",):
        if not os.path.exists(args.out+"/"+subdir):
            os.makedirs(args.out+"/"+subdir)
    # Get a list of all the ligands and PDB IDs.
    ligands = [ ligand for ligand in os.listdir(args.lig_dir) if ligand.endswith(".lig") ]
    print("Ligs:")
    for lig in ligands[:10]:
        print("  %s"%lig)
    if len(ligands) > 10:
        print("  [...]")
    pdbs = [ pdb for pdb in args.pdbs.split(",") if len(pdb) > 0 ]
    print("PDBs:")
    for pdb in pdbs[:10]:
        print("  %s"%pdb)
    if len(pdbs) > 10:
        print("  [...]")
    # Write the new map file.
    print("Writing map file:")
    write_map_file(ligands,pdbs,args.out,args.map_name)
    # Done.
    print("Success!")
    
    
############################################################
