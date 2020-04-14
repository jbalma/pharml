#!/usr/bin/env python
############################################################


import argparse
from random import shuffle
from readvox import read_map


############################################################


def write_map_file(items,outfn):
    with open(outfn,"w") as mapf:
        out = ""
        for indx,item in enumerate(items):
            # Parse and print some examples.
            for sndx,section in enumerate(item):
                if sndx != 0:
                    out += " "
                out += "%d"%(len(section))
                for subitem in section:
                    out += " %s"%(subitem)
            out += "\n"
            # Flush the output buffer.
            if indx % 100 == 0:
                mapf.write(out)
                out = ""
        # Flush buffer if needed.
        if out != "":
            mapf.write(out)
    return


def split_map_file(mapfn, split_pct, out_prefix="split"):
    # Read the input map file.
    print("Reading input map file '%s'."%(mapfn))
    in_map = read_map(mapfn)
    print("Read %d items:"%(len(in_map)))
    ligands = {}
    proteins = {}
    binds = 0
    nobinds = 0
    for ndx,item in enumerate(in_map):
        # Parse and print some examples.
        inputs, outputs, tags = item
        nhg_fn, lig_fn = inputs
        bind = outputs[0]
        tag = tags[0]
        if ndx < 10:
            print("  %s"%str( (nhg_fn, lig_fn, bind, tag) ))
        elif ndx == 10:
            print("  ...")
        # Add to list of proteins and ligands.
        if nhg_fn not in proteins:
            proteins[nhg_fn] = []
        proteins[nhg_fn].append( item )
        if lig_fn not in ligands:
            ligands[lig_fn] = []
        ligands[lig_fn].append( item )
        # Add to bind / nobind counts.
        if float(bind) == 1.0:
            binds += 1
        else:
            nobinds += 1
    # Print some stats.
    print("Unique protein files: %d"%(len(proteins)))
    print("Unique ligand files:  %d"%(len(ligands)))
    print("Bind pairs:   %d"%(binds))
    print("Nobind pairs: %d"%(nobinds))
    # Split along the "protein axis".
    split_point = (split_pct/100.0) * len(proteins)
    print("Splitting in two at ndx %d."%(split_point))
    p_keys = [ key for key in proteins ]
    shuffle(p_keys)
    a_bind = []
    a_nobind = []
    b_bind = []
    b_nobind = []
    for ndx,protein in enumerate(p_keys):
        # Pick a target to dump into.
        t_bind = a_bind
        t_nobind = a_nobind        
        if ndx >= split_point:
            t_bind = b_bind
            t_nobind = b_nobind
        # Add to the correct section.
        for item in proteins[protein]:
            inputs, outputs, tags = item
            nhg_fn, lig_fn = inputs
            bind = outputs[0]
            tag = tags[0]
            if float(bind) == 1.0:
                t_bind.append(item)
            else:
                t_nobind.append(item)
    # Print some split results before bind / nobind balancing.            
    print("Chunk A:   %d (%.2f%s)"%(len(a_bind)+len(a_nobind),(len(a_bind)+len(a_nobind))*100.0/(binds+nobinds),'%'))
    print("  Binds:   %d"%(len(a_bind)))
    print("  Nobinds: %d"%(len(a_nobind)))
    print("Chunk B:   %d (%.2f%s)"%(len(b_bind)+len(b_nobind),(len(b_bind)+len(b_nobind))*100.0/(binds+nobinds),'%'))
    print("  Binds:   %d"%(len(b_bind)))
    print("  Nobinds: %d"%(len(b_nobind)))
    # Now balance within A and within B for bind / nobind.
    print("Balancing bind / nobind within A and within B.")
    a_min_size = min(len(a_bind),len(a_nobind))
    b_min_size = min(len(b_bind),len(b_nobind))
    shuffle(a_bind)
    shuffle(a_nobind)
    shuffle(b_bind)
    shuffle(b_nobind)
    a_bind = a_bind[:a_min_size]
    a_nobind = a_nobind[:a_min_size]
    b_bind = b_bind[:b_min_size]
    b_nobind = b_nobind[:b_min_size]
    binds = len(a_bind) + len(b_bind)
    nobinds = len(a_nobind) + len(b_nobind)
    print("Bind pairs:   %d"%(binds))
    print("Nobind pairs: %d"%(nobinds))
    print("Chunk A:   %d (%.2f%s)"%(len(a_bind)+len(a_nobind),(len(a_bind)+len(a_nobind))*100.0/(binds+nobinds),'%'))
    print("  Binds:   %d"%(len(a_bind)))
    print("  Nobinds: %d"%(len(a_nobind)))
    print("Chunk B:   %d (%.2f%s)"%(len(b_bind)+len(b_nobind),(len(b_bind)+len(b_nobind))*100.0/(binds+nobinds),'%'))
    print("  Binds:   %d"%(len(b_bind)))
    print("  Nobinds: %d"%(len(b_nobind)))
    # Write out the new split and balanced mapfiles.
    print("Writing new map output files.")
    print("  Merge binds/nobinds and shuffle.")
    dataset_a = a_bind + a_nobind
    dataset_b = b_bind + b_nobind
    shuffle(dataset_a)
    shuffle(dataset_b)
    print("  %s"%(out_prefix+"_a.map"))
    write_map_file(dataset_a,out_prefix+"_a.map")
    print("  %s"%(out_prefix+"_b.map"))
    write_map_file(dataset_b,out_prefix+"_b.map")
    # Stats for the output.
    ligands = {}
    proteins = {}
    binds = 0
    nobinds = 0
    for ndx,item in enumerate(dataset_a+dataset_b):
        # Parse and print some examples.
        inputs, outputs, tags = item
        nhg_fn, lig_fn = inputs
        bind = outputs[0]
        tag = tags[0]
        if ndx < 10:
            print("  %s"%str( (nhg_fn, lig_fn, bind, tag) ))
        elif ndx == 10:
            print("  ...")
        # Add to list of proteins and ligands.
        if nhg_fn not in proteins:
            proteins[nhg_fn] = []
        proteins[nhg_fn].append( item )
        if lig_fn not in ligands:
            ligands[lig_fn] = []
        ligands[lig_fn].append( item )
        # Add to bind / nobind counts.
        if float(bind) == 1.0:
            binds += 1
        else:
            nobinds += 1
    print("Unique protein files: %d"%(len(proteins)))
    print("Unique ligand files:  %d"%(len(ligands)))
    print("Bind pairs:   %d"%(binds))
    print("Nobind pairs: %d"%(nobinds))
    return
    

if __name__ == "__main__":
    # Parse command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str,   required=True,   help='Path to map file.')
    parser.add_argument('--out', type=str,   default="split", help='Output file prefix.')
    parser.add_argument('--pct', type=float, default=20.0,    help='Split point (percent).')
    args = parser.parse_args()
    # Split the map file!
    split_map_file(args.map, args.pct, out_prefix=args.out)
    print("Success!")
    
    
############################################################
