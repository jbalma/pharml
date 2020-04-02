#!/usr/bin/python


from itertools import product
from mlvoxelizer import voxelize, ligify, graphify
from balancer import train_item, category


############################################################


def main():
    #
    # Voxelization Options
    #
    fdp = False  # Force-Directed Placement
    #
    # Start by making all needed .vox / .lig / .nhg data
    #   
    data_out_name = './dataset-C-graph'
    print('Saving new dataset to {0}'.format(data_out_name)) 
    notes = '''#
# protein_target       pdb_file      sdf_bind                           sdf_nobind        training_pair
#
# Carbonic_Anhydrase2  1bnm_ca2_ykp  1BNM_Validation_Affinities_3D.sdf  ca2_nobind.sdf    A
# Carbonic_Anhydrase4  3f7b_ca4_ykp  3F7B_Validation_Affinities_3D.sdf  ca4_nobind.sdf    A
# PDE4B                1y2j_ykp      1Y2J_Validation_Affinities_3D.sdf  pde4b_nobind.sdf  B
# PDE5                 2h42_ykp      2H42_Validation_Affinities_3D.sdf  pde5_nobind.sdf   B'''
    print("#---------------------------------------------------------------")
    print("# Notes:")
    print("#---------------------------------------------------------------")
    print("%s"%(notes))
    #
    a_ca2_nhg, a_ca2_lig_bind  = graphify("data/pdb/1bnm_ca2_ykp.pdb", outpath=data_out_name, fdp=fdp)
    a_ca2_lig_bind            += ligify(  "data/sdf/1BNM_Validation_Affinities_3D.sdf", outpath=data_out_name, fdp=fdp)
    a_ca2_lig_nobind           = ligify(  "data/sdf/ca2_nobind.sdf", outpath=data_out_name, fdp=fdp)
    a_ca4_nhg, a_ca4_lig_bind  = graphify("data/pdb/3f7b_ca4_ykp.pdb", outpath=data_out_name, fdp=fdp)
    a_ca4_lig_bind            += ligify(  "data/sdf/3F7B_Validation_Affinities_3D.sdf", outpath=data_out_name, fdp=fdp)
    a_ca4_lig_nobind           = ligify(  "data/sdf/ca4_nobind.sdf", outpath=data_out_name, fdp=fdp)
    # 
    b_pde4b_nhg, b_pde4b_lig_bind = graphify("data/pdb/1y2j_ykp.pdb", outpath=data_out_name, fdp=fdp)
    b_pde4b_lig_bind             += ligify(  "data/sdf/1Y2J_Validation_Affinities_3D.sdf", outpath=data_out_name, fdp=fdp)
    b_pde4b_lig_nobind            = ligify(  "data/sdf/pde4b_nobind.sdf", outpath=data_out_name, fdp=fdp)
    b_pde5_nhg, b_pde5_lig_bind   = graphify("data/pdb/2h42_ykp.pdb", outpath=data_out_name, fdp=fdp)
    b_pde5_lig_bind              += ligify(  "data/sdf/2H42_Validation_Affinities_3D.sdf", outpath=data_out_name, fdp=fdp)
    b_pde5_lig_nobind             = ligify(  "data/sdf/pde5_nobind.sdf", outpath=data_out_name, fdp=fdp)
    print("#---------------------------------------------------------------")
    print("# Notes:")
    print("#---------------------------------------------------------------")
    print("%s"%(notes))
    print("#---------------------------------------------------------------")
    print("# Proteins / Ligands:")
    print("#---------------------------------------------------------------")
    print("a_ca2_nhg          = %d"%(len(a_ca2_nhg)))
    print("a_ca2_lig_bind     = %d"%(len(a_ca2_lig_bind)))
    print("a_ca2_lig_nobind   = %d"%(len(a_ca2_lig_nobind)))
    print("a_ca4_nhg          = %d"%(len(a_ca4_nhg)))
    print("a_ca4_lig_bind     = %d"%(len(a_ca4_lig_bind)))
    print("a_ca4_lig_nobind   = %d"%(len(a_ca4_lig_nobind)))
    # 
    print("b_pde4b_nhg        = %d"%(len(b_pde4b_nhg)))
    print("b_pde4b_lig_bind   = %d"%(len(b_pde4b_lig_bind)))
    print("b_pde4b_lig_nobind = %d"%(len(b_pde4b_lig_nobind)))
    print("b_pde5_nhg         = %d"%(len(b_pde5_nhg)))
    print("b_pde5_lig_bind    = %d"%(len(b_pde5_lig_bind)))
    print("b_pde5_lig_nobind  = %d"%(len(b_pde5_lig_nobind)))
    #
    # Pair up the nhg and lig files to make train / test items
    #
    a_ca2_bind_items   = [ train_item(inputs=p, outputs=[1.0], tags=["a","ca2","bind"])   for p in product(a_ca2_nhg, a_ca2_lig_bind)   ]
    a_ca2_nobind_items = [ train_item(inputs=p, outputs=[0.0], tags=["a","ca2","nobind"]) for p in product(a_ca2_nhg, a_ca2_lig_nobind) ]
    a_ca4_bind_items   = [ train_item(inputs=p, outputs=[1.0], tags=["a","ca4","bind"])   for p in product(a_ca4_nhg, a_ca4_lig_bind)   ]
    a_ca4_nobind_items = [ train_item(inputs=p, outputs=[0.0], tags=["a","ca4","nobind"]) for p in product(a_ca4_nhg, a_ca4_lig_nobind) ]
    #
    b_pde4b_bind_items   = [ train_item(inputs=p, outputs=[1.0], tags=["a","pde4b","bind"])   for p in product(b_pde4b_nhg, b_pde4b_lig_bind)   ]
    b_pde4b_nobind_items = [ train_item(inputs=p, outputs=[0.0], tags=["a","pde4b","nobind"]) for p in product(b_pde4b_nhg, b_pde4b_lig_nobind) ]
    b_pde5_bind_items    = [ train_item(inputs=p, outputs=[1.0], tags=["a","pde5","bind"])    for p in product(b_pde5_nhg,  b_pde5_lig_bind)    ]
    b_pde5_nobind_items  = [ train_item(inputs=p, outputs=[0.0], tags=["a","pde5","nobind"])  for p in product(b_pde5_nhg,  b_pde5_lig_nobind)  ]
    #
    # Print stats on unbalanced data
    #
    print("#---------------------------------------------------------------")
    print("# Unbalanced data sets:")
    print("#---------------------------------------------------------------")
    print("a_ca2_bind     = %d"%(len(a_ca2_bind_items)))
    print("a_ca2_nobind   = %d"%(len(a_ca2_nobind_items)))
    print("a_ca4_bind     = %d"%(len(a_ca4_bind_items)))
    print("a_ca4_nobind   = %d"%(len(a_ca4_nobind_items)))
    #
    print("b_pde4b_bind   = %d"%(len(b_pde4b_bind_items)))
    print("b_pde4b_nobind = %d"%(len(b_pde4b_nobind_items)))
    print("b_pde5_bind    = %d"%(len(b_pde5_bind_items)))
    print("b_pde5_nobind  = %d"%(len(b_pde5_nobind_items)))
    #
    # With .nhg/.lig files from PDB/SDF data, build balanced data sets.
    #
    print("#---------------------------------------------------------------")
    print("# Balanced data sets:")
    print("#---------------------------------------------------------------")
    a_ca2        = category(name="a_ca2",        frac=1.0, shuffle=True)
    a_ca2_bind   = category(name="a_ca2_bind",   frac=0.5, parent=a_ca2, items=a_ca2_bind_items, shuffle=True)
    a_ca2_nobind = category(name="a_ca2_nobind", frac=0.5, parent=a_ca2, items=a_ca2_nobind_items, shuffle=True)
    a_ca2.balance()
    print("a_ca2            = %d"%(a_ca2.bcount))
    print("  a_ca2_bind     = %d"%(a_ca2_bind.bcount))
    print("  a_ca2_nobind   = %d"%(a_ca2_nobind.bcount))
    a_ca4        = category(name="a_ca4",        frac=1.0, shuffle=False)
    a_ca4_bind   = category(name="a_ca4_bind",   frac=0.5, parent=a_ca4, items=a_ca4_bind_items, shuffle=True)
    a_ca4_nobind = category(name="a_ca4_nobind", frac=0.5, parent=a_ca4, items=a_ca4_nobind_items, shuffle=True)
    a_ca4.balance()
    print("a_ca4            = %d"%(a_ca4.bcount))
    print("  a_ca4_bind     = %d"%(a_ca4_bind.bcount))
    print("  a_ca4_nobind   = %d"%(a_ca4_nobind.bcount))
    b_pde4b        = category(name="b_pde4b",        frac=1.0, shuffle=True)
    b_pde4b_bind   = category(name="b_pde4b_bind",   frac=0.5, parent=b_pde4b, items=b_pde4b_bind_items, shuffle=True)
    b_pde4b_nobind = category(name="b_pde4b_nobind", frac=0.5, parent=b_pde4b, items=b_pde4b_nobind_items, shuffle=True)
    b_pde4b.balance()
    print("b_pde4b          = %d"%(b_pde4b.bcount))
    print("  b_pde4b_bind   = %d"%(b_pde4b_bind.bcount))
    print("  b_pde4b_nobind = %d"%(b_pde4b_nobind.bcount))
    b_pde5        = category(name="b_pde5",        frac=1.0, shuffle=False)
    b_pde5_bind   = category(name="b_pde5_bind",   frac=0.5, parent=b_pde5, items=b_pde5_bind_items, shuffle=True)
    b_pde5_nobind = category(name="b_pde5_nobind", frac=0.5, parent=b_pde5, items=b_pde5_nobind_items, shuffle=True)
    b_pde5.balance()
    print("b_pde5           = %d"%(b_pde5.bcount))
    print("  b_pde5_bind    = %d"%(b_pde5_bind.bcount))
    print("  b_pde5_nobind  = %d"%(b_pde5_nobind.bcount))
    #
    # Write the balanced data sets (and all unbalanced) out to .map files
    #
    a_ca2.write(path=data_out_name)
    a_ca2_bind.write(balanced=False,path=data_out_name)
    a_ca2_nobind.write(balanced=False,path=data_out_name)
    #
    a_ca4.write(path=data_out_name)
    a_ca4_bind.write(balanced=False,path=data_out_name)
    a_ca4_nobind.write(balanced=False,path=data_out_name)
    #
    b_pde4b.write(path=data_out_name)
    b_pde4b_bind.write(balanced=False,path=data_out_name)
    b_pde4b_nobind.write(balanced=False,path=data_out_name)
    #
    b_pde5.write(path=data_out_name)
    b_pde5_bind.write(balanced=False,path=data_out_name)
    b_pde5_nobind.write(balanced=False,path=data_out_name)


if __name__ == "__main__":
    main()


############################################################
