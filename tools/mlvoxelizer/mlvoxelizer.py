#!/usr/bin/python


import os
import shutil
import re


############################################################


def find_files(spath,extension):
    # Return list of files with specified extension
    found = []
    for root, dirs, files in os.walk(spath):
        found = found + [ os.path.abspath(root)+"/"+f for f in files if f.endswith(extension) ]
    return found


############################################################


def voxelize(pdbfn,outpath="./data/",resolution=1.0,fdp=False,rotations=1,threads=4,win=20,stride=3,valid='ligand'):
    #
    # Create a temp dir to hold outputs
    #
    path, fn = os.path.split(pdbfn)
    tdir = outpath+"/"+fn+".voxels"
    if not os.path.exists(tdir):
        os.makedirs(tdir)
    #
    # Voxelize
    #
    if rotations > 0:
        for i in range(rotations):
            print("#---------------------------------------------------------------")
            print("# PDB \"%s\" Rotation %d of %d"%(fn,i+1,rotations))
            print("#---------------------------------------------------------------")
            os.system(("OMP_NUM_THREADS=%d ./mlvoxelizer --out %s --valid " + valid + " --protonate --pdb %s --res %f --rotate --win %dx%dx%dx --stride %d %s")%(threads, tdir, "%s/%s"%(path,fn), resolution, win, win, win, stride, "--fdp" if fdp else ""))
    else:
            print("#---------------------------------------------------------------")
            print("# PDB \"%s\""%(fn))
            print("#---------------------------------------------------------------")
            os.system(("OMP_NUM_THREADS=%d ./mlvoxelizer --out %s --valid "+ valid +" --protonate --pdb %s --res %f --win %dx%dx%dx --stride %d %s")%(threads, tdir, "%s/%s"%(path,fn), resolution, win, win, win, stride, "--fdp" if fdp else ""))
                  
    #
    # Scan the outputs to see what was produced
    #
    vox_files = find_files(tdir,".vox")
    lig_files = find_files(tdir,".lig")
    #
    # Copy the outputs to the desired output dir
    #
    vox_out = outpath + "/vox/"
    lig_out = outpath + "/lig/"

    if not os.path.exists(vox_out):
        os.makedirs(vox_out)

    if not os.path.exists(lig_out):
        os.makedirs(lig_out)

    
    vox_out = []
    for vox_file in vox_files:
        out = outpath+"/vox/"+os.path.basename(vox_file)
        shutil.copyfile(vox_file, out)
        vox_out.append(out)
    lig_out = []    
    for lig_file in lig_files:
        out = outpath+"/lig/"+os.path.basename(lig_file)
        shutil.copyfile(lig_file, out)
        lig_out.append(out)
    #
    # Cleanup the temp files
    #
    shutil.rmtree(tdir)
    #
    # Return a list of the found vox / lig files.
    #
    return (vox_out, lig_out)


def graphify(pdbfn,outpath="./data/",fdp=False):
    #
    # Create a temp dir to hold outputs
    #
    path, fn = os.path.split(pdbfn)
    tdir = outpath+"/"+fn+".graph"
    if not os.path.exists(tdir):
        os.makedirs(tdir)
    #
    # Turn into annotated neighborhood graph.
    #
    print("#---------------------------------------------------------------")
    print("# PDB \"%s\""%(fn))
    print("#---------------------------------------------------------------")
    os.system(("./mlvoxelizer --out %s --protonate --pdb %s --res 1.0 --win 1x1x1 --stride 100 %s")%(tdir, "%s/%s"%(path,fn), "--fdp" if fdp else ""))
                  
    #
    # Scan the outputs to see what was produced
    #
    nhg_files = find_files(tdir,".nhg")
    lig_files = find_files(tdir,".lig")
    #
    # Copy the outputs to the desired output dir
    #
    nhg_out = outpath + "/nhg/"
    lig_out = outpath + "/lig/"

    if not os.path.exists(nhg_out):
        os.makedirs(nhg_out)

    if not os.path.exists(lig_out):
        os.makedirs(lig_out)

    nhg_out = []
    for nhg_file in nhg_files:
        out = outpath+"/nhg/"+os.path.basename(nhg_file)
        shutil.copyfile(nhg_file, out)
        nhg_out.append(out)
    lig_out = []    
    for lig_file in lig_files:
        out = outpath+"/lig/"+os.path.basename(lig_file)
        shutil.copyfile(lig_file, out)
        lig_out.append(out)
    #
    # Cleanup the temp files
    #
    shutil.rmtree(tdir)
    #
    # Return a list of the found vox / lig files.
    #
    return (nhg_out, lig_out)


def ligify(sdffn,outpath="./data/",fdp=False):
    #
    # Create a temp dir to hold outputs
    #
    path, fn = os.path.split(sdffn)
    print("#---------------------------------------------------------------")
    print("# SDF \"%s\" Ligand %d"%(fn,1))
    print("#---------------------------------------------------------------")
    tdir = outpath+"/"+fn+".ligands"
    if not os.path.exists(tdir):
        os.makedirs(tdir)
    #
    # Extract first .lig file from .sdf
    #
    os.system("./mlvoxelizer --out %s --sdf %s --protonate --ligndx %d %s | tee %s/log"%
              (tdir, "%s/%s"%(path,fn), 0, "--fdp" if fdp else "", tdir))
    #
    # Find out if there are any more in the SDF file.
    #
    nligs = None        
    with open("%s/log"%(tdir), 'r') as sdffile:
        for line in sdffile:
            m = re.search('nligs += (\d+)', line)
            if m:
                nligs = int(m.group(1))
                break
    nligs = nligs if nligs else 0
    #
    # Found total lig count; do any left over.
    #
    for ligndx in range(1,nligs):
        print("#---------------------------------------------------------------")
        print("# SDF \"%s\" Ligand %d of %d"%(fn,ligndx+1,nligs))
        print("#---------------------------------------------------------------")
        os.system("./mlvoxelizer --out %s --sdf %s --protonate --ligndx %d %s | tee %s/log"%
                  (tdir, "%s/%s"%(path,fn), ligndx, "--fdp" if fdp else "", tdir))
    #
    # Scan the outputs to see what was produced
    #
    lig_files = find_files(tdir,".lig")
    #
    # Copy the outputs to the desired output dir
    #
    lig_out = []
    for lig_file in lig_files:
        out = outpath+"/lig/"+os.path.basename(lig_file)
        shutil.copyfile(lig_file, out)
        lig_out.append(out)
    #
    # Cleanup the temp files
    #
    shutil.rmtree(tdir)
    #
    # Return a list of the found lig files.
    #
    return lig_out


############################################################


def main():
    print("main(): Stub!")


if __name__ == "__main__":
    main()


############################################################
