# mlvoxelizer

```
+---+ +---+ +---+
| M | | L | | Vo|xelizer v0.0.24 -- (C) Cray Inc. 2018
+---+ +---+ +---+
```

----------------

The mlvoxelizer tool helps manipulate data in the pharma / drug
discovery space (i.e., protein-ligand binding) for machine learning
approaches.

Features support for voxelization of PDB protein files as well as
transformation of SDF ligand files.  The tool also includes an OpenGL
GUI for viewing the data in 3D.

```
usage:
	mlvoxelizer <in> [opt_1] [...] [opt_n]

	<in> is from:
	--pdb    <path to input PDB file>
	--vox    <path to input voxel file>
	--lig    <path to input ligand file>
	--sdf    <path to input SDF ligand file>
	--smiles <ligand SMILES string>

	[opt_*] is from:
	--out    <output directory>          // ./data/
	--res    <voxels per angstrom>       // 3.0
	--win    <win_w>x<win_h>x<win_d>     // 20.0x20.0x20.0
	--stride <win stride in angstroms>   // 3.0
	--gauss  <sigma>                     // 0.001000
	--fftwplanrigor <seconds>            // 0.000000
	--valid  <all|ligand>                // ligand
	--ligndx <ligand index>              // 0
	--inf    <path to inference file>    // n/a
	--gui                                // no
	--gui-basic                          // no
	--fdp                                // no
	--protonate                          // no
	--readonly                           // no
	--rotate                             // no
```

![screen01](https://github.com/bigdatumrich/ML_Protein/raw/master/mlvoxelizer/screen01.png "Screen01")

The above is a screenshot of the "mlvoxelizer" tool showing a voxelized protein.

----------------
