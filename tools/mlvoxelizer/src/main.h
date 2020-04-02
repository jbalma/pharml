#ifndef MAIN_H
#define MAIN_H

#include "types.h"
#include "atom.h"
#include "random.h"
#include "vector.h"
#include "io_bitmap.h"

#define VERSION "v0.0.24"

// This needs to match the one in gui.h ...
typedef struct str_state_t {
  u64b_t      time;         // Current time
  rnd_t       random;       // Random number generator state
  u32b_t      seed;         // Initial random seed
  int         gui;          // Flag for using the GUI.
  int         fdp;          // Force-directed placement flag.
  int         pvox;         // Protein input is vox file (flag).
  int         protonate;    // Flag for adding assumed hydrogen.
  int         ligndx;       // Ligand index for multi-lig files.
  int         rotate;       // Rotate the protein (random) flag.
  int         readonly;     // Readonly flag.

  char       *pfn;          // Protein file name.
  char       *sdf;          // Ligand SDF library file name.
  char       *lfn;          // Ligand file name.
  char       *smiles;       // Ligand smiles string.
  char       *dout;         // Output data directory.

  atom_t     *atoms;        // Array of atoms (protein).
  int         natoms;       // number of atoms (protein).
  vector3_t   amin;         // Min coord of atoms.
  vector3_t   amax;         // Max coord of atoms.

  float       win_w;        // Stencil window width
  float       win_h;        // Stencil window height
  float       win_d;        // Stencil window depth
  float       stride;       // Stencil window stride
  float       valid;        // Stencil window min non-empty voxels.
  int         lonly;        // Ligand only mode.

  int         chnls;        // Number of atom channels (types).
  char       *chnlt;        // Atom type of each channel.
  float       res;          // Resolution in voxels per Angstrom.
  float      *voxels;       // Voxelized version.
  int        *vatoms;       // Index of each atom into voxel array.
  int         vx;           // Width of voxels.
  int         vy;           // Height of voxels.
  int         vz;           // Depth of voxels.
  vector3_t   vmin;         // Min coord of atoms.
  vector3_t   vmax;         // Max coord of atoms.
  int         nsb;          // Number of stencil boxes.
  float       sigma;        // Gaussian filter for voxels.
  float       fftwplanrigor;// Time limit in seconds for FFTW plan.
                            // If>0 then FFTW_MEASURE, else FFTW_ESTIMATE

  atom_t     *ligand;       // Array of atoms (ligand).
  int         nligand;      // number of atoms (ligand).
  int       (*bonds)[4];    // Array of bonds in the ligand.
  int        nbonds;        // Number of bonds in bond array.
  vector3_t   lmin;         // Min coord of ligand.
  vector3_t   lmax;         // Max coord of ligand.

  vector4_t  *inference;    // Inference results / bind predictions.
  int        ninference;    // Number of predictions in inference array.
} state_t;

#endif
