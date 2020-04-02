#ifndef CHEMIO_H
#define CHEMIO_H

#include "main.h"

extern void read_pdb      (state_t *state, char *pdb);
extern int  read_sdf      (state_t *state, char *sdf);
extern void read_voxels   (state_t *state, char *vox);
extern void read_ligand   (state_t *state, char *lig);
extern void read_smiles   (state_t *state, char *smiles);
extern void read_inference(state_t *state, char *inf);

extern void write_ligand       (state_t *state, char *fn);
extern void write_voxels       (state_t *state, int x, int y, int z, int w, int h, int d, char *fn);
extern void write_stencil_boxes(state_t *state, int w, int h, int d, int s, int t, int l);
extern void write_protein_graph(state_t *state, double rad);

#endif
