#ifndef VOXELIZER_H
#define VOXELIZER_H

#include "main.h"

#define VOXELIZER_GLOBAL_SIZE (128) // Angstroms.

extern void voxelize(state_t *state, float res);
extern void gaussian(double *v, int w, int h, int d, double s, double fftwplanrigor);
extern void rotation(state_t *state);

extern void clean_protein(state_t *state);

extern char* fftw_version_string();

#endif
