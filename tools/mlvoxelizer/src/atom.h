#ifndef ATOM_H
#define ATOM_H

#include "vector.h"


typedef struct str_atom_t {
  vector3_t  pos;
  vector3_t  force;
  int        id;
  int        bonds;
  int        charge;
  char       type;
  int        flags;
} atom_t;


extern int   atom_bonds   (u8b_t type, int charge);
extern int   atom_color   (u8b_t type, vector3_t *color);
extern float atom_radius  (u8b_t type);
extern int   atom_charge  (u8b_t type);
extern int   atom_channel (u8b_t type);
extern char *atom_channels();


#endif
