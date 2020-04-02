#include "types.h"
#include "util.h"
#include "vector.h"
#include "atom.h"

int atom_bonds(u8b_t type, int charge)
{
  switch( type ) {
  case 'L': // Cl
  case 'F':
  case 'R': // Br
  case 'H':
    return 1+charge;
  case 'C':
    return 4+charge;
  case 'Z': // Zn
  case 'P':
  case 'N':
    return 3+charge;
  case 'G': // Mg
  case 'S':
  case 'O':
  case 'E': // Se
    return 2+charge;
  }

  return -1;
}

int atom_color(u8b_t type, vector3_t *color)
{
  switch( type ) {
  case 'Z': // Zn
    color->a[0] = ((float)0x7d)/255.0f;
    color->a[1] = ((float)0x80)/255.0f;
    color->a[2] = ((float)0xb0)/255.0f;
    return 0;
  case 'G': // Mg
    color->a[0] = 0.0f;
    color->a[1] = 77.0f/255.0f;
    color->a[2] = 0.0f;
    return 0;
  case 'S':
    color->a[0] = 1.0f;
    color->a[1] = 1.0f;
    color->a[2] = 0.0f;
    return 0;
  case 'O':
    color->a[0] = 1.0f;
    color->a[1] = 0.0f;
    color->a[2] = 0.0f;
    return 0;
  case 'E': // Se
    color->a[0] = 1.0f;
    color->a[1] = 80/255.0f;
    color->a[2] = 0.1f;
    return 0;
  case 'N':
    color->a[0] = 0.0f;
    color->a[1] = 0.0f;
    color->a[2] = 1.0f;
    return 0;
  case 'C':
    color->a[0] = 0.33f;
    color->a[1] = 0.33f;
    color->a[2] = 0.33f;
    return 0;
  case 'H':
    color->a[0] = 0.8f;
    color->a[1] = 0.8f;
    color->a[2] = 0.8f;
    return 0;
  case 'P':
    color->a[0] = 1.0f;
    color->a[1] = 69/255.0f;
    color->a[2] = 0.0f;
    return 0;
  case 'F':
    color->a[0] = 0.0f;
    color->a[1] = 1.0f;
    color->a[2] = 0.0f;
    return 0;
  case 'L': // Cl
    color->a[0] = 0.0f;
    color->a[1] = 1.0f;
    color->a[2] = 0.0f;
    return 0;
  case 'R': // Br
    color->a[0] = 99/255.0f;
    color->a[1] = 22/255.0f;
    color->a[2] = 0.0f;
    return 0;
  }

  // Default color is white.
  color->a[0] = 1.0f;
  color->a[1] = 1.0f;
  color->a[2] = 1.0f;
  return -1;
}

float atom_radius(u8b_t type)
{
  switch(type) {
  case 'Z': // Zn
    return 1.42f;
  case 'G': // Mg
    return 1.45f;
  case 'S':
    return 0.88f;
  case 'O':
    return 0.48f;
  case 'E': // Se
    return 1.03f;
  case 'N':
    return 0.56f;
  case 'C':
    return 0.67f;
  case 'H':
    return 0.53f;
  case 'P':
    return 0.98f;
  case 'F':
    return 0.42f;
  case 'L': // Cl
    return 0.79f;
  case 'R': // Br
    return 0.94f;
  }
  
  return -1.0f;
}

int atom_charge(u8b_t type)
{
  switch(type) {
  case 'H':
    return 1;
  case 'C':
    return 6;
  case 'N':
    return 7;
  case 'O':
    return 8;
  case 'F':
    return 9;
  case 'G': // Mg
    return 12;
  case 'P':
    return 15;
  case 'S':
    return 16;
  case 'L': // Cl
    return 17;
  case 'Z': // Zn
    return 30;
  case 'E': // Se
    return 34;
  case 'R': // Br
    return 35;
  }
  
  return -1;
}

int atom_channel(u8b_t type)
{
  // "HCNOSE"
  switch(type) {
  case 'H':
    return 0;
  case 'C':
    return 1;
  case 'N':
    return 2;
  case 'O':
    return 3;
  case 'S':
    return 4;
  case 'P':
    return 5;
  }
  
  return -1;
}

char *atom_channels()
{
  return "HCNOSP";
}
