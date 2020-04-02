#ifndef RANDOM_H
#define RANDOM_H

#include "types.h"

#define  TWO_32 (4294967296.0)


// State type for the random number generator
typedef struct {
  u32b_t  rtab[55];
  u32b_t  rndx;
} rnd_t;

// for generating random number i with probability p[i]
typedef struct { 
    double *p;
    u32b_t *a;
    u32b_t  n;
    double  m1;
    double  m2;
} dist_t;


////////////////////////////////////////////////////////////
// Interface (for everyone)
////////////////////////////////////////////////////////////
#define random_rndm(r)  ((++(r)->rndx>54)?(r)->rtab[(r)->rndx=random_nrndm(r)]:(r)->rtab[(r)->rndx]) // random 32-bit generator
#define random_U01(r)   (random_rndm(r)/TWO_32)                                                      // random in interval [0,1)
#define random_U(r,x)   (random_U01(r)*(x))                                                          // random in interval [0,x)
#define random_rnd(r,n) ((u32b_t)random_U(r,n))                                                      // random from set {0..n-1}


////////////////////////////////////////////////////////////
// Interface (not for random.c)
////////////////////////////////////////////////////////////
#ifndef RANDOM_C
extern u32b_t  random_nrndm    (rnd_t *r);                     // Not used directly; call the macros instead
extern void    random_initrand (rnd_t *r, u32b_t j);           // Initialize random number generator with seed j

extern u32b_t  random_drand    (rnd_t *r, dist_t *d);          // Return element from {0..d->n-1} according to d
extern u32b_t  random_Poisson  (rnd_t *r, double mu);

extern dist_t* random_allocdist(u32b_t n);                     // allocate a distribution over {0..n-1}
extern void    random_initdist (dist_t *d, double s );         // Initialize the distribution d
extern void    random_freedist (dist_t *d);                    // frees dist
#endif


#endif
