#include <stdlib.h>
#include <math.h>

#include "types.h"
#define RANDOM_C
#include "random.h"

// Frees dist
void random_freedist(dist_t *d)
{
  free(d->a);
  free(d->p);
  free(d);
}

// Fills rtab buffer; not used directly, call macros instead
u32b_t random_nrndm(rnd_t *r)
{
  u8b_t i;  

  for (i =  0; i < 24; i++) r->rtab[i] -= r->rtab[i+31];
  for (i = 24; i < 55; i++) r->rtab[i] -= r->rtab[i-24];
  return 0;
}

// Initialize the random number generator with seed j
void random_initrand(rnd_t *r, u32b_t j)
{
  u32b_t k,i,h;

  for (r->rtab[54] = j |= (k = i = 1); i < 55; i++)
    h = (21*i)%55, r->rtab[--h] = k, k = j - k, j = r->rtab[h];

  while (i--) random_nrndm(r);
  r->rndx = 0;
}

// Allocate a distribution over {0..n-1}
dist_t* random_allocdist(u32b_t n)
{
  dist_t *d;

  d = (dist_t*) malloc(sizeof(dist_t));
  d->n = n;
  d->a = (u32b_t*)malloc(d->n * sizeof(u32b_t));
  d->p = (double*)malloc(d->n * sizeof(double));
  return d;
}

// Initialize the distribution d
//
// Note: d->p must have d->n elements which sum to s on entry to initdist.
// The elements of d->p and d->a are overwritten by the initialization process.
#define GETSMALL { while (p[j] >= q) if ((++j) == stop) return; t = j++; }
#define GETLARGE { while (p[k] <  q) if ((++k) == stop) goto cleanup;    }
void random_initdist(dist_t *d, double s) 
{
  u32b_t j,k,t,stop,*a;  double q,*p;

  stop = d->n, q = s/stop, j = k = 0;

  d->m1 = stop/TWO_32;
  d->m2 = s/(stop * TWO_32);

  a = d->a;
  p = d->p;

  GETSMALL; GETLARGE;

  while(1) {
    a[t]  = k;
    p[k] += p[t] - q;
    
    if (p[k] >= q) { 
      if (j == stop) return;
      GETSMALL;
      continue;
    }

    t = k++;
    if (k == stop) break;
    if (j < k)     GETSMALL;
    GETLARGE;
  }

 cleanup:
  for(a[t]=t; j < stop; j++) 
    a[j] = j;
}
#undef GETSMALL
#undef GETLARGE

// Return element from {0..d->n-1} according to d
u32b_t random_drand(rnd_t *r, dist_t *d)
{
  u32b_t r1 = random_rndm(r),j;

  if ((random_rndm(r)+(r1&0xffff)/65536.0)*(d->m2) < d->p[j=r1*(d->m1)]) return j;

  return d->a[j];
}

//
//  Returns choice from poisson distribution with mean mu
//
u32b_t random_Poisson(rnd_t *r, double mu)
{
  double p = exp( -mu ), q = 1;
  s32b_t n=-1;

  do q *= random_U01(r), n++;
  while( (q > p) || (q == p) );

  return n;
}
