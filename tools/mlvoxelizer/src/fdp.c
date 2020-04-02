#include <math.h>
#include <string.h>
#include <stdio.h>

#include "types.h"
#include "util.h"
#include "vector.h"
#include "atom.h"
#include "random.h"
#include "fdp.h"
#include "main.h"

////////////////////////////////////////////////////////////
// Force-directed placement code.
////////////////////////////////////////////////////////////

#define EPSILON       (1.0e-7)
#define MAX_STEPS     1000000
#define SCALE_FORCE   (2.0)
#define GRAVITY_FORCE (-0.05)
#define SPRING_FORCE  (0.1)

int fdp_ligand(state_t *state)
{
  vector3_t v;
  double    f,s;
  int       i,j,k=0;

  // Is force-directed placement enabled?
  if( state->fdp == 1 ) {
    // Give some random nudges to the positions to avoid some pathological cases.
    for(j=0; j<state->nligand; j++) {
      state->ligand[j].pos.s.x += random_U(&(state->random),EPSILON);
      state->ligand[j].pos.s.y += random_U(&(state->random),EPSILON);
      state->ligand[j].pos.s.z += random_U(&(state->random),EPSILON);
      state->ligand[j].pos.s.x -= random_U(&(state->random),EPSILON);
      state->ligand[j].pos.s.y -= random_U(&(state->random),EPSILON);
      state->ligand[j].pos.s.z -= random_U(&(state->random),EPSILON);
    }
    for(k=0; k<MAX_STEPS; k++) {
      // Clear forces.
      for(i=0; i<state->nligand; i++) {
	memset(&(state->ligand[i].force),0,sizeof(vector3_t));
      }
      // Compute force of anti-gravity on atoms.
//#pragma omp parallel for private(i,j,v,f)
      for(i=0; i<state->nligand; i++) {
	for(j=0; j<state->nligand; j++) {
	  if( i != j ) {
	    vector3_sub_vector(&(state->ligand[j].pos),
			       &(state->ligand[i].pos),
			       &v);
	    f = vector3_length(&v);
	    f = GRAVITY_FORCE / (f*f);
	    vector3_normalize(&v,&v);
	    vector3_mult_scalar(&v,&v,f);
	    vector3_add_vector(&v,
			       &(state->ligand[i].force),
			       &(state->ligand[i].force));
	  }
	}
      }
      // Compute force of bonds on atoms.
      for(i=0; i<state->nbonds; i++) {
	vector3_sub_vector(&(state->ligand[state->bonds[i][1]].pos),
			   &(state->ligand[state->bonds[i][0]].pos),
			   &v);
	f = vector3_length(&v);
	f = SPRING_FORCE * f;
	vector3_normalize(&v,&v);
	vector3_mult_scalar(&v,&v,f);
	vector3_add_vector(&v,
			   &(state->ligand[state->bonds[i][0]].force),
			   &(state->ligand[state->bonds[i][0]].force));
	vector3_mult_scalar(&v,&v,-1.0);
	vector3_add_vector(&v,
			   &(state->ligand[state->bonds[i][1]].force),
			   &(state->ligand[state->bonds[i][1]].force));
      }
      // Test for convergence.
      s = 0.0;
      for(i=0; i<state->nligand; i++) {
	f = vector3_length(&(state->ligand[i].force));
	s += f;
      }
      if( s < EPSILON ) {
	// Converged.
	state->fdp = 2;
	break;
      }
      // Update atom positions.
      for(i=0; i<state->nligand; i++) {
	vector3_mult_scalar(&(state->ligand[i].force),&(state->ligand[i].force),SCALE_FORCE);
	vector3_add_vector(&(state->ligand[i].force),
			   &(state->ligand[i].pos),
			   &(state->ligand[i].pos));
      }
    }
    if( k >= MAX_STEPS ) {
      Warn("fdp_ligand(): Did not converge for %d timesteps.\n",k);
      return -k;
    }
  }

  // Return convergence time in steps.
  return k;
}
