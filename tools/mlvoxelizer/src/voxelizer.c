////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <limits.h>

#include "types.h"
#include "util.h"
#include "vector.h"
#include "matrix.h"
#include "rotation.h"
#include "voxelizer.h"

#define TWOPI 6.28318530718


////////////////////////////////////////////////////////////


static void dfs_find_aromaticity(state_t *state, int l, int depth, int type)
{
  int next_type = type;
  int neighbor;
  int rel_depth;
  int i,j,k;

  // Have we exhausted the search depth?
  if( depth >= 8 ) {
    // No aromaticity.
    return;
  }
  
  // Have we hit a marked atom (indicating a cycle with alternating 1/2 bond types)?
  if( state->ligand[l].flags ) {
    // We found a loop!
    rel_depth = depth - (state->ligand[l].flags-1);
    if( rel_depth > 3 ) {
      // Cool! Found a possible aromatic ring!
      //printf("        Found (possible) aromatic ring of size %d! (starting and ending at ligand atom %d)\n",rel_depth,l);
      // Update all atoms that are marked as part of the currently found loop.
      for(i=0; i<state->nligand; i++) {
	// Find all pairs of ligand atoms such that both atoms are part of the aromatic ring.
	// Remember that we want to update bonds, not atoms.
	for(j=i+1; j<state->nligand; j++) {
	  // First be sure the pair of atoms are part of the current ring.
	  if( (state->ligand[i].flags > depth-rel_depth) &&
	      (state->ligand[j].flags > depth-rel_depth)    ) {
	    // Now make sure there is a bond between the pair.
	    for(k=0; k<state->nbonds; k++) {
	      if( ((state->bonds[k][0] == i) && (state->bonds[k][1] == j)) ||
		  ((state->bonds[k][0] == j) && (state->bonds[k][1] == i))    ) {
		// Found a bond between i and j in the ligand atom array.
		//printf("        Update bond (%d) between (%d,%d) to aromatic.\n",k,i,j);
		state->bonds[k][3] = 1;
	      }
	    }
	  }
	}
      }
    }
    return;
  }

  // What bond type do we expect next?
  if( next_type != -1 ) {
    if( next_type == 1 ) {
      next_type = 2;
    } else if( next_type == 2 ) {
      next_type = 1;
    } else {
      fprintf(stderr,"dfs_find_aromaticity(): Unexpected current bond type.\n");
      exit(1);
    }
  }
  
  // Find all edges from the current atom to other atoms.
  for(i=0; i<state->nbonds; i++) {
    if( (state->bonds[i][0] == l) || (state->bonds[i][1] == l) ) {
      // OK, we have a neighbor, what is that atom's index?
      if( state->bonds[i][0] == l ) {
	neighbor = state->bonds[i][1];
      } else {
	neighbor = state->bonds[i][0];	
      }
      // Found an edge involving atom a, recurse to the neighbor?
      if( (next_type == -1) || (next_type == state->bonds[i][2]) ) {
	// !!avose: Double check the current / neighboring atom?
	// Mark as part of search with the current search depth+1.
	state->ligand[l].flags = depth+1;
	// Recurse to the neighbor.
	dfs_find_aromaticity(state, neighbor, depth+1, state->bonds[i][2]);
	// Unmark the current node as part of the current search.
	state->ligand[l].flags = 0;
      }
    }
  }

}


static void find_aromaticity(state_t *state)
{
  int i,first=1;

  // Consider each atom as the start of a depth-first search.
  for(i=0; i<state->nligand; i++) {
    dfs_find_aromaticity(state, i, 0, -1);
  }

  // Update bonds
  for(i=0; i<state->nbonds; i++) {
    if( state->bonds[i][3] ) {
      if( first ) {
	// Only print the text header if (and once) there is an update.
	first = 0;
	printf("        Update bonds to aromatic: ");
      }
      printf("(%d,(%d,%d)) ",i,state->bonds[i][0],state->bonds[i][1]);
      // Set the bond type to 4 (aromatic).
      state->bonds[i][2] = 4;
    }
  }
  // If we printed anything, be sure to add a newline.
  if( !first ) {
    printf("\n");
  }
  
}


////////////////////////////////////////////////////////////


void clean_protein(state_t *state)
{
  float c;
  int   i,j,k,a,b;

  // Do some post-read processing (e.g., center, normalize). 
  for(i=0; i<state->natoms; i++) {
    // Initialize min/max from first atom.
    if( i == 0 ) {
      for(j=0; j<3; j++) {
	state->amin.a[j] = state->atoms[i].pos.a[j];
	state->amax.a[j] = state->atoms[i].pos.a[j];
      }
    }
    // Do x, y, and z
    for(j=0; j<3; j++) {
      // Min
      if( state->amin.a[j] > state->atoms[i].pos.a[j] ) {
	state->amin.a[j] = state->atoms[i].pos.a[j];
      }
      // Max
      if( state->amax.a[j] < state->atoms[i].pos.a[j] ) {
	state->amax.a[j] = state->atoms[i].pos.a[j];
      }
    }
  }
  // And let's do a min/max for ligand as well.
  for(i=0; i<state->nligand; i++) {
    // Initialize min/max from first atom.
    if( i == 0 ) {
      for(j=0; j<3; j++) {
	state->lmin.a[j] = state->ligand[i].pos.a[j];
	state->lmax.a[j] = state->ligand[i].pos.a[j];
      }
    }
    // Do x, y, and z
    for(j=0; j<3; j++) {
      // Min
      if( state->lmin.a[j] > state->ligand[i].pos.a[j] ) {
	state->lmin.a[j] = state->ligand[i].pos.a[j];
      }
      // Max
      if( state->lmax.a[j] < state->ligand[i].pos.a[j] ) {
	state->lmax.a[j] = state->ligand[i].pos.a[j];
      }
    }
  }
  // Center protien and ligand.
  for(i=0; i<state->natoms; i++) {
    for(j=0; j<3; j++) {
      state->atoms[i].pos.a[j] -= state->amin.a[j] + (state->amax.a[j] - state->amin.a[j])/2.0;
    }
  }
  for(i=0; i<state->nligand; i++) {
    if( state->natoms ) {
      for(j=0; j<3; j++) {
	state->ligand[i].pos.a[j] -= state->amin.a[j] + (state->amax.a[j] - state->amin.a[j])/2.0;
      }
    } else {
      for(j=0; j<3; j++) {
	state->ligand[i].pos.a[j] -= state->lmin.a[j] + (state->lmax.a[j] - state->lmin.a[j])/2.0;
      }
    }
  }
  // Get the new min/max after centering.
  // (There's a faster way than this..)
  // (Remember we just centered it..)
  for(i=0; i<state->natoms; i++) {
    // Initialize min/max from first atom.
    if( i == 0 ) {
      for(j=0; j<3; j++) {
	state->amin.a[j] = state->atoms[i].pos.a[j];
	state->amax.a[j] = state->atoms[i].pos.a[j];
      }
    }
    // Do x, y, and z
    for(j=0; j<3; j++) {
      // Min
      if( state->amin.a[j] > state->atoms[i].pos.a[j] ) {
	state->amin.a[j] = state->atoms[i].pos.a[j];
      }
      // Max
      if( state->amax.a[j] < state->atoms[i].pos.a[j] ) {
	state->amax.a[j] = state->atoms[i].pos.a[j];
      }
    }
  }

  // Check all the bonds real quick.
  int explicit_bonds = 0;
  for(i=0; i<state->nbonds; i++) {
    if( (state->bonds[i][2] < 1) || (state->bonds[i][2] > 3) ) {
      if( state->bonds[i][2] == 4 ) {
	// If we see an aromatic bond, we know the file must be explicit.
	explicit_bonds = 1;
      } else {
	Error("  clean_protein(): Ligand bond with unexpected type: %d.\n",state->bonds[i][2]);
      }
    }
    for(j=0; j<state->nbonds; j++) {
      if( i != j ) {
	if( ((state->bonds[i][0] == state->bonds[j][0]) && (state->bonds[i][1] == state->bonds[j][1])) ||
	    ((state->bonds[i][0] == state->bonds[j][1]) && (state->bonds[i][1] == state->bonds[j][0]))    ) {
	  Error("  clean_protein(): Ligand bond not coalesced.\n");
	}
      }
    }
  }

  if( !explicit_bonds ) {
    // If we don't have "explicit" bond types, that means that some 1-2-1 / 2-1-2 sequences
    // may actually be aromatic, but the file didn't explicitly mark the bonds as such.
    // Here, we will _attempt_ to convert some of these sequences to bond type 4.
    printf("    Converting assumed to explicit bonds:\n");
    find_aromaticity(state);
  }

  // Recompute the atoms' bond counts from the bond list.
  for(a=0; a < state->nligand; a++) {
    // Init bond count involving atom a
    c = 0.0;
    for(b=0; b < state->nligand; b++) {
      // Ignore self
      if( a == b ) {
	continue;
      }
      // Consider all bonds in list and accumulate bond count
      for(j=0; j<state->nbonds; j++) {
	if( ((state->bonds[j][0] == a) && (state->bonds[j][1] == b)) ||
	    ((state->bonds[j][0] == b) && (state->bonds[j][1] == a))    ) {
	  // Add bond cound with aromatic bonds in mind
	  switch( state->bonds[j][2] ) {
	  case 4:
	    c += 1.5;
	    break;
	  default:
	    c += state->bonds[j][2];
	    break;
	  }
	}
      }
    }
    // Record atom a's accumulated bond count
    if( c == 4.5 ) {
      // !!avose: My aromaticity handling is not good here.
      // !!avose: An atom can have 3 aromatic bonds.
      c = 4;
    }
    if( ((float)((int)(c))) != c ) {
      Error("  clean_protein(): Non-integer bond count %f on '%c' id %d (unbalanced aromaticity?).\n",
	    c, state->ligand[a].type, state->ligand[a].id);
    }
    state->ligand[a].bonds = c;
  }
  
  // Now, a sanity check on ligand's atoms' bound counts.
  for(i=0; i<state->nligand; i++) {
    if( state->protonate ) {
      // Remove free-floating water / other common free-floating junk.
      if( ((state->ligand[i].type == 'O')||(state->ligand[i].type == 'Z')||(state->ligand[i].type == 'G')) &&
	  (state->ligand[i].bonds == 0) ) {
	// Remove the free atom.
	memcpy(&(state->ligand[i]), &(state->ligand[state->nligand-1]), sizeof(atom_t));
	// Update bonds.
	for(j=0; j<state->nbonds; j++) {
	  if( state->bonds[j][0] == state->nligand-1 ) {
	    state->bonds[j][0] = i;
	  }
	  if( state->bonds[j][1] == state->nligand-1 ) {
	    state->bonds[j][1] = i;
	  }
	}
	// Update counts to finish removal.
	state->nligand--;
	i--;
	continue;
      }
      // Add hydrogens if we are in protonating mode.
      k = atom_bonds(state->ligand[i].type, state->ligand[i].charge) - state->ligand[i].bonds;
      for(j=0; (k > 0) && (j < k); j++) {
	// Add hydrogen atom.
	state->nligand++;
	if( !(state->ligand=realloc(state->ligand,state->nligand*sizeof(atom_t))) ) {
	  Error("  clean_protein(): Grow of ligand array failed.\n");
	}
	memset(&(state->ligand[state->nligand-1]),0,sizeof(atom_t));
	state->ligand[state->nligand-1].type = 'H';
	vector3_copy(&(state->ligand[i].pos),&(state->ligand[state->nligand-1].pos));
	// Add the bond to the new H.
	state->nbonds++;
	if( !(state->bonds=realloc(state->bonds,state->nbonds*4*sizeof(int))) ) {
	  Error("  clean_protein(): Grow of bond array failed.\n");
	}
	state->bonds[state->nbonds-1][0] = i;                // Atom_a.
	state->bonds[state->nbonds-1][1] = state->nligand-1; // Atom_b.
	state->bonds[state->nbonds-1][2] = 1;                // Bond type.
	state->bonds[state->nbonds-1][3] = 0;                // Aromatic flag?
	state->ligand[i].bonds++;
	state->ligand[state->nligand-1].bonds++;
      }
    }
    // Now that we are done with optional additions, do the bond counts make sense?
    if( state->ligand[i].bonds != atom_bonds(state->ligand[i].type, state->ligand[i].charge) ) {
      Warn("  clean_protein(): Unexpected ligand atom bond count: '%c' (id %d) %d != %d.\n",
	   state->ligand[i].type, state->ligand[i].id,
	   state->ligand[i].bonds, atom_bonds(state->ligand[i].type, state->ligand[i].charge));
    }
  }  

  // And let's do a min/max for ligand as well.
  for(i=0; i<state->nligand; i++) {
    // Initialize min/max from first atom.
    if( i == 0 ) {
      for(j=0; j<3; j++) {
	state->lmin.a[j] = state->ligand[i].pos.a[j];
	state->lmax.a[j] = state->ligand[i].pos.a[j];
      }
    }
    // Do x, y, and z
    for(j=0; j<3; j++) {
      // Min
      if( state->lmin.a[j] > state->ligand[i].pos.a[j] ) {
	state->lmin.a[j] = state->ligand[i].pos.a[j];
      }
      // Max
      if( state->lmax.a[j] < state->ligand[i].pos.a[j] ) {
	state->lmax.a[j] = state->ligand[i].pos.a[j];
      }
    }
  }

}


////////////////////////////////////////////////////////////


void rotation(state_t *state)
{
  double    angle;
  matrix4_t rot;
  vector3_t axis;
  vector4_t pos;
  int       i;

  // Get a random axis and rotation angle.
  angle    = random_U(&(state->random), 360.0);
  axis.s.x = random_U01(&(state->random)) - 0.5;
  axis.s.y = random_U01(&(state->random)) - 0.5;
  axis.s.z = random_U01(&(state->random)) - 0.5;
  vector3_normalize(&axis,&axis);
  printf("  rotation   = %.1lf deg about (%.2lf,%.2lf,%.2lf)\n",
	 angle, axis.s.x, axis.s.y, axis.s.z);
  
  // Get a rotation matrix.
  matrix4_rotation(angle, &axis, &rot);

  // Rotate all the atoms in the protein and ligand.
  for(i=0; i<state->natoms; i++) {
    // Get vector4_t for position.
    pos.s.x = state->atoms[i].pos.s.x;
    pos.s.y = state->atoms[i].pos.s.y;
    pos.s.z = state->atoms[i].pos.s.z;
    pos.s.w = 1.0;
    // Rotate.
    matrix4_mult_vector(&rot, &pos, &pos);
    // Put the vector4_t back into the vec3 pos.
    state->atoms[i].pos.s.x = pos.s.x;
    state->atoms[i].pos.s.y = pos.s.y;
    state->atoms[i].pos.s.z = pos.s.z;
  }  
  for(i=0; i<state->nligand; i++) {
    // Get vector4_t for position.
    pos.s.x = state->ligand[i].pos.s.x;
    pos.s.y = state->ligand[i].pos.s.y;
    pos.s.z = state->ligand[i].pos.s.z;
    pos.s.w = 1.0;
    // Rotate.
    matrix4_mult_vector(&rot, &pos, &pos);
    // Put the vector4_t back into the vec3 pos.
    state->ligand[i].pos.s.x = pos.s.x;
    state->ligand[i].pos.s.y = pos.s.y;
    state->ligand[i].pos.s.z = pos.s.z;
  }  

  // This is called again to re-calc some bounds, etc.
  clean_protein(state);
}


////////////////////////////////////////////////////////////


char* fftw_version_string()
{
  static char  buf[1024];
  char        *str,c[2]=" ";

  str = fftw_export_wisdom_to_string();
  if( !str ) {
    return "unknown";
  }
  if( !strstr(str,c) ) {
    return "unknown";
  }
  strstr(str,c)[0] = '\0';
  sprintf(buf,"%s",str+1);
  free(str);

  return buf;
}


void gaussian(double *v, int w, int h, int d, double s, double fftwplanrigor)
{
  vector3_t     p,q,t;
  static double *vsd = NULL;
  double        dist;
  int           nf  = w*h*(2*(d/2+1));
  int           ns  = w*h*d;
  int           i,j,x,y,z,wx,wy,wz;

  static double        gs = -1.0;
  static int           gw,gh,gd;
  static fftw_plan     pv1,pv2,pg1;
  static double       *gsd = NULL;
  static fftw_complex *gfd = NULL;
  static fftw_complex *vfd = NULL; 
  static unsigned      fftw_flags;
  //double t0, t1; //timers

  // 3D Gaussian kernel amplitude
  double Agsd = 1.0 / ( s*s*s * pow(TWOPI,1.5) );

  // I just crammed a cleanup thing in here too.
  if( v == NULL ) {
    if( gsd ) { fftw_free(gsd); gsd = NULL; }
    if( gfd ) { fftw_free(gfd); gfd = NULL; }
    if( vsd ) { fftw_free(vsd); vsd = NULL; }
    if( vfd ) { fftw_free(vfd); vfd = NULL; }
    if( gs != -1.0 ) {
      gs = -1.0;
      fftw_destroy_plan(pv1);
      fftw_destroy_plan(pv2);
      fftw_destroy_plan(pg1);
      fftw_cleanup_threads();
    }
    return;
  }

  // Interpret FFTW plan rigor
  // i.e., if fftwplanrigor<=0 then plan with FFTW_ESTIMATE, else
  // plan with FFTW_MEASURE with time limit of fftwplanrigor seconds.
  if ( fftwplanrigor <= 0.0 ) {
    fftw_flags = FFTW_ESTIMATE;
  }
  else {
    fftw_flags = FFTW_MEASURE;
  }

  // Only rebuild the Gaussian and plans if something changed.
  if( !((s == gs) && (w == gw) && (h = gh) && (d = gd)) ) {
    // Get FFTW ready for threads.
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    fftw_set_timelimit(fftwplanrigor);
    // Save parameters.
    gs = s;
    gw = w;
    gh = h;
    gd = d;
    // Space for gaussian.
    if( gsd ) { fftw_free(gsd); }
    if( gfd ) { fftw_free(gfd); }
    if( !(gsd=fftw_malloc(ns*sizeof(double))) ) {
      Error("gaussian_filer(): Failed to allocate array of doubles: gaussian (%d).\n",ns);
    }
    memset(gsd,0,ns*sizeof(double));
    if( !(gfd=fftw_malloc(nf*sizeof(fftw_complex))) ) {
      Error("gaussian_filer(): Failed to allocate array of fftw_complex: gaussian (%d).\n",nf);
    }
    memset(gfd,0,nf*sizeof(fftw_complex));

    // Plan the FFT for Gaussian to frequency domain.
    //t0 = omp_get_wtime();
    pg1 = fftw_plan_dft_r2c_3d(d, h, w, gsd, gfd, fftw_flags);
    //t1 = omp_get_wtime();
    //printf("R2C plan for gaussian took %10.4e sec.\n", t1-t0);

    // Fill in Gaussian in spatial domain.
    //t0 = omp_get_wtime();
#if 0
    // Identity kernel.
    gsd[ 0 ] = 1.0; 
#else
    // Gaussian kernel.
    p.s.x = 0.0;
    p.s.y = 0.0;
    p.s.z = 0.0;
#pragma omp parallel for private(z,y,x,j,dist,wx,wy,wz,q,t)
    for(z=0; z<d; z++) {
      for(y=0; y<h; y++) {
	for(x=0; x<w; x++) {
	  j = x + y*w + z*w*h;
	  dist = DBL_MAX;
	  for(wx=0; wx<2; wx++) {
	    for(wy=0; wy<2; wy++) {
	      for(wz=0; wz<2; wz++) {
		q.s.x = ((double)x)/(w-1) - wx;
		q.s.y = ((double)y)/(h-1) - wy;
		q.s.z = ((double)z)/(d-1) - wz;
		vector3_sub_vector(&p, &q, &t);
		dist = ((vector3_length(&t) < dist) ? (vector3_length(&t)) : (dist));
	      }
	    }
	  }
	  gsd[j] = Agsd * exp( (dist*dist) / (-2.0*s*s) );
	}
      }
    }
#endif
    //t1 = omp_get_wtime();
    //printf("Building gaussian kernel took %10.4e sec.\n", t1-t0);

    // Execute the FFT for Gaussian to frequency domain
    //t0 = omp_get_wtime();
    fftw_execute(pg1);
    //t1 = omp_get_wtime();
    //printf("R2C execute for gaussian took %10.4e sec.\n", t1-t0);

    // Get space for pixels' frequency domain.
    if( vsd ) { fftw_free(vsd); }
    if( !(vsd=fftw_malloc(ns*sizeof(double))) ) {
      Error("gaussian_filer(): Failed to allocate array of double: voxels (%d).\n",ns);
    }
    if( vfd ) { fftw_free(vfd); }
    if( !(vfd=fftw_malloc(nf*sizeof(fftw_complex))) ) {
      Error("gaussian_filer(): Failed to allocate array of fftw_complex: voxels (%d).\n",nf);
    }

    // Do the plan for the pixel FFTs.
    //t0 = omp_get_wtime();
    pv1 = fftw_plan_dft_r2c_3d(d, h, w, vsd, vfd, fftw_flags);
    //t1 = omp_get_wtime();
    //printf("R2C plan for pv1 took %10.4e sec.\n", t1-t0);

    //t0 = omp_get_wtime();
    pv2 = fftw_plan_dft_c2r_3d(d, h, w, vfd, vsd, fftw_flags);
    //t1 = omp_get_wtime();
    //printf("C2R plan for pv2 took %10.4e sec.\n", t1-t0);

  }

  // Populate arrays after planning since planning may clobber array data
  //t0 = omp_get_wtime();
  memcpy(vsd,v,ns*sizeof(double));
  //t1 = omp_get_wtime();
  //printf("memcpy input took %10.4e sec.\n", t1-t0);
  //t0 = omp_get_wtime();
  memset(vfd,0,nf*sizeof(fftw_complex));
  //t1 = omp_get_wtime();
  //printf("memset vfd took %10.4e sec.\n", t1-t0);

  // Do the FFT to frequency domain.
  //t0 = omp_get_wtime();
  fftw_execute(pv1);
  //t1 = omp_get_wtime();
  //printf("R2C execute for pv1 took %10.4e sec.\n", t1-t0);

  // Apply the gaussian filter in frequency space.
  for(i=0; i<nf; i++) {
    vfd[i] *= gfd[i];
  }

  // Do the reverse FFT back to voxel space.
  //t0 = omp_get_wtime();
  fftw_execute(pv2);
  //t1 = omp_get_wtime();
  //printf("C2R execute for pv2 took %10.4e sec.\n", t1-t0);

  // Normalize.
  for(i=0; i<w*h*d; i++) {
    vsd[i] /= (w*h*d);
  } 

  // Copy results to v
  //t0 = omp_get_wtime();
  memcpy(v, vsd, ns*sizeof(double));
  //t1 = omp_get_wtime();
  //printf("memcpy solution took %10.4e sec.\n", t1-t0);
}


////////////////////////////////////////////////////////////


void voxelize(state_t *state, float res)
{
  u64b_t nvoxels;
  int    i,j,k,x,y,z,c;

  // Allocate space; resolution is in voxels per Angstrom.
  state->res = res;
  state->vx  = VOXELIZER_GLOBAL_SIZE * res;
  state->vy  = VOXELIZER_GLOBAL_SIZE * res;
  state->vz  = VOXELIZER_GLOBAL_SIZE * res;
  nvoxels = state->vx * state->vy * state->vz;
  if( !(state->voxels=malloc(state->chnls*nvoxels*sizeof(float))) ) {
    Error("voxelize(): Failed to allocate %d bytes for voxels.\n",nvoxels);
  }
  if( !(state->vatoms=malloc(state->chnls*state->natoms*sizeof(int))) ) {
    Error("voxelize(): Failed to allocate %d bytes for atom indices.\n",
	  state->chnls*state->natoms*sizeof(int));
  }
  memset(state->voxels,0,state->chnls*nvoxels*sizeof(float));

  // Voxelize.
  for(i=0; i<state->natoms; i++) {
    // Discritize around the center in voxel space.
    // Already centered around origin (0,0,0) in "real" space.
    x = (state->atoms[i].pos.s.x*res) + state->vx/2.0 + state->res/2.0;
    y = (state->atoms[i].pos.s.y*res) + state->vy/2.0 + state->res/2.0;
    z = (state->atoms[i].pos.s.z*res) + state->vz/2.0 + state->res/2.0;
    // Bounds check; is the box large enough?
    c = atom_channel(state->atoms[i].type);
    if( c == -1 ) {
      Warn("voxelize(): Protein contains unexpected atom '%c'. Ignored.\n",state->atoms[i].type);
      continue;
    }   
    j = x +
        y*(state->vx) +
        z*(state->vx*state->vy) +
        c*(state->vx*state->vy*state->vz);
    if( !((z < state->vz) && (y < state->vy) && (x < state->vx) && (z >= 0) && (y >= 0) && (x >= 0)) ) {
      Error("voxelize(): Voxel index %d (%d,%d,%d) out of bounds (%d); increase size?\n",
	    j,x,y,z,state->chnls*nvoxels);
    }
    // Fill in the voxel.
    state->voxels[j] += 1.0f;
    state->vatoms[i] = j;
    // Record min and max index along each axis as we go.
    int tv[3] = {x,y,z};
    if( i == 0 ) {
      // Initial min/max from first item.
      for(k=0; k<3; k++) {
	state->vmin.a[k] = tv[k];
	state->vmax.a[k] = tv[k];
      }
    }
    for(k=0; k<3; k++) {
      if( state->vmin.a[k] > tv[k]) {
	state->vmin.a[k] = tv[k];
      }
      if( state->vmax.a[k] < tv[k]) {
	state->vmax.a[k] = tv[k];
      }
    }
  }
}


////////////////////////////////////////////////////////////
