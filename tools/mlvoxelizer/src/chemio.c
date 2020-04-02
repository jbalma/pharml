////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <limits.h>

#include "types.h"
#include "util.h"
#include "vector.h"
#include "voxelizer.h"
#include "chemio.h"


////////////////////////////////////////////////////////////
// Input
////////////////////////////////////////////////////////////


void read_inference(state_t *state, char *inf)
{
  char    buf[1024];
  FILE   *f;
  int     l,x,y,z;
  double  bind,xd,yd,zd;

  // Open input file.
  if( !(f=fopen(inf,"r")) ) {
    Error("read_inference(): Failed to open input file.\n");
  }

  // Read a line at a time
  for(l=0; (fgets(buf,sizeof(buf),f) != NULL); l++) {
    // Clip off the new line and any carraige return.
    buf[strlen(buf)-1] = '\0';
    if( buf[strlen(buf)-1] == '\r' ) {
      buf[strlen(buf)-1] = '\0';
    }
    // Parse the line as a simple 5-tuple
    if( sscanf(buf,"%lf, %lf, %lf, %lf\n",&xd,&yd,&zd,&bind) != 4 ) {
      Error("read_inference(): Failed to parse line %d!\n",l);
    }
    x = xd;
    y = yd;
    z = zd;
    // Enlarge inference list.
    state->ninference++;
    if( !(state->inference = realloc(state->inference,state->ninference*sizeof(vector4_t))) ) {
      Error("read_inference(): Failed to enlarge inference array (%d)!\n",state->ninference);
    }
    // Save the x,y,z coord of the prediction.
    state->inference[state->ninference-1].s.x = x;
    state->inference[state->ninference-1].s.y = y;
    state->inference[state->ninference-1].s.z = z;
    // Save the prediction in the inference array.
    state->inference[state->ninference-1].s.w = bind;
  }

  // Close inference data file.
  fclose(f);
}


////////////////////////////////////////////////////////////


void read_pdb(state_t *state, char *pdb)
{
  char  buf[1024],tmp[128];
  FILE *f;
  int   i,j,l,a,t;

  // Open input PDB file.
  if( !(f=fopen(pdb,"r")) ) {
    Error("read_pdb(): Failed to open input PDB file.\n");
  }
  
  // Save a clean version of file name.
  state->pfn = strdup(pdb);
  for(i=strlen(state->pfn)-1; (i >= 0) && (state->pfn[i] != '/'); i--);
  if( state->pfn[i] == '/' ) {
    i++;
  }
  state->pfn = &(state->pfn[i]);
  for(i=strlen(state->pfn)-1; (i >= 0) && (state->pfn[i] != '.'); i--);
  if( state->pfn[i] == '.' ) {
    state->pfn[i] = '\0';
  }

  // Consider the file a line at a time.
  for(l=0; (fgets(buf,sizeof(buf),f) != NULL); l++) {
    buf[strlen(buf)-1] = '\0';
    if( buf[strlen(buf)-1] == '\r' ) {
      buf[strlen(buf)-1] = '\0';
    }

    if( strncmp(buf,"ENDMDL",strlen("ENDMLD")) == 0 ) {
      // For now, just stop parsing after the first "model" is read in.
      break;
    } else if( strncmp(buf,"ATOM",strlen("ATOM")) == 0 ) {
      //
      // Parse "ATOM" lines (protein).
      //
      // ATOM  10143  N   GLY A 831      39.912  17.726  32.558  1.00-0.730           N
      // ATOM  10145  C   GLY A 831      42.161  18.193  33.468  1.00 0.449           C
      if( strlen(buf) != 80 ) {
      	Error("read_pdb(): ATOM line not 80 cols (%d) (l %d).\n",strlen(buf),l);
      }
      state->natoms++;
      if( !(state->atoms=realloc(state->atoms,state->natoms*sizeof(atom_t))) ) {
	Error("read_pdb(): Grow of atoms array failed.\n");
      }
      memset(&(state->atoms[state->natoms-1]),0,sizeof(atom_t));
      // ID.
      memcpy(tmp,buf+6,5);
      tmp[5] = '\0';
      if( sscanf(tmp,"%d",&(state->atoms[state->natoms-1].id)) != 1 ) {
	Error("read_pdb(): Parse ATOM ID failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Atom type.
      memcpy(tmp,buf+76,2);
      tmp[2] = '\0';
      state->atoms[state->natoms-1].type = tmp[1];
      // X coord.
      memcpy(tmp,buf+30,8);
      tmp[8] = '\0';
      if( sscanf(tmp,"%lf",&(state->atoms[state->natoms-1].pos.s.x)) != 1 ) {
	Error("read_pdb(): Parse ATOM x-pos failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Y coord.
      memcpy(tmp,buf+38,8);
      tmp[8] = '\0';
      if( sscanf(tmp,"%lf",&(state->atoms[state->natoms-1].pos.s.y)) != 1 ) {
	Error("read_pdb(): Parse ATOM y-pos failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Z coord.
      memcpy(tmp,buf+46,8);
      tmp[8] = '\0';
      if( sscanf(tmp,"%lf",&(state->atoms[state->natoms-1].pos.s.z)) != 1 ) {
	Error("read_pdb(): Parse ATOM z-pos failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Charge.
      memcpy(tmp,buf+78,2);
      tmp[2] = '\0';
      if( tmp[0] != ' ' ) {
	if( sscanf(tmp,"%d",&(state->atoms[state->natoms-1].charge)) != 1 ) {
	  Error("read_pdb(): Parse ATOM charge failed (\"%s\"). (l %d)\n",tmp,l);
	}
      } else {
	state->atoms[state->natoms-1].charge = 0;
      }
      if( tmp[1] == '-' ) {
	state->atoms[state->natoms-1].charge *= -1;
      }
    } else if( strncmp(buf,"HETATM",strlen("HETATM")) == 0 ) {
      //
      // Parse "HETATOM" lines (ligand).
      //
      // HETATM10211  C2A FA9 A   1      19.653  34.274  30.097  0.50 0.455           C
      if( strlen(buf) != 80 ) {
      	Error("read_pdb(): HETATM line not 80 cols (%d) (l %d).\n",strlen(buf),l);
      }
      state->nligand++;
      if( !(state->ligand=realloc(state->ligand,state->nligand*sizeof(atom_t))) ) {
	Error("read_pdb(): Grow of ligand array failed.\n");
      }
      memset(&(state->ligand[state->nligand-1]),0,sizeof(atom_t));
      // ID.
      memcpy(tmp,buf+6,5);
      tmp[5] = '\0';
      if( sscanf(tmp,"%d",&(state->ligand[state->nligand-1].id)) != 1 ) {
	Error("read_pdb(): Parse HETATM ID failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Atom type.
      memcpy(tmp,buf+76,2);
      tmp[2] = '\0';
      state->ligand[state->nligand-1].type = tmp[1];
      if( tmp[0] == 'Z' && tmp[1] == 'N' ) {
	state->ligand[state->nligand-1].type = 'Z';
      }
      // X coord.
      memcpy(tmp,buf+30,8);
      tmp[8] = '\0';
      if( sscanf(tmp,"%lf",&(state->ligand[state->nligand-1].pos.s.x)) != 1 ) {
	Error("read_pdb(): Parse HETATM x-pos failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Y coord.
      memcpy(tmp,buf+38,8);
      tmp[8] = '\0';
      if( sscanf(tmp,"%lf",&(state->ligand[state->nligand-1].pos.s.y)) != 1 ) {
	Error("read_pdb(): Parse HETATM y-pos failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Z coord.
      memcpy(tmp,buf+46,8);
      tmp[8] = '\0';
      if( sscanf(tmp,"%lf",&(state->ligand[state->nligand-1].pos.s.z)) != 1 ) {
	Error("read_pdb(): Parse HETATM z-pos failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Charge.
      memcpy(tmp,buf+78,2);
      tmp[2] = '\0';
      if( tmp[0] != ' ' ) {
	if( sscanf(tmp,"%d",&(state->ligand[state->nligand-1].charge)) != 1 ) {
	  Error("read_pdb(): Parse HETATM charge failed (\"%s\"). (l %d)\n",tmp,l);
	}
      } else {
	state->ligand[state->nligand-1].charge = 0;
      }
      if( tmp[1] == '-' ) {
	state->ligand[state->nligand-1].charge *= -1;
      }
    } else if( strncmp(buf,"CONECT",strlen("CONECT")) == 0 ) {
      //
      // Parse "CONNECT" lines (ligand).
      //
      // CONECT1015610153101531015510157
      for(i=strlen(buf)-1; i >= 0; i--) {
	if( !isspace(buf[i]) ) {
	  buf[i+1] = '\0';
	  break;
	}
      }
      if( strlen(buf) < 16 ) {
      	Error("read_pdb(): CONNECT line < 16 cols (%d) (l %d).\n",strlen(buf),l);
      }
      if( (strlen(buf)-strlen("CONECT"))%5 != 0 ) {
      	Error("read_pdb(): CONNECT line has unexpected cols (%d) (l %d).\n",strlen(buf),l);	
      }
      // Start.
      memcpy(tmp,buf+6,5);
      tmp[5] = '\0';
      if( sscanf(tmp,"%d",&a) != 1 ) {
	Error("read_pdb(): Parse CONECT start atom failed (\"%s\"). (l %d)\n",tmp,l);
      }
      // Read connected.
      for(i=0; i < (strlen(buf)-strlen("CONECT"))/5; i++) {
	// Parse connected atom id.
	memcpy(tmp,buf+strlen("CONECT")+i*5,5);
	tmp[5] = '\0';
	if( sscanf(tmp,"%d",&t) != 1 ) {
	  Error("read_pdb(): Parse CONECT line failed on id \"%s\". (l %d)\n",tmp,l);
	}
	if( i == 0 ) {
	  // Find first ligand atom in bond.
	  a = -1;
	  for(j=0; j<state->nligand; j++) {
	    if( state->ligand[j].id == t ) {
	      a = j;
	    }
	  }
	  if( a == -1 ) {
	    // Bond connects to protein, discard it for now.
	    // Well, presumably.. It doesn't connect to the ligand anyway.
	    i = 2;
	    break;
	  }
	} else {
	  // Connect to later atoms.
	  // Convert IDs into indicies.
	  int b = -1;
	  for(j=0; j<state->nligand; j++) {
	    if( state->ligand[j].id == t ) {
	      b = j;
	    }
	  }
	  if( b == -1 ) {
	    Warn("  read_pdb(): Could not find ligand atom with id %d.\n",t);
	    continue;
	  }
	  // Do we already have this bond (i.e., upgrade single to double, etc.)?
	  for(j=0; j<state->nbonds; j++) {
	    if( ((state->bonds[j][0] == a) && (state->bonds[j][1] == b)) ||
		((state->bonds[j][0] == b) && (state->bonds[j][1] == a))    ) {
	      if( state->bonds[j][0] == a ) {
		// Only increment bonds starting from a.
		state->bonds[j][2]++;
		state->ligand[a].bonds++;
		state->ligand[b].bonds++;
	      }
	      break;
	    }
	  }
	  if( j >= state->nbonds ) {
	    // Save the new bond's ligand atom indicies.
	    state->nbonds++;
	    if( !(state->bonds=realloc(state->bonds,state->nbonds*4*sizeof(int))) ) {
	      Error("read_pdb(): Grow of bond array failed.\n");
	    }
	    state->bonds[state->nbonds-1][0] = a; // Atom_a.
	    state->bonds[state->nbonds-1][1] = b; // Atom_b.
	    state->bonds[state->nbonds-1][2] = 1; // Bond type.
	    state->bonds[state->nbonds-1][3] = 0; // Aromatic?
	    state->ligand[a].bonds++;
	    state->ligand[b].bonds++;
	  }
	}
      }
    }
  }

  // Finish up with the file.     
  fclose(f);

  // Do some post-processing / cleanup on the data.
  clean_protein(state);
}


////////////////////////////////////////////////////////////


int read_sdf(state_t *state, char *sdf)
{
  char  buf[1024],atype[3];
  FILE *f;
  int   i,l,a,b,t,natoms,nbonds,nligs;

  // Open input SDF file.
  if( !(f=fopen(sdf,"r")) ) {
    Error("read_sdf(): Failed to open input SDF file.\n");
  }
  
  // Save a clean version of file name.
  state->pfn = strdup(sdf);
  for(i=strlen(state->pfn)-1; (i >= 0) && (state->pfn[i] != '/'); i--);
  if( state->pfn[i] == '/' ) {
    i++;
  }
  state->pfn = &(state->pfn[i]);
  for(i=strlen(state->pfn)-1; (i >= 0) && (state->pfn[i] != '.'); i--);
  if( state->pfn[i] == '.' ) {
    state->pfn[i] = '\0';
  }

  // Count number of ligs in this SDF
  for(l=0; fgets(buf,sizeof(buf),f) != NULL; ) {
    if( strncmp(buf,"$$$$",strlen("$$$$")) == 0 ) {
      l++;
    }
  }
  nligs = l;
  if( nligs <= state->ligndx ) {
    Error("read_sdf(): Ligand with index %d not found (out of %d).\n",state->ligndx,l);
  }
  rewind(f);
  
  // Skip past as many entries as needed to reach the requested one.
  for(l=0; (l < state->ligndx) && (fgets(buf,sizeof(buf),f) != NULL); ) {
    if( strncmp(buf,"$$$$",strlen("$$$$")) == 0 ) {
      l++;
    }
  }
  
  // Look for ligand name line.
  if( !fgets(buf,sizeof(buf),f) ) {
    Error("read_sdf(): Read header \"4k\" failed.\n");
  }
  // Skip the two comment lines.
  for(l=0; (l < 2) && (fgets(buf,sizeof(buf),f) != NULL); l++);
  if( l != 2 ) {
    Error("read_sdf(): Read header commend lines failed.\n");
  }
  // Finally read the header line we care about.
  //
  //  70 72  0  0  0  0  0  0  0  0999 V2000
  if( !fgets(buf,sizeof(buf),f) ) {
    Error("read_sdf(): Read header failed.\n");
  }
  if( (sscanf(buf,"%d %d",&natoms,&nbonds) != 2) ) {
    Error("read_sdf(): Parse header (natoms,nbonds) failed.\n");
  }
  // Read atom lines.
  state->nligand = natoms;
  if( !(state->ligand=malloc(state->nligand*sizeof(atom_t))) ) {
    Error("read_sdf(): Grow of ligand array (%d) failed.\n",state->nligand);
  }
  memset(state->ligand,0,state->nligand*sizeof(atom_t));
  for(l=0; (l < state->nligand) && (fgets(buf,sizeof(buf),f) != NULL); l++) {
    //
    // Parse atom lines (ligand).
    //
    //  -1.3810    1.7640    1.8510 C   0  0  0  0  0  0  0  0  0  0  0
    atype[0] = atype[1] = atype[2] = '\0';
    if( sscanf(buf,"%lf %lf %lf %c%c",
	       &(state->ligand[l].pos.s.x),
	       &(state->ligand[l].pos.s.y),
	       &(state->ligand[l].pos.s.z),
	       &(atype[0]),&(atype[1])     ) != 5 ) {
      Error("read_sdf(): Parse atom line failed.\n");
    }
    // Fill in id.
    state->ligand[l].id = l;
    // Convert atom symbol to single char type.
    state->ligand[l].type = toupper(atype[1]);
    if( atom_charge(state->ligand[l].type) == -1 ) {
      state->ligand[l].type = toupper(atype[0]);
      if( atom_charge(state->ligand[l].type) == -1 ) {
	Error("read_sdf(): Unexpected atom type '%c'.\n",state->ligand[l].type);
      }
    }
  }
  // Read bond lines.
  state->nbonds = nbonds;
  if( !(state->bonds=malloc(state->nbonds*4*sizeof(int))) ) {
    Error("read_sdf(): Grow of bond array (%d) failed.\n",state->nbonds);
  }
  memset(state->bonds,0,state->nbonds*4*sizeof(int));
  for(l=0; (l < state->nbonds) && (fgets(buf,sizeof(buf),f) != NULL); l++) {
    //
    // Parse bond lines (ligand).
    //
    //  2  3  1  0  0  0  0
    //  2 13  1  0  0  0  0
    if( (sscanf(buf,"%d %d %d",&a,&b,&t)) != 3 ) {
      Error("read_sdf(): Parse atom line failed.\n");
    }
    // Save the new bond's ligand atom indicies.
    a--;
    b--;
    state->bonds[l][0] = a;   // Atom_a.
    state->bonds[l][1] = b;   // Atom_b.
    state->bonds[l][2] = t;   // Bond type.
    state->bonds[l][3] = 0;   // Aromatic.
    state->ligand[a].bonds += t;
    state->ligand[b].bonds += t;
  }

  // Finish up with the file.     
  fclose(f);

  // Do some post-processing / cleanup on the data.
  clean_protein(state);

  // Return the number of ligands in this SDF (just read one)
  return nligs;
}


////////////////////////////////////////////////////////////


void read_voxels(state_t *state, char *vox)
{
  Warn("read_voxels(): Not yet implemented.\n");
}


////////////////////////////////////////////////////////////

void read_ligand(state_t *state, char *lig)
{
  Warn("read_ligand(): Not yet implemented.\n");
}


////////////////////////////////////////////////////////////


void read_smiles(state_t *state, char *smiles)
{
  Warn("read_smiles(): Not yet implemented.\n");
}


////////////////////////////////////////////////////////////
// Output
////////////////////////////////////////////////////////////


void write_protein_graph(state_t *state, double rad)
{
  static int warn = 0;
  vector3_t  v;
  FILE      *f;
  int        i,j,ti,ne,e,me,Me;
  float      t;
  char       fn[PATH_MAX];

  // Open output file.
  sprintf(fn,"%s_seed_%d.nhg",state->pfn,state->seed);
  if( !(f=fopen(fn,"w")) ) {
    Error("write_protein_graph(): Failed to open output file \"%s\".\n",fn);
  }
  
  // Write stub for n-edges, write real n-nodes.
  ne = 0;
  if( fwrite(&ne, sizeof(int), 1, f) != 1 ) {
    Error("write_protein_graph(): Failed to write n-edges.\n");
  }
  ti = state->natoms;
  if( fwrite(&ti, sizeof(int), 1, f) != 1 ) {
    Error("write_protein_graph(): Failed to write n-verticies.\n");
  }
  
  // Write the nodes:
  int num_atm_active=0;
  for(i=0; i<state->natoms; i++) {
    // Write type.
    t = atom_charge(state->atoms[i].type);
    if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
      Error("write_protein_graph(): Failed to write atom type.\n");
    }
    // Determine if the ith protein atom is within rad and call that active
    t = 0.0f;
    for(j=0; j<state->nligand; j++) {
      vector3_sub_vector(&(state->atoms[i].pos), &(state->ligand[j].pos), &v);
      if( vector3_length(&v) <= rad ) {
	    t = 1.0;
        //num_atm_active++;
      }
    }
    num_atm_active++;

    // Write active site state
    if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
      Error("write_protein_graph(): Failed to write atom bind.\n");
    }    
    // Write position.
    t = state->atoms[i].pos.s.x;
    if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
      Error("write_protein_graph(): Failed to write atom x.\n");
    }
    t = state->atoms[i].pos.s.y;
    if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
      Error("write_protein_graph(): Failed to write atom y.\n");
    }
    t = state->atoms[i].pos.s.z;
    if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
      Error("write_protein_graph(): Failed to write atom z.\n");
    }
  }
  printf("write_protein_graph(): Applied Active Site Labels to %d out of %d protein atoms with rad=%lf... \n",num_atm_active,state->natoms, rad);
  
  // Write the edges.
  me = 100000;
  Me = 0;
  for(i=0; i<state->natoms; i++) {
    // For each atom, consider all other atoms in a radius,
    e = 0;
    for(j=0; j<state->natoms; j++) {
      if( i == j ) {
	continue;
      }
      vector3_sub_vector(&(state->atoms[i].pos), &(state->atoms[j].pos), &v);
      if( vector3_length(&v) <= rad ) {
	// In bounds, add edge to this neighbor to the file.
	t = vector3_length(&v);
	if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
	  Error("write_protein_graph(): Failed to write edge property.\n");
	}
	t = i;
	if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
	  Error("write_protein_graph(): Failed to write edge start.\n");
	}
	t = j;
	if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
	  Error("write_protein_graph(): Failed to write edge end.\n");
	}
	ne++;
	e++;
      }
    }
    // Neighborhood stats for this atom
    if( e < me ) {
      me = e;
    }
    if( e > Me ) {
      Me = e;
    }
    if( !e && !warn ) {
      warn = 1;
      printf("write_protein_graph(): Found protein atom with empty neighborhood!\n");
    }
  }
  
  // Now that we know the number of edges, overwrite the stub.
  rewind(f);
  if( fwrite(&ne, sizeof(int), 1, f) != 1 ) {
    Error("write_protein_graph(): Failed to write n-edges.\n");
  }
  
  // Close file.
  fclose(f);
  
  // Some verbose info
  printf("  edges    =  %d\n",ne);
  printf("  min nbrs =  %d\n",me);
  printf("  max nbrs =  %d\n",Me);
  printf("  avg nbrs =  %.3f\n",((double)ne)/state->natoms);
}


////////////////////////////////////////////////////////////


void write_voxels(state_t *state, int x, int y, int z, int w, int h, int d, char *fn)
{
  FILE *f;
  int   c,i,j,k,ndx;
  float t;

  // Open output file.
  if( !(f=fopen(fn,"w")) ) {
    Error("write_voxels(): Failed to open output file \"%s\".\n",fn);
  }

  // Write channel count and channel names.
  if( fwrite(&(state->chnls), sizeof(int), 1, f) != 1 ) {
    Error("write_voxels(): Failed to write num channels.\n");
  }
  if( fwrite(state->chnlt, strlen(state->chnlt)*sizeof(char), 1, f) != 1 ) {
    Error("write_voxels(): Failed to write num channels.\n");
  }

  // Write position of stencil box/win within global bounds.
  i = x + w/2;
  if( fwrite(&i, sizeof(int), 1, f) != 1 ) {
    Error("write_voxels(): Failed to write x.\n");
  }
  i = y + h/2;
  if( fwrite(&i, sizeof(int), 1, f) != 1 ) {
    Error("write_voxels(): Failed to write y.\n");
  }
  i = z + d/2;
  if( fwrite(&i, sizeof(int), 1, f) != 1 ) {
    Error("write_voxels(): Failed to write z.\n");
  }

  // Write the size of the box.
  if( fwrite(&w, sizeof(int), 1, f) != 1 ) {
    Error("write_voxels(): Failed to write width.\n");
  }
  if( fwrite(&h, sizeof(int), 1, f) != 1 ) {
    Error("write_voxels(): Failed to write height.\n");
  }
  if( fwrite(&d, sizeof(int), 1, f) != 1 ) {
    Error("write_voxels(): Failed to write depth.\n");
  }

  // Now write the voxels.
  for(c=0; c < state->chnls; c++) {
    for(k=z; k < z+d; k++) {
      for(j=y; j < y+h; j++) {
	for(i=x; i < x+w; i++) {
	  ndx = c*(state->vz*state->vy*state->vx) + k*(state->vy*state->vx) + j*(state->vx) + i;
	  if( (k < state->vz) && (j < state->vy) && (i < state->vx) && (k >= 0) && (j >= 0) && (i >= 0) ) {
	    // In bounds, write voxel.
	    if( fwrite(&(state->voxels[ndx]), sizeof(float), 1, f) != 1 ) {
	      Error("write_voxels(): Failed to write voxel (%d,%d,%d) [%d].\n",
		    i, j, k, ndx);
	    }
	  } else {
	    // Out of bounds, so pad.
	    t = 0.0;
	    if( fwrite(&t, sizeof(float), 1, f) != 1 ) {
	      Error("write_voxels(): Failed to write voxel (%d,%d,%d) [%d].\n",
		    i, j, k, ndx);
	    }
	  }
	}
      }
    }
  }

  // Close file.
  fclose(f);
}


////////////////////////////////////////////////////////////
void label_protein_active_site_nodes(state_t *state, int w, int h, int d, int s, int t, int l)
{

}

////////////////////////////////////////////////////////////


void write_stencil_boxes(state_t *state, int w, int h, int d, int s, int t, int l)
{
  char  buf[1024];
  float pct,tot,lpct,o,nv;
  int   c,i,j,k,x,y,z,ndx,nb,f;

  // Consider each possible stencil box position.
  tot = 0.0;
  for(x=-(w-1); x < state->vx+(w-1); x+=s) {
    for(y=-(h-1); y < state->vy+(h-1); y+=s) {
      for(z=-(d-1); z < state->vz+(d-1); z+=s) {
	tot++;
      }
    }
  }
  //tot  = (2.0*(w-1.0))/s * (2.0*(h-1.0))/s * (2.0*(d-1.0)/s);
  pct  = 0.0f;
  lpct = 0.0f;
  o    = 0.0f;
  nb   = 0;
  for(x=-(w-1); x < state->vx+(w-1); x+=s) {
    for(y=-(h-1); y < state->vy+(h-1); y+=s) {
      for(z=-(d-1); z < state->vz+(d-1); z+=s) {
	// Would a stencil box here hold enough non-empty voxels?
	if( l ) {
	  // Ligand-only mode, see if the ligand fits.
	  if( ((x/((double)state->res))    -(VOXELIZER_GLOBAL_SIZE/2.0) < state->lmin.s.x) &&
	      (((x+w)/((double)state->res))-(VOXELIZER_GLOBAL_SIZE/2.0) > state->lmax.s.x) &&
	      ((y/((double)state->res))    -(VOXELIZER_GLOBAL_SIZE/2.0) < state->lmin.s.y) &&
	      (((y+h)/((double)state->res))-(VOXELIZER_GLOBAL_SIZE/2.0) > state->lmax.s.y) &&
	      ((z/((double)state->res))    -(VOXELIZER_GLOBAL_SIZE/2.0) < state->lmin.s.z) &&
	      (((z+d)/((double)state->res))-(VOXELIZER_GLOBAL_SIZE/2.0) > state->lmax.s.z)    ) {
	    // The ligand fits in the box starting at x,y,z with size w,h,d.
	    f = 1;
	  } else {
	    // The ligand doesn't fit in the box.
	    f = 0;
	  }
	} else {
	  // Not ligand-only mode, always say "fits"
	  f = 1;
	}
	nv = 0;
	if( f ) {
	  // Skip the no-fit case.
#pragma omp parallel for private(c,k,j,i,ndx) reduction(+:nv) collapse(2)
	  for(c=0; c < state->chnls; c++) {
	    for(k=z; k < z+d; k++) {
	      for(j=y; j < y+h; j++) {
		for(i=x; i < x+w; i++) {
		  if( (k < state->vz) && (j < state->vy) && (i < state->vx) && (k >= 0) && (j >= 0) && (i >= 0) ) {
		    // Increment num-valid (occupancy) count.
		    ndx = c*(state->vz*state->vy*state->vx) + k*(state->vy*state->vx) + j*(state->vx) + i;
		    nv += state->voxels[ndx];
		  }
		}
	      }
	    }
	  }
	}
	// Make a progress print to stdout.
	pct++;
	if( (pct-lpct)/tot > 0.1 ) {
	  lpct = pct;
	  printf(".");
	  fflush(stdout);
	}
	if( f && (nv >= t) ) {
	  // Non-empty threshold met, write the stencil box.
	  nb++;
	  o += nv;
	  // Do the write.
	  sprintf(buf,"%s_box_%d_%d_%d_seed_%d.vox",state->pfn,x,y,z,state->seed);
	  write_voxels(state, x, y, z, w, h, d, buf);
	}
      }
    }
  }

  // Print out some summary info to stdout.
  printf("\n  Wrote %d stencil boxes, avg occupancy: %.1f.\n",nb,o/nb);
  state->nsb = nb;
}


////////////////////////////////////////////////////////////


void write_ligand(state_t *state, char *fn)
{
  vector3_t  v;
  FILE      *f;
  char       buf[128];
  int        i,j,k;

  // Open output ligand file.
  if( !(f=fopen(fn,"w")) ) {
    Error("write_ligand(): Failed to open output file \"%s\".\n",fn);
  }

  // Write atom count header.
  sprintf(buf,"%d\n",state->nligand);
  if( fwrite(buf, strlen(buf), 1, f) != 1 ) {
    Error("write_ligand(): Failed to write atom count.\n");
  }

  // Write the atomic charge vector.
  for(i=0; i<state->nligand; i++) {
    sprintf(buf,"%2d%c",atom_charge(state->ligand[i].type),
	    ((i==(state->nligand-1))?('\n'):(' ')));
    if( fwrite(buf, strlen(buf), 1, f) != 1 ) {
      Error("write_ligand(): Failed to write atom charge (%d).\n",i);
    }   
  }

  // Write formal charge vector.
  for(i=0; i<state->nligand; i++) {
    if( state->ligand[i].charge < 0 ) {
      sprintf(buf,"%1d%c",state->ligand[i].charge,
	      ((i==(state->nligand-1))?('\n'):(' ')));
    } else if( state->ligand[i].charge > 0 ) {
      sprintf(buf,"+%1d%c",state->ligand[i].charge,
	      ((i==(state->nligand-1))?('\n'):(' ')));
    } else {
      sprintf(buf,"%2d%c",0,
	      ((i==(state->nligand-1))?('\n'):(' ')));      
    }
    if( fwrite(buf, strlen(buf), 1, f) != 1 ) {
      Error("write_ligand(): Failed to write atom charge (%d).\n",i);
    }   
  }

  // Write bond matrix.
  for(i=0; i<state->nligand; i++) {
    for(j=0; j<state->nligand; j++) {
      for(k=0; k<state->nbonds; k++) {
	if( ((state->bonds[k][0] == i) && (state->bonds[k][1] == j)) ||
	    ((state->bonds[k][1] == i) && (state->bonds[k][0] == j))    ) {
	  // Found a bond.
	  sprintf(buf,"%2d%c",state->bonds[k][2],
		  ((j==(state->nligand-1))?('\n'):(' ')));
	  if( fwrite(buf, strlen(buf), 1, f) != 1 ) {
	    Error("write_ligand(): Failed to write bond number (%d,%d).\n",i,j);
	  }
	  break;
	}
      }
      if( k >= state->nbonds ) {
	// No bond.
	sprintf(buf,"%2d%c",0,((j==(state->nligand-1))?('\n'):(' ')));
	if( fwrite(buf, strlen(buf), 1, f) != 1 ) {
	  Error("write_ligand(): Failed to write bond number (%d,%d).\n",i,j);
	}
      }
    }
  }

  // Write distance matrix.
  for(i=0; i<state->nligand; i++) {
    for(j=0; j<state->nligand; j++) {
      vector3_sub_vector(&(state->ligand[i].pos), &(state->ligand[j].pos), &v);
      sprintf(buf,"%.3le%c",vector3_length(&v),
	      ((j==(state->nligand-1))?('\n'):(' ')));
      if( fwrite(buf, strlen(buf), 1, f) != 1 ) {
	Error("write_ligand(): Failed to write bond number (%d,%d).\n",i,j);
      }
    }
  }

  // Close file.
  fclose(f);
}


////////////////////////////////////////////////////////////
