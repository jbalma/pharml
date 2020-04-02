////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <ctype.h>
#include <limits.h>
#include <omp.h>

#include "types.h"
#include "util.h"
#include "random.h"
#include "vector.h"
#include "voxelizer.h"
#include "chemio.h"
#include "fdp.h"
#define MAIN_C
#include "main.h"
#if !defined(NOGUI)
#include "gui.h"
#endif

static state_t *State;

////////////////////////////////////////////////////////////

void tick()
{
  // For demos, running the FDP slowly in a live setting is fun.
  // fdp_ligand(State);
}

static void main_loop()
{
  u64b_t         ticks = 0;
  u64b_t         t1,t2,sleep;

  // Start the GUI
#if !defined(NOGUI)
  printf("Starting GUI:\n");
  StartGUI("MLVoxelizer "VERSION" -- (C) Cray Inc. 2018",((gstate_t*)State));
  UpdateGuiState(((gstate_t*)State));
  printf("Done.\n");
  printf("Control now handed to the GUI.\n");

  while( (State->time = ++ticks) ) {
    // Progress the game one time step
    t1 = get_time();
    tick();
    t2 = get_time();
    // Update the gui
    UpdateGuiState(((gstate_t*)State));
    // Sleep for the remainder of the tick cycle
    if( (t1 < t2) && ((t2-t1) < 10000) ) {
      sleep = 10000 - (t2-t1);
      if( sleep > 0 ) {
	usleep( sleep );
      }
    }
  }
#else
  Error("Compiled without GUI support; rebuild.\n");
#endif
}

////////////////////////////////////////////////////////////

static void init_state()
{
  struct timeval tv;
  
  // Allocate the main state structure
  if( !(State=malloc(sizeof(state_t))) ) {
    Error("Could not allocate state structure (%u)\n",sizeof(state_t));
  }

  // Go ahead and clear the memory to start with.
  memset(State,0,sizeof(state_t));

  // Initialize the random number generator
  if( gettimeofday(&tv, NULL) == -1 ) {
    Warn("gettimeofday() failed; seeding random generator with 7 instead.\n");
    State->seed = 7;
  } else {
    State->seed = (u32b_t)(tv.tv_usec);
  }
  random_initrand(&State->random,State->seed);
  printf("Random Seed:  %d\n",State->seed);

  // Set the current time
  State->time = 1;

  // Fill in some default options.
  State->res    = 3.0;
  State->win_w  = State->win_h = State->win_d = 20;
  State->stride = 3.0;
  State->valid  = 10;
  State->lonly  = 1;
  State->sigma  = 0.001;
  State->fftwplanrigor  = 0.0;
  State->dout   = "./data/";
  State->chnlt  = atom_channels();
  State->chnls  = strlen(State->chnlt);
}

////////////////////////////////////////////////////////////

static void protein()
{
  double *a;
  char    cwd[PATH_MAX];
  int     nv,i,c;
  long    t;

  // Make sure there is some kind of protein input file.
  if( !State->pfn ) {
    return;
  }

  printf("\n+---+\n"
	 "| P |\n"
	 "+---+\n\n");

  // Read input PDB or VOX file.
  if( State->pvox ) {
    printf("Reading VOX \"%s\":\n",State->pfn);
    t = get_time();
    read_voxels(State, State->pfn);
  } else {
    printf("Reading PDB \"%s\":\n",State->pfn);
    t = get_time();
    read_pdb(State, State->pfn);
  }

  // Print protein stats from read file.
  printf("  protein:\n");
  printf("    atoms = %d\n",State->natoms);
  printf("    size  = %.1f x %.1f x %.1f A\n",
	 (State->amax.s.x-State->amin.s.x)/2.0,
	 (State->amax.s.y-State->amin.s.y)/2.0,
	 (State->amax.s.z-State->amin.s.z)/2.0);
  if( !(State->pvox) ) {
    // PDB files can contain ligans so print those stats as well.
    printf("  ligand:\n");
    printf("    nligs = 1\n");
    printf("    atoms = %d\n",State->nligand);
    printf("    bonds = %d\n",State->nbonds);
    printf("    size  = %.1f x %.1f x %.1f A\n",
	   (State->lmax.s.x-State->lmin.s.x)/2.0,
	   (State->lmax.s.y-State->lmin.s.y)/2.0,
	   (State->lmax.s.z-State->lmin.s.z)/2.0);
  }
  t = get_time() - t;
  printf("Done.  (%.3lf s)\n",t/1000000.0);

  // Voxelize atoms.
  printf("\n+---+\n"
	 "| V |\n"
	 "+---+\n\n");

  printf("Voxelizing:\n");
  printf("  resolution = %.1f voxels per Angstrom.\n",State->res);
  printf("  channels   = %s\n",State->chnlt);
  t = get_time();
  if( State->rotate ) {
    rotation(State);
  }
  voxelize(State, State->res);
  // Apply gaussian filter if needed.
  if( State->sigma != 0.0 ) {
    printf("  Applying gaussian:\n");
    printf("    sigma    = %f\n",State->sigma);
    nv = State->vx*State->vy*State->vz;
    if( !(a=malloc(nv*sizeof(double))) ) {
      Error("  voxelize_pdb(): Failed to allocate space for FFT window.\n");
    }
    printf("    progress = ");
    fflush(stdout);
    for(c=0; c<State->chnls; c++) {
      printf(".");
      fflush(stdout);
      for(i=0; i<nv; i++) {
	a[i] = State->voxels[c*nv+i];
      }
      gaussian(a, State->vx, State->vy, State->vz, State->sigma, State->fftwplanrigor);
      for(i=0; i<nv; i++) {
	State->voxels[c*nv+i] = a[i];
      }
    }
    free(a);
    gaussian(NULL, 0, 0, 0, 0.0, 0.0);
    printf("\n  Done.\n");
  }
  t = get_time() - t;
  printf("Done.  (%.3lf s)\n",t/1000000.0);

  // Write protein output
  if( !State->readonly ) {
    // Write stencil boxes.
    printf("\nWriting stencil boxes:\n"
	   "  size     =  %.1fx%.1fx%.1f A,\n"
	   "  stride   =  %.1f A,\n",
	   State->win_w, State->win_h, State->win_d,
	   State->stride);
    printf("  valid    =  %s >%.1f non-empty voxels.\n",
	   ((State->lonly)?("ligand"):("all")),
	   State->valid);
    t = get_time();
    mkdir(State->dout,S_IRWXU|S_IRWXG);
    if( !getcwd(cwd,sizeof(cwd)) ) {
      Error("Failed to save current working dir path (getcwd).\n");
    }
    if( chdir(State->dout) ) {
      Error("Failed to cd to data dir \"%s\".\n",State->dout);
    }
    printf("  progress = .");
    fflush(stdout);
    write_stencil_boxes(State, 
			State->win_w*(State->res), 
			State->win_h*(State->res), 
			State->win_d*(State->res), 
			State->stride*(State->res),
			State->valid,
			State->lonly);
    t = get_time() - t;
    printf("Done.  (%.3lf s)\n",t/1000000.0);
    if( chdir(cwd) ) {
      Error("Failed to cd to data dir parent.\n");
    }
    // Write protein as "neighborhood" graph.
    double rad = 2.5;
    printf("\nWriting protein graph:\n"
	   "  nbr_size =  %.1f A (radius)\n",
	   rad);
    t = get_time();
    mkdir(State->dout,S_IRWXU|S_IRWXG);
    if( !getcwd(cwd,sizeof(cwd)) ) {
      Error("Failed to save current working dir path (getcwd).\n");
    }
    if( chdir(State->dout) ) {
      Error("Failed to cd to data dir \"%s\".\n",State->dout);
    }
    write_protein_graph(State, rad);
    t = get_time() - t;
    printf("Done.  (%.3lf s)\n",t/1000000.0);
    if( chdir(cwd) ) {
      Error("Failed to cd to data dir parent.\n");
    }
  }
}

static void ligand()
{
  char buf[PATH_MAX],cwd[PATH_MAX];
  int  i,nligs;
  long t;

  printf("\n+---+\n"
	 "| L |\n"
	 "+---+\n\n");

  // Stdout header.
  printf("Processing ligand:\n");
  t = get_time();

  // Read ligand input.
  if( State->lfn ) {
    // Overwrite previous?
    printf("  Reading ligand file: \"%s\"\n",State->lfn);
    read_ligand(State, State->lfn);
  }
  if( State->sdf ) {
    // Overwrite previous?
    printf("  Reading SDF ligand file: \"%s\"\n",State->sdf);
    nligs = read_sdf(State, State->sdf);
    printf("    nligs  = %d\n",nligs);
    printf("    ligndx = %d\n",State->ligndx);
    printf("    atoms  = %d\n",State->nligand);
    printf("    bonds  = %d\n",State->nbonds);
    printf("    size   = %.1f x %.1f x %.1f A\n",
	   (State->lmax.s.x-State->lmin.s.x)/2.0,
	   (State->lmax.s.y-State->lmin.s.y)/2.0,
	   (State->lmax.s.z-State->lmin.s.z)/2.0);
    printf("  Done.\n");
  }

  // Do the manual SMILES string last if present.
  if( State->smiles ) {
    // Overwrite PDB / LIG?
    printf("  Parse SMILES: \"%s\"\n",State->smiles);
    read_smiles(State, State->smiles);
  }

  // Sanity check.
  if( State->nligand == 0 ) {
    Warn("  ligand(): No ligand; add a source (--pdb, --sdf, --lig, --smiles).\n");
    return;
  }

  // Apply force-directed placement if needed.
  if( State->fdp ) {
    i = fdp_ligand(State);
    if( i < 0 ) {
      printf("  FDP failed to converge in %d timesteps.\n",-i);
    } else {
      printf("  FDP converged in %d timesteps.\n",i);
    }
  }

  // Get into data dir.
  if( !State->readonly ) {
    mkdir(State->dout,S_IRWXU|S_IRWXG);
    if( !getcwd(cwd,sizeof(cwd)) ) {
      Error("Failed to save current working dir path (getcwd).\n");
    }
    if( chdir(State->dout) ) {
      Error("Failed to cd to data dir \"%s\".\n",State->dout);
    }
    // Write ligand output.
    int ridx = rand() % 100 + 1;
    if( State->lfn ) {
      sprintf(buf,"%s_%d_%d.lig",State->lfn,State->ligndx,ridx);
    } else {
      sprintf(buf,"%s_%d_%d.lig",State->pfn,State->ligndx,ridx);
    }
    printf("  write VM format: \"%s\"\n",buf);
    write_ligand(State, buf);
    t = get_time() - t;
    printf("Done.  (%.3lf s)\n",t/1000000.0); 
    // Change dir back to parent.
    if( chdir(cwd) ) {
      Error("Failed to cd to data dir parent.\n");
    }
  }

  // Need to call this just for the init if needed.
  if( !(State->vx) ) {
    voxelize(State, State->res);
  }
}

////////////////////////////////////////////////////////////

static void usage_error()
{
  Error("\nusage:\n\tmlvoxelizer <in> [opt_1] [...] [opt_n]\n"
	"\n"
	"\t<in> is from:\n"
	"\t--pdb    <path to input PDB file>\n"
	"\t--vox    <path to input voxel file>\n"
	"\t--lig    <path to input ligand file>\n"
	"\t--sdf    <path to input SDF ligand file>\n"
	"\t--smiles <ligand SMILES string>\n"
	"\n"
	"\t[opt_*] is from:\n"
	"\t--out    <output directory>          // ./data/\n"
	"\t--res    <voxels per angstrom>       // %.1f\n"
	"\t--win    <win_w>x<win_h>x<win_d>     // %.1fx%.1fx%.1f\n"
	"\t--stride <win stride in angstroms>   // %.1f\n"
	"\t--gauss  <sigma>                     // %f\n"
	"\t--fftwplanrigor <seconds>            // %f\n"
	"\t--valid  <all|ligand>                // ligand\n"
	"\t--ligndx <ligand index>              // 0\n"
	"\t--inf    <path to inference file>    // n/a\n"
	"\t--gui                                // no\n"
	"\t--gui-basic                          // no\n"
	"\t--fdp                                // no\n"
	"\t--protonate                          // no\n"
	"\t--readonly                           // no\n"
	"\t--rotate                             // no\n",
	State->res,
	State->win_w, State->win_h, State->win_d,
	State->stride,
	State->sigma);
}

static void parsecl(int argc, char **argv)
{
  float tf,w,h,d;
  int   i,ti;

  // Check command line args.
  if( argc < 3 ) {
    usage_error();
  }
  
  for(i=1; i<argc; i++) {
    if( !strcmp(argv[i],"--pdb") ) {
      // Input PDB file.
      if( ++i >= argc ) {
	usage_error();	
      }
      State->pfn = strdup(argv[i]);
    } else if( !strcmp(argv[i],"--vox") ) {
      // Input voxel file.
      if( ++i >= argc ) {
	usage_error();	
      }
      State->pfn  = strdup(argv[i]);
      State->pvox = 1;
    } else if( !strcmp(argv[i],"--lig") ) {
      // Input ligand file.
      if( ++i >= argc ) {
	usage_error();	
      }
      State->lfn = strdup(argv[i]);     
    } else if( !strcmp(argv[i],"--sdf") ) {
      // Input SDF ligand file.
      if( ++i >= argc ) {
	usage_error();	
      }
      State->sdf = strdup(argv[i]);     
    } else if( !strcmp(argv[i],"--smiles") ) {
      // Input ligand smiles string.
      if( ++i >= argc ) {
	usage_error();	
      }
      State->smiles = strdup(argv[i]);     
    } else if( !strcmp(argv[i],"--out") ) {
      // Output directory.
      if( ++i >= argc ) {
	usage_error();	
      }
      State->dout = strdup(argv[i]);
    } else if( !strcmp(argv[i],"--inf") ) {
      // Input inference file.
      if( ++i >= argc ) {
	usage_error();	
      }
      // Should be harmless to just read this in right here.
      read_inference(State, argv[i]);
    } else if( !strcmp(argv[i],"--protonate") ) {
      // Protonate flag.
      State->protonate = 1;
    } else if( !strcmp(argv[i],"--rotate") ) {
      // Rotate protein flag.
      State->rotate = 1;
    } else if( !strcmp(argv[i],"--readonly") ) {
      // Rotate protein flag.
      State->readonly = 1;
    } else if( !strcmp(argv[i],"--gui") ) {
      // GUI flag.
      State->gui = 1;
    } else if( !strcmp(argv[i],"--fdp") ) {
      // Force-directed placement flag.
      State->fdp = 1;
    } else if( !strcmp(argv[i],"--gui-basic") ) {
      // GUI basic (low-quality) flag.
      State->gui = 2;
    } else if( !strcmp(argv[i],"--ligndx") ) {
      // Ligand index (multi-lig files).
      if( ++i >= argc ) {
	usage_error();
      }
      if( sscanf(argv[i],"%d",&ti) != 1 ) {
	usage_error();	
      }
      State->ligndx = ti;
    } else if( !strcmp(argv[i],"--res") ) {
      // Resolution.
      if( ++i >= argc ) {
	usage_error();
      }
      if( sscanf(argv[i],"%f",&tf) != 1 ) {
	usage_error();	
      }
      State->res = tf;
    } else if( !strcmp(argv[i],"--win") ) {
      // Stencil window size.
      if( ++i >= argc ) {
	usage_error();
      }
      if( sscanf(argv[i],"%fx%fx%f",&w,&h,&d) != 3 ) {
	usage_error();	
      }
      State->win_w = w;
      State->win_h = h;
      State->win_d = d;
    } else if( !strcmp(argv[i],"--stride") ) {
      // Stencil stride.
      if( ++i >= argc ) {
	usage_error();
      }
      if( sscanf(argv[i],"%f",&tf) != 1 ) {
	usage_error();	
      }
      State->stride = tf;
    } else if( !strcmp(argv[i],"--valid") ) {
      // Valid count.
      if( ++i >= argc ) {
	usage_error();
      }
      if( !strcmp(argv[i],"ligand") ) {
	State->lonly = 1;
      } else {
	if( !strcmp(argv[i],"all") ) {
	  State->lonly = 0;
	} else {
	  usage_error();
	}
      }
    } else if( !strcmp(argv[i],"--gauss") ) {
      // Resolution.
      if( ++i >= argc ) {
	usage_error();
      }
      if( sscanf(argv[i],"%f",&tf) != 1 ) {
	usage_error();	
      }
      State->sigma = tf;
    } else if( !strcmp(argv[i],"--fftwplanrigor") ) {
      // Resolution.
      if( ++i >= argc ) {
	usage_error();
      }
      if( sscanf(argv[i],"%f",&tf) != 1 ) {
	usage_error();
      }
      State->fftwplanrigor = tf;
    }  else {
      usage_error();
    }
  }

  // Post parse checks.
  if( (!State->pfn) && (!State->lfn) && (!State->sdf) ) {
    usage_error();
  }
}

int main(int argc, char *argv[])
{
  // Print a startup message.
  printf("+---+ +---+ +---+\n"
	 "| M | | L | | Vo|xelizer "VERSION" -- (C) Cray Inc. 2018\n"
	 "+---+ +---+ +---+\n\n");

  printf("Version:      %s\n",VERSION);
  printf("FFTW Version: %s\n",fftw_version_string());
  printf("OMP Threads:  %d\n",omp_get_max_threads());

  // Get ready to do stuffs.
  init_state();

  // Parse command-line args.
  parsecl(argc, argv);

  // Protein: read, voxelize, output in voxel format.
  protein();

  // Ligand: read, verify, output in VM format.
  ligand();

  // Main loop and GUI startup.
  if( State->gui ) {
    printf("\n+---+\n"
	   "| G |\n"
	   "+---+\n\n");
    main_loop();
  }

  // Done, exit.
  return 0;
}

////////////////////////////////////////////////////////////
