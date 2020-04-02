#ifndef GUI_H
#define GUI_H

#include <pthread.h>

#include "types.h"
#include "atom.h"
#include "random.h"
#include "io_bitmap.h"

// GUI's version of state
typedef struct str_gstate_t {
  u64b_t      time;         // Current time
  rnd_t       random;       // Random number generator state
  u32b_t      seed;         // Initial random seed
  int         gui;          // Flag for using the GUI.
  int         fdp;          // Force-directed placement flag.
  int         pvox;         // Protein input is vox file (flag).
  int         protonate;    // Flag for adding assumed hydrogen.
  int         ligndx;       // Ligand index for multi-lig files.
  int         rotate;       // Rotate the protein (random) flag.
  int         readonly;     // Readonly flag.

  char       *pfn;          // Protein file name.
  char       *sdf;          // Ligand SDF library file name.
  char       *lfn;          // Ligand file name.
  char       *smiles;       // Ligand smiles string.
  char       *dout;         // Output data directory.

  atom_t     *atoms;        // Array of atoms (protein).
  int         natoms;       // number of atoms (protein).
  vector3_t   amin;         // Min coord of atoms.
  vector3_t   amax;         // Max coord of atoms.

  float       win_w;        // Stencil window width
  float       win_h;        // Stencil window height
  float       win_d;        // Stencil window depth
  float       stride;       // Stencil window stride
  float       valid;        // Stencil window min non-empty voxels.
  int         lonly;        // Ligand only mode.

  int         chnls;        // Number of atom channels (types).
  char       *chnlt;        // Atom type of each channel.
  float       res;          // Resolution in voxels per Angstrom.
  float      *voxels;       // Voxelized version.
  int        *vatoms;       // Index of each atom into voxel array.
  int         vx;           // Width of voxels.
  int         vy;           // Height of voxels.
  int         vz;           // Depth of voxels.
  vector3_t   vmin;         // Min coord of atoms.
  vector3_t   vmax;         // Max coord of atoms.
  int         nsb;          // Number of stencil boxes.
  float       sigma;        // Gaussian filter for voxels.
  float       fftwplanrigor;// Time limit in seconds for FFTW plan.  
                            // If>0 then FFTW_MEASURE, else FFTW_ESTIMATE

  atom_t     *ligand;       // Array of atoms (ligand).
  int         nligand;      // number of atoms (ligand).
  int       (*bonds)[4];    // Array of bonds in the ligand.
  int        nbonds;        // Number of bonds in bond array.
  vector3_t   lmin;         // Min coord of ligand.
  vector3_t   lmax;         // Max coord of ligand.

  vector4_t  *inference;    // Inference results / bind predictions.
  int        ninference;    // Number of predictions in inference array.
} gstate_t;

// GUI's internal state
typedef struct str_guistate_t {
  u32b_t     mouse_item_ndx;  // Current item "held" by the mouse
  vector3_t  hand_pos;        // Last position of mouse / hand
  vector3_t  enemy_hit_pos;   // Where the enemy hit the player
  vector3_t  enemy_hit_color; // Color of the enemy who hit
} guistate_t;

////////////////////////////////////////////////////////////
// For GUI and widget files
////////////////////////////////////////////////////////////
#ifdef GUI_WIDGET
#include <X11/Xlib.h>
#include <GL/glx.h>
#include <GL/gl.h>

// Mouse button codes
#define MOUSE_LEFT   1
#define MOUSE_MIDDLE 2
#define MOUSE_RIGHT  3
#define MOUSE_UP     4
#define MOUSE_DOWN   5

// key and keycode
// These are probably machine + OS/xwindows specific
#define BACKSPACE 8   // Key
#define ESCAPE    27
#define ENTER     13
#define PAGEUP    99  // Keycode
#define PAGEDOWN  105
#define UP        98
#define DOWN      104
#define LEFT      100
#define RIGHT     102
#define F1        67
#define F2        68
#define F3        69
#define F4        70
#define F5        71
#define F6        72
#define F7        73
#define F8        74
#define F9        75
#define F10       76
#define F11       95
#define F12       96

// Holds what is needed to manage an opengl enabled window
typedef struct {
  Display             *dpy;
  int                  screen;
  Window               win;
  GLXContext           ctx;
  XSetWindowAttributes attr;
  int                  x, y;
  unsigned int         width,  height;
  unsigned int         pwidth, pheight;
  unsigned int         depth;   
  GLuint               font;
  int                  id;
} glwindow_t;

// Widget types and callbacks
typedef struct st_widget widget_t;

typedef void (*cbktick_t)      (widget_t *w);
typedef void (*cbkdraw_t)      (widget_t *w);
typedef void (*cbkkeypress_t)  (widget_t *w, char key, unsigned int keycode);
typedef void (*cbkmousedown_t) (widget_t *w, int x, int y, int b);
typedef void (*cbkmouseup_t)   (widget_t *w, int x, int y, int b);
typedef void (*cbkmousemove_t) (widget_t *w, int x, int y);

struct st_widget {
  glwindow_t     *glw;        // Window information
  float           x;          // x-pos
  float           y;          // y-pos
  float           w;          // width
  float           h;          // height
  cbkdraw_t       draw;       //
  cbkkeypress_t   keypress;   // 
  cbkmousedown_t  mousedown;  // Event callbacks
  cbkmouseup_t    mouseup;    //
  cbkmousemove_t  mousemove;  // 
  cbktick_t       tick;       // 
  int             update;     // Update request flag
  void           *wd;         // Custom widget data
};
#endif

////////////////////////////////////////////////////////////
// For files other than gui.c
////////////////////////////////////////////////////////////
#ifndef GUI_C

#ifdef MAIN_C
// These are intended for the simulation to start/update the gui
extern int  StartGUI(char *version, gstate_t *s); 
extern void UpdateGuiState(gstate_t *s);
#endif // !GUI_WIDGET

// These are intended for use by widgets / gui subcomponents
#ifdef GUI_WIDGET
extern guistate_t        GuiState;
extern gstate_t         *Stateg,*Statec,*Statep;
extern pthread_mutex_t   StateLock;

extern void   printGLf(GLuint font, const char *fmt, ...);
extern u32b_t LoadTexture(char *fn);
extern void   ViewPort3D(int x, int y, int w, int h);
extern void   ViewPort2D(glwindow_t *glw);

extern void Cyan();
extern void Yellow();
extern void Red();
extern void Purple();
extern void Blue();
extern void White();
extern void Green();
extern void Black();

extern float ScaleX(widget_t *w, const float x);
extern float ScaleY(widget_t *w, const float y);
extern void  GuiExit();

#endif // GUI_WIDGET
#endif // !GUI_C

#endif // !GUI_H
