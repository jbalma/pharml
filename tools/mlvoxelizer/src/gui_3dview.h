#ifndef GUI_3DVIEW_H
#define GUI_3DVIEW_H


////////////////////////////////////////////////////////////
// For all files

#define GGF_KILL_TIME       (200.0f)
#define GGF_KILL_LIGHT_TIME (100.0f)
#define GGF_FIRE_TIME       (14.0f)
#define GGF_FIRE_LIGHT_TIME (14.0f)

#define GGF_GEM_SLICES 16
#define GGF_GEM_STACKS 16

#include "types.h"
#include "io_bitmap.h"

typedef struct {
  char       *text;     // Text drawn on frame
  u32b_t      help;     // Help mode.
  u32b_t      mode;     // View mode.
  u32b_t      axes;     // Flag for axes drawing.
  u32b_t      boxes;    // Flag for bounding boxes.
  u32b_t      arot;     // Auto-rotate flag.
  u32b_t      md;       // Mouse down flags
  u32b_t      mdx;      // Mouse x-axis coordinate when mouse went down
  u32b_t      mdy;      // Mouse y-axis coordinate when mouse went down
  float       rotx;     // Rotation around x-axis
  float       roty;     // Rotation around y-axis
  float       trnx;     // Translation in x.
  float       trny;     // Translation in y.
  float       zoom;     // Zoom into the origin

  u32b_t      lig;      // Render flag for the ligand.
  u32b_t      prot;     // Render flag for the protein.
  u32b_t      H;        // Render flag for the atom types/channels.
  u32b_t      C;
  u32b_t      O;
  u32b_t      N;
  u32b_t      other;    // All other (less common) atoms.
} frame_gui_t;


////////////////////////////////////////////////////////////
#ifndef GUI_3DVIEW_C
extern void Frame_Draw     (widget_t *w);
extern void Frame_MouseDown(widget_t *w, int x, int y, int b);
extern void Frame_MouseUp  (widget_t *w, int x, int y, int b);
extern void Frame_MouseMove(widget_t *w, int x, int y);
#endif


#endif // !GUI_3DVIEW_H
