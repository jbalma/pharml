#ifndef GUI_C
#define GUI_C
#define GUI_WIDGET

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <malloc.h>
#include <sys/types.h>
#include <sys/time.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <X11/Xlib.h>
#include <GL/glx.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <pthread.h>
#include <math.h> 
#include <signal.h>

#include "types.h"
#include "util.h"
#include "main.h"
#include "gui.h"
#include "gui_button.h"
#include "gui_3dview.h"
#include "gui_stats.h"
#include "io_bitmap.h"

////////////////////////////////////////////////////////////////////////////////

// Local to gui.c only
static volatile int  Redraw=1;
static char         *Version;
static widget_t      Widgets[128];
static u32b_t        nWidgets;
static int           Done;

// Exported to gui_*.c
gstate_t         *Stateg,*Statec,*Statep;
pthread_mutex_t   StateLock=PTHREAD_MUTEX_INITIALIZER;
guistate_t        GuiState;

////////////////////////////////////////////////////////////////////////////////


//
// All of these are exported and made available to widgets in other files
//

void printGLf(unsigned int font, const char *fmt, ...)
{
  va_list ap;  char text[256];

  if (fmt == NULL) return;

  va_start(ap, fmt);  
  vsprintf(text, fmt, ap);
  va_end(ap);
  glPushAttrib(GL_LIST_BIT);
  glListBase(font - 32);
  glCallLists(strlen(text), GL_UNSIGNED_BYTE, text);
  glPopAttrib();
}

u32b_t LoadTexture(char *fn)
{
  io_bitmap_t bmp;
  u32b_t      tex;

  // Read the bitmap data from a file
  io_bitmap_load(fn, &bmp);
  
  // Turn the bmp into an OpenGL texture with linear filtering
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, bmp.w, bmp.h, 0, GL_RGB, GL_UNSIGNED_BYTE, bmp.d);
  
  // Free the bmp pixel data
  io_bitmap_free(&bmp);
  
  // Return the OpenGL texture handle
  return tex;
}

void ViewPort2D(glwindow_t *glw)
{
  glViewport(0, 0, glw->width, glw->height); 
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, glw->width, glw->height, 0.0, 1, -100);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void ViewPort3D(int x, int y, int w, int h)
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glViewport(x, y, w, h); 
  gluPerspective(60.0f,(float)w/(float)h,0.1f,100.0f);
  glTranslatef(0.0f, 0.0f, -2.0f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void Black()  { glColor3ub(  0,   0,   0); }
void Cyan()   { glColor3ub(  0, 255, 255); }
void Yellow() { glColor3ub(255, 255,   0); }
void Red()    { glColor3ub(255,   0,   0); }
void Purple() { glColor3ub(255,   0, 255); }
void Blue()   { glColor3ub(  0,   0, 255); }
void White()  { glColor3ub(255, 255, 255); }
void Green()  { glColor3ub(  0, 255,   0); }

float ScaleX(widget_t *w, const float x)
{
  return x * ((float)w->glw->width)/w->glw->pwidth;
}

float ScaleY(widget_t *w, const float y)
{
  return y * ((float)w->glw->height)/w->glw->pheight;
}

void GuiExit()
{
  // We are done, probably because the user clicked on the "x" (close button)
  // or some other input event (keystroke) was interpreted as a quit request.
  // I want the signal hanlder of the simulation thread to be the only exit 
  // point, so I'll just signal the main process here.
  kill(getpid(),SIGTERM);
}

////////////////////////////////////////////////////////////////////////////////

static u32b_t BuildFont(glwindow_t *glw, char *fname)
{
  XFontStruct *font;  GLuint fbase;   
    
  // Storage for 96 characters
  fbase = glGenLists(96);      

  // Load a font with a specific name in "Host Portable Character Encoding"
  if ( !(font = XLoadQueryFont(glw->dpy, fname)) ) {
    fprintf(stderr,"Could not load font \"%s\"\n",fname);
    exit(-1);
  }

  // Build 96 display lists out of our font starting at char 32
  glXUseXFont(font->fid, 32, 96, fbase);
  XFreeFont(glw->dpy, font);
  
  return fbase;
}

static void InitGLWindow(glwindow_t *glw)
{
  glShadeModel(GL_SMOOTH);
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClearDepth(1.0f);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_DEPTH_TEST);
  glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
  glEnable(GL_NORMALIZE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);
  glw->font = BuildFont(glw, "fixed");   
  ViewPort2D(glw);
  glFlush();
}

static void StartUp(glwindow_t *glw, char* title, int width, int height)
{
  XVisualInfo *vi;  Colormap cmap;  Window twin;  Atom wmDelete;  XSizeHints *xsh;  int glxM, glxm, t;
  int attrListSgl[] = {GLX_RGBA,GLX_RED_SIZE,4,GLX_GREEN_SIZE,4,GLX_BLUE_SIZE,4,GLX_DEPTH_SIZE,16,None};
  int attrListDbl[] = {GLX_RGBA,GLX_DOUBLEBUFFER,GLX_RED_SIZE,4,GLX_GREEN_SIZE,4,GLX_BLUE_SIZE,4,GLX_DEPTH_SIZE,16,None};

  // Connect to X
  if( !(glw->dpy = XOpenDisplay(0)) ) {
    fprintf(stderr,"Cannot connect to X server!\n");
    exit(1);
  }
  glxM = XProtocolVersion(glw->dpy);
  glxm = XProtocolRevision(glw->dpy);
  printf("  X:      %d.%d %s %d\n", glxM, glxm,
	 XServerVendor(glw->dpy),XVendorRelease(glw->dpy));
  glw->screen = DefaultScreen(glw->dpy);

  // Get the appropriate visual
  glXQueryVersion(glw->dpy, &glxM, &glxm);
  printf("  GLX:    %d.%d\n", glxM, glxm);
  if ( (vi = glXChooseVisual(glw->dpy, glw->screen, attrListDbl)) ) {
    //printf("  Buffer: double\n");
  } else {
    vi = glXChooseVisual(glw->dpy, glw->screen, attrListSgl);
    //printf("  Buffer: single\n");
  }
  
  // Create a GLX context and color map
  glw->ctx               = glXCreateContext(glw->dpy, vi, 0, GL_TRUE);
  cmap                   = XCreateColormap(glw->dpy, RootWindow(glw->dpy, vi->screen), vi->visual, AllocNone);
  glw->attr.colormap     = cmap;
  glw->attr.border_pixel = 0;

  // Create the window
  glw->attr.event_mask = ExposureMask | KeyPressMask | ButtonPressMask| ButtonReleaseMask | StructureNotifyMask | PointerMotionMask;
  glw->win = XCreateWindow(glw->dpy, RootWindow(glw->dpy, vi->screen),
			   0, 0, width, height, 0, vi->depth, InputOutput, vi->visual,
			   CWBorderPixel | CWColormap | CWEventMask, &glw->attr);
  wmDelete = XInternAtom(glw->dpy, "WM_DELETE_WINDOW", True);
  XSetWMProtocols(glw->dpy, glw->win, &wmDelete, 1);
  XSetStandardProperties(glw->dpy, glw->win, title, title, None, NULL, 0, NULL);
  XMapRaised(glw->dpy, glw->win);
    
  // Connect the glx-context to the window
  glXMakeCurrent(glw->dpy, glw->win, glw->ctx);
  printf("  OpenGL: %s\n", glGetString(GL_VERSION));
  XGetGeometry(glw->dpy, glw->win, &twin, &glw->x, &glw->y,
	       &glw->width, &glw->height, (unsigned int*)&t, &glw->depth);
  //printf("  Depth:  %d\n", glw->depth);
  if (glXIsDirect(glw->dpy, glw->ctx)) {
    //printf("  Direct: yes\n");
  } else {
    //printf("  Direct: no\n");
  }

  // Attempt to set a size and aspect hint
  if( (xsh=XAllocSizeHints()) ) {
    xsh->flags      = (PMinSize|PAspect);
    xsh->min_width  = width/1.25;
    xsh->min_height = height/1.25;
    xsh->min_aspect.x = width;  // n
    xsh->min_aspect.y = height; // d
    xsh->max_aspect.x = width;  // n
    xsh->max_aspect.y = height; // d 
    XSetWMNormalHints(glw->dpy, glw->win, xsh);
    XFree(xsh);
  }

  InitGLWindow(glw);
}

////////////////////////////////////////////////////////////////////////////////

//
// Button callbacks
//

void Quit_Down(widget_t *w, const int x, const int y, const int b)
{
  if( (x > ScaleX(w,w->x)) && (x < ScaleX(w,w->x+w->w)) && 
      (y > ScaleY(w,w->y)) && (y < ScaleY(w,w->y+w->h))     ) {
    GuiExit();
  }
}

void Mode_Down(widget_t *w, const int x, const int y, const int b)
{
  button_gui_t *btn = (button_gui_t*)(w->wd);
  
  if( (x > ScaleX(w,w->x)) && (x < ScaleX(w,w->x+w->w)) && 
      (y > ScaleY(w,w->y)) && (y < ScaleY(w,w->y+w->h))     ) {
    *(btn->link) = (*(btn->link)+1)%3;
    switch( *(btn->link) ) {
    case 0:
      sprintf(btn->text,"Render: Voxels");
      break;
    case 1:
      sprintf(btn->text,"Render: Atoms");
      break;
    case 2:
      sprintf(btn->text,"Render: Volume");
      break;
    }
  }
}

void Toggle_Down(widget_t *w, const int x, const int y, const int b)
{
  button_gui_t *btn = (button_gui_t*)(w->wd);
  
  if( (x > ScaleX(w,w->x)) && (x < ScaleX(w,w->x+w->w)) && 
      (y > ScaleY(w,w->y)) && (y < ScaleY(w,w->y+w->h))     ) {
    *(btn->link) ^= 1;
    btn->sel ^= 1;
  }
}

////////////////////////////////////////////////////////////////////////////////

static void AddWidget(glwindow_t *glw, float  x, float y, float w, float h,
	       cbkdraw_t draw, cbkkeypress_t keypress, cbkmousedown_t mousedown, 
	       cbkmouseup_t mouseup, cbkmousemove_t  mousemove, cbktick_t tick, void *wd)
{
  Widgets[nWidgets].glw       = glw;
  Widgets[nWidgets].x         = x;
  Widgets[nWidgets].y         = y;
  Widgets[nWidgets].w         = w;
  Widgets[nWidgets].h         = h;
  Widgets[nWidgets].tick      = tick;
  Widgets[nWidgets].draw      = draw;
  Widgets[nWidgets].keypress  = keypress;
  Widgets[nWidgets].mousedown = mousedown;
  Widgets[nWidgets].mouseup   = mouseup;
  Widgets[nWidgets].mousemove = mousemove;
  Widgets[nWidgets].wd        = wd;
  nWidgets++;
}

static void DrawWidgets(glwindow_t *glw)
{
  int i;

  // Clear the old scene
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glLoadIdentity();
  glPushMatrix();
  glScalef(((float)glw->width)/glw->pwidth,((float)glw->height)/glw->pheight,1.0f);

  // Draw all the widgets
  for(i=0; i<nWidgets; i++) {
    if( Widgets[i].draw ) {
      glPushMatrix();
      glTranslatef(Widgets[i].x, Widgets[i].y+1, 1.0f);
      glScalef(Widgets[i].w,Widgets[i].h,0.0f);
      (*Widgets[i].draw)(&Widgets[i]);
      glPopMatrix();
    }
  } 
  glPopMatrix();

  // Swap buffers to draw to screen
  glXSwapBuffers(glw->dpy, glw->win);
}

static void LayoutWidgets(glwindow_t *glw)
{
  static char mode_str[] = "Render: Voxels";
  static frame_gui_t gf;
  static button_gui_t    bhelp  = {"About",       0, 0, &(gf.help)};
  static button_gui_t    bquit  = {"Quit",        0, 0, NULL};
  static button_gui_t    bmode  = {mode_str,      0, 0, &(gf.mode)};
  static button_gui_t    baxes  = {"Axes",        0, 1, &(gf.axes)};
  static button_gui_t    barot  = {"Auto Rotate", 0, 1, &(gf.arot)};
  static button_gui_t    bboxes = {"Bound Boxes", 0, 1, &(gf.boxes)};
  static button_gui_t    blig   = {"Ligand",      0, 1, &(gf.lig)};
  static button_gui_t    bprot  = {"Protein",     0, 1, &(gf.prot)};
  static button_gui_t    bH     = {"H",           0, 1, &(gf.H)};
  static button_gui_t    bC     = {"C",           0, 1, &(gf.C)};
  static button_gui_t    bN     = {"N",           0, 1, &(gf.N)};
  static button_gui_t    bO     = {"O",           0, 1, &(gf.O)};
  static button_gui_t    bother = {"Other",       0, 1, &(gf.other)};
  static stats_gui_t     stats;

  // Main view frame
  memset(&gf,0,sizeof(gf));
  gf.axes   = 1;
  gf.arot   = 1;
  gf.boxes  = 1;
  gf.lig    = 1;
  gf.prot   = 1;
  gf.H = gf.C = gf.N = gf.O = gf.other = 1;
  gf.text = "gf";
  AddWidget(glw, 8, 8, 768-16, 768-16, 
	    Frame_Draw, 
	    NULL, 
	    Frame_MouseDown, Frame_MouseUp, Frame_MouseMove,
	    NULL,
	    &gf);

  // Status / info widget
  AddWidget(glw, 768, (768/2)+8, 128-8, (768/2)-16, Stats_Draw, NULL, NULL, NULL, NULL, NULL, &stats);

  // Main Buttons
  AddWidget(glw, 768, 8+(24+8)*0,  128-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL, &bhelp);
  AddWidget(glw, 768, 8+(24+8)*1,  128-8,  24, Button_Draw,    NULL, Quit_Down,   NULL, NULL, NULL, &bquit);
  AddWidget(glw, 768, 8+(24+8)*2,  128-8,  24, Button_Draw,    NULL, Mode_Down,   NULL, NULL, NULL, &bmode);
  AddWidget(glw, 768, 8+(24+8)*3,  128-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL, &baxes);
  AddWidget(glw, 768, 8+(24+8)*4,  128-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL, &bboxes);
  AddWidget(glw, 768, 8+(24+8)*5,  128-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL, &barot);

  // Ligand and protein filter buttons.
  AddWidget(glw, 768, 8+(24+8)*7,  128-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL, &blig);
  AddWidget(glw, 768, 8+(24+8)*8,  128-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL, &bprot);

  // Atom channel filter buttons
  AddWidget(glw, 768,    8+(24+8)*9,  64-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL,  &bH);
  AddWidget(glw, 768+64, 8+(24+8)*9,  64-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL,  &bC);
  AddWidget(glw, 768,    8+(24+8)*10, 64-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL,  &bN);
  AddWidget(glw, 768+64, 8+(24+8)*10, 64-8,  24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL,  &bO);
  AddWidget(glw, 768,    8+(24+8)*11, 128-8, 24, Button_Draw,    NULL, Toggle_Down, NULL, NULL, NULL,  &bother);
}

static void HandleEvent(glwindow_t *glw)
{ 
  XEvent xe;
  KeySym key;
  char   keys[255];
  int    i;

  while( XPending(glw->dpy) ) {
    XNextEvent(glw->dpy, &xe);
    switch(xe.type) {
      // WindowManager and similar
    case ConfigureNotify:
      if ((xe.xconfigure.width != glw->width) || (xe.xconfigure.height != glw->height)) {
	glw->width = xe.xconfigure.width;
	glw->height = xe.xconfigure.height;
	ViewPort2D(glw);
	Redraw = 1;
      }
      break;
    case ClientMessage:    
      if (*XGetAtomName(glw->dpy, xe.xclient.message_type) == *"WM_PROTOCOLS")
	Done = 1;
      break;
    case Expose:
      if ( !xe.xexpose.count )
	Redraw = 1;
      break;
      // Input events
    case ButtonPress:
      for(i=0; i<nWidgets; i++) {
	if( Widgets[i].mousedown )
	  (*Widgets[i].mousedown)(&Widgets[i],xe.xbutton.x, xe.xbutton.y, xe.xbutton.button);
	
      }
      break;
    case ButtonRelease:
      for(i=0; i<nWidgets; i++) {
	if( Widgets[i].mouseup )
	  (*Widgets[i].mouseup)(&Widgets[i],xe.xbutton.x, xe.xbutton.y, xe.xbutton.button);
      }
      break;
    case MotionNotify:
      GuiState.hand_pos.s.x = xe.xbutton.x;
      GuiState.hand_pos.s.y = xe.xbutton.y;
      for(i=0; i<nWidgets; i++) {
	if( Widgets[i].mousemove )
	  (*Widgets[i].mousemove)(&Widgets[i],xe.xbutton.x, xe.xbutton.y);
      }
      break;
    case KeyPress:
      if( XLookupString(&xe.xkey, keys, 255, &key, 0) == 1 ) {
	for(i=0; i<nWidgets; i++) {
	  if( Widgets[i].keypress )
	    (*Widgets[i].keypress)(&Widgets[i],*keys,0);
	} 
      } else {
	for(i=0; i<nWidgets; i++) {
	  if( Widgets[i].keypress )
	    (*Widgets[i].keypress)(&Widgets[i],0,xe.xkey.keycode);
	} 
      }
      break;
    }
  }
}

static void *EventHandler(void *arg)
{
  glwindow_t glw;
  char       title[256];
  timeval_t  slice, now, end, period;
  fd_set     in_fds, out_fds;
  int        x11_fd,i;
  
  // Initialize the window
  sprintf(title, "%s", Version);
  glw.pwidth  = 768+128+1;
  glw.pheight = 768+1;
  StartUp(&glw,title,glw.pwidth,glw.pheight);
  x11_fd = ConnectionNumber(glw.dpy);

  // Add widgets
  LayoutWidgets(&glw);

  // Setup timer
  period.tv_usec = 10000;
  period.tv_sec  = 0;
  slice.tv_sec   = 0;
  slice.tv_usec  = 0;
  gettimeofday(&now, 0);
  timeradd(&now, &period, &end);

  while( !Done ) {
    // Create a file descriptor set containing x11_fd
    FD_ZERO(&in_fds);
    FD_SET(x11_fd, &in_fds);
    FD_ZERO(&out_fds);

    // Wait for X Event or a Timer
    if( select(x11_fd+1, &in_fds, &out_fds, 0, &slice) > 0 ) {
      ////////////////////////////////////////////////////////////
      // File descriptor event:  Process X event
      HandleEvent(&glw);

      // Check to see if the any widget(s) need a redraw
      for(i=0; i<nWidgets; i++) {
	if( Widgets[i].update ) {
	  Widgets[i].update = 0;
	  Redraw = 1;
	}
      }
      if( Redraw == 1 ) {
	DrawWidgets(&glw);
        Redraw = 0;
      }

      // Set the slice up for time remaining
      gettimeofday(&now, 0);
      timersub(&end, &now, &slice);
    } else {
      ////////////////////////////////////////////////////////////
      // Timer fired

      // Check for any "missed" X events
      HandleEvent(&glw);

      // Give each widget a "tick"
      for(i=0; i<nWidgets; i++) {
	if( Widgets[i].tick ) {
	  Widgets[i].tick(&Widgets[i]);
	}
      }

      // Force a full update
      DrawWidgets(&glw);
      Redraw = 0;

      // Setup for a new whole slice
      slice.tv_sec  = period.tv_sec;
      slice.tv_usec = period.tv_usec;
      gettimeofday(&now, 0);
      timeradd(&now, &period, &end);
    }
  }

  // Finished, call our exit/quit function.
  GuiExit();
  return NULL;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void UpdateGuiState(gstate_t *s)
{
  pthread_mutex_lock(&StateLock);
  // Copy state-level vars
  if( Statec ) {
    // !!av: copy stuffs hack alert!!!
    memcpy(Statec,s,sizeof(gstate_t));
  }
  pthread_mutex_unlock(&StateLock);

  Redraw = 1;
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

void InitGUI(char *version, gstate_t *s)
{
  // Set the version string (passing this makes the makefile cleaner)
  Version = version;
 
  // Malloc two gstate structures: current/last and picked
  if( !(Stateg=Statec=malloc(sizeof(gstate_t))) )
    Error("GUI: malloc(gstate_t) failed!");
  if( !(Statep=malloc(sizeof(gstate_t))) )
    Error("GUI: malloc(gstate_t) failed!");
  
  UpdateGuiState(s);

  GuiState.mouse_item_ndx = -1;
}

int StartGUI(char *version, gstate_t *s)
{
  pthread_t t;  pthread_attr_t a;

  // Initialize first
  InitGUI(version, s);
  
  // Set up the thread attributes
  pthread_attr_init(&a);
  pthread_attr_setdetachstate(&a,PTHREAD_CREATE_DETACHED);
  pthread_attr_setscope(&a,PTHREAD_SCOPE_SYSTEM);

  // Create thread
  if( pthread_create(&t, &a, EventHandler, NULL) ) 
    return 0;
  
  // Return success
  return 1;
}

#endif // GUI_C
