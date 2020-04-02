#ifndef GUI_3DVIEW_C
#define GUI_3DVIEW_C

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
#define GUI_WIDGET
#include "gui.h"
#undef GUI_WIDGET
#include "gui_3dview.h"
#include "cray_logo.h"
#include "io_bitmap.h"
#include "main.h"
#include "voxelizer.h"

////////////////////////////////////////////////////////////////////////////////

#define VOX_TH (0.9)
#define TEX_TH (0.05)

////////////////////////////////////////////////////////////////////////////////

static int LoadCrayLogo()
{
  u32b_t tex;
  int    i,pixel[3];
  char  *data = cl_header_data;
  char  *rdata;

  // Read the bitmap data from the header file.
  if( !(rdata = malloc(cl_width*cl_height*3)) ) {
    Error("LoadCrayLogo(): Failed to allocate space for raw bitmap data.\n");
  }
  for(i=0; i<cl_width*cl_height; i++) {
    CL_HEADER_PIXEL(data,pixel);
    rdata[i*3+0] = pixel[0];
    rdata[i*3+1] = pixel[1];
    rdata[i*3+2] = pixel[2];
  }
  
  // Turn the bmp into an OpenGL texture with linear filtering.
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, cl_width, cl_height, 0, GL_RGB, GL_UNSIGNED_BYTE, rdata);
  
  // Free the bmp pixel data
  free(rdata);
  
  // Return the OpenGL texture handle
  return tex;
}

static void DrawLoadingScreen(widget_t *w)
{
  char  buf[1024];

  // Draw a background to render text on.
  glPushMatrix();
  glTranslatef(0.5f-0.3f/2.0f, 0.5f-0.1f/2.0f, 0.0f);
  glScalef(0.3f, 0.1f, 1.0f);
  glColor4f(0.0f, 0.0f, 0.0f, 0.8f);
  glBegin(GL_QUADS);
  glVertex2f(0.0f,0.0f);
  glVertex2f(0.0f,1.0f);
  glVertex2f(1.0f,1.0f);
  glVertex2f(1.0f,0.0f);
  glEnd();
  Yellow();
  glBegin(GL_LINE_LOOP);
  glVertex2f(0.0f,0.0f);
  glVertex2f(0.0f,1.0f);
  glVertex2f(1.0f,1.0f);
  glVertex2f(1.0f,0.0f);
  glEnd();
  glPopMatrix();
  
  // Give a text description of what's going on.
  Yellow();
  sprintf(buf," Processing..");
  glRasterPos2f(0.5f-((strlen(buf)*6.0f/2.0f)/ScaleX(w,w->w)), 0.5f+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
}

static void DrawHelpScreen(widget_t *w)
{
  char  buf[1024];
  float a=0.0f, b=1.0f;
  int   l, glxM, glxm;
  
  // Build Cray logo if needed.
  static int clogo = -1;
  if( clogo == -1 ) {
    clogo = LoadCrayLogo();
  }

  // Draw the logo and header.
  glPushMatrix();
  glTranslatef(0.0f,0.0f,1.0f);
  glEnable(GL_TEXTURE_2D);
  glColor3f(1.0f,1.0f,1.0f);
  glBindTexture(GL_TEXTURE_2D,clogo);
  glBegin(GL_QUADS);
  glTexCoord2f(a,b); glVertex2f(0.0f, 0.2f/ScaleY(w,1.0f));
  glTexCoord2f(a,a); glVertex2f(0.0f, 0.0f);
  glTexCoord2f(b,a); glVertex2f(0.2f/ScaleX(w,1.0f), 0.0f);
  glTexCoord2f(b,b); glVertex2f(0.2f/ScaleX(w,1.0f), 0.2f/ScaleY(w,1.0f));
  glEnd();
  glDisable(GL_TEXTURE_2D);
  glPopMatrix();
  Yellow();
  sprintf(buf,"MLVoxelizer %s -- (C) Cray Inc. 2018",VERSION);
  glRasterPos2f(0.21f/ScaleX(w,1.0f), 0.1f/ScaleY(w,1.0f)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);

  // Draw the help text.
  l = 0;
  Yellow();
  sprintf(buf,"Controls:");
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
  White();
  sprintf(buf,"Rotate     --  Right-Click + Drag");
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
  sprintf(buf,"Translate  --  Left-Click  + Drag");
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
  sprintf(buf,"Zoom       --  Mouse Wheel");
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;

  // Draw some version information.
  l++;
  Yellow();
  sprintf(buf,"Versions:");
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
  White();
  sprintf(buf,"MLVoxelizer  --  %s",VERSION);
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
  sprintf(buf,"FFTW         --  %s",fftw_version_string());
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
  glXQueryVersion(w->glw->dpy, &glxM, &glxm);
  sprintf(buf,"GLX          --  %d.%d",glxM,glxm);
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
  sprintf(buf,"OpenGL       --  %s",glGetString(GL_VERSION));
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
  glxM = XProtocolVersion(w->glw->dpy);
  glxm = XProtocolRevision(w->glw->dpy);
  sprintf(buf,"XWindows     --  %d.%d %s %d",glxM,glxm,
	  XServerVendor(w->glw->dpy),XVendorRelease(w->glw->dpy));
  glRasterPos2f(0.02f, 0.2f/ScaleY(w,1.0f)+0.025f*(1+l)+4.0f/ScaleY(w,w->h));
  printGLf(w->glw->font,"%s",buf);
  l++;
}

////////////////////////////////////////////////////////////////////////////////

void Project(vector3_t *outv)
{
  GLint    viewport[4];
  GLdouble modelview[16];
  GLdouble projection[16];
  GLdouble winX, winY, winZ;
  GLdouble posX, posY, posZ;

  posX=0.0;
  posY=0.0;
  posZ=0.0;

  glGetDoublev(  GL_MODELVIEW_MATRIX, modelview );
  glGetDoublev(  GL_PROJECTION_MATRIX, projection );
  glGetIntegerv( GL_VIEWPORT, viewport );

  gluProject(posX,posY,posZ,modelview,projection,viewport,&winX,&winY,&winZ);

  outv->s.x = winX;
  outv->s.y = winY;
  outv->s.z = winZ;
}

////////////////////////////////////////////////////////////////////////////////

static void DrawBox(float w, float h, float d, int t)
{
  glPushMatrix();
  glScalef(w, h, d);

  // Top face
  glBegin(t);
  glNormal3f(0.0f, 1.0f, 0.0f);
  glVertex3f(0.5f, 0.5f, 0.5f);
  glVertex3f(0.5f, 0.5f, -0.5f);
  glVertex3f(-0.5f, 0.5f, -0.5f);
  glVertex3f(-0.5f, 0.5f, 0.5f);
  glEnd();
  
  // front face
  glBegin(t);
  glNormal3f(0.0f, 0.0f, 1.0f);
  glVertex3f(0.5f, 0.5f, 0.5f);
  glVertex3f(-0.5f, 0.5f, 0.5f);
  glVertex3f(-0.5f, -0.5f, 0.5f);
  glVertex3f(0.5f, -0.5f, 0.5f);
  glEnd();
  
  // right face
  glBegin(t);
  glNormal3f(1.0f, 0.0f, 0.0f);
  glVertex3f(0.5f, 0.5f, 0.5f);
  glVertex3f(0.5f, -0.5f, 0.5f);
  glVertex3f(0.5f, -0.5f, -0.5f);
  glVertex3f(0.5f, 0.5f, -0.5f);
  glEnd();
  
  // left face
  glBegin(t);
  glNormal3f(-1.0f, 0.0f, 0.0f);
  glVertex3f(-0.5f, 0.5f, 0.5f);
  glVertex3f(-0.5f, 0.5f, -0.5f);
  glVertex3f(-0.5f, -0.5f, -0.5f);
  glVertex3f(-0.5f, -0.5f, 0.5f);
  glEnd();
  
  // bottom face
  glBegin(t);
  glNormal3f(0.0f, -1.0f, 0.0f);
  glVertex3f(0.5f, -0.5f, 0.5f);
  glVertex3f(-0.5f, -0.5f, 0.5f);
  glVertex3f(-0.5f, -0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, -0.5f);
  glEnd();
  
  // back face
  glBegin(t);
  glNormal3f(0.0f, 0.0f, -1.0f);
  glVertex3f(0.5f, 0.5f, -0.5f);
  glVertex3f(0.5f, -0.5f, -0.5f);
  glVertex3f(-0.5f, -0.5f, -0.5f);
  glVertex3f(-0.5f, 0.5f, -0.5f);
  glEnd();
  glPopMatrix();
} 

static void DrawAtom(widget_t *w, char type, float x, float y, float z, float r, int hl)
{
  float     black[4]={0.0f,0.0f,0.0f,0.0f};
  float     color[4],ar;
  vector3_t cv;
  int       slices=10,stacks=10;

  if( Statec->gui == 2 ) {
    // Lower quality.
    slices = stacks = 6;
  }

  static int         warn_uk=0;
  static int         init=1;
  static GLUquadric *qdrc;

  if( init ) {
    // Init if needed
    init = 0;
    // For drawing 3D "primitives"
    qdrc=gluNewQuadric();
  }

  // Get atom's visual properties.
  if( (ar=atom_radius(type)) == -1.0 ) {
    if( !warn_uk ) {
      Warn("DrawAtom(): Unknown atom type: '%c'.\n",type);
      warn_uk = 1;
    }
    return;
  }
  if( atom_color(type, &cv) == -1 ) {
    Error("DrawAtom(): Atom type has radius but not color?!?\n");
  }
  color[0] = cv.a[0];
  color[1] = cv.a[1];
  color[2] = cv.a[2];
  color[3] = 1.0f;

  // Draw the atom.
  glMateriali(GL_FRONT_AND_BACK,  GL_SHININESS, 128);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  color);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   color);
  if( hl ) {
    // Highlight.
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, color);
  } else {
    // No highlight.
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
  }
  glPushMatrix();
  glTranslatef(x,y,z);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  // Draw atom as a sphere.
  gluSphere(qdrc, 2*r*ar, slices, stacks);
  glDisable(GL_CULL_FACE);
  glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
  glPopMatrix();
}

////////////////////////////////////////////////////////////////////////////////

static void DrawInference(widget_t *w)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;
  int i;
  float x,y,z;

  // Enabled?
  if( !(gf->lig) ) {
    return;
  }

  // Draw ligand without fog.
  glDisable(GL_FOG);

  // Draw all "bind" predictions.
  for(i=0; i<Statec->ninference; i++) {
    x = Statec->inference[i].s.x;
    x = (x/((double)Statec->res)) - (VOXELIZER_GLOBAL_SIZE/2.0);
    x = (Statec->vx/2.0+x*Statec->res) / Statec->vx;
    y = Statec->inference[i].s.y;
    y = (y/((double)Statec->res)) - (VOXELIZER_GLOBAL_SIZE/2.0);
    y = (Statec->vy/2.0+y*Statec->res) / Statec->vy;
    z = Statec->inference[i].s.z;
    z = (z/((double)Statec->res)) - (VOXELIZER_GLOBAL_SIZE/2.0);
    z = (Statec->vz/2.0+z*Statec->res) / Statec->vz;
    if( Statec->inference[i].s.w > 0.5 ) {
      DrawAtom(w, 'L', x, y, z, 0.005, 1);
    } else {
      DrawAtom(w, 'O', x, y, z, 0.005, 1);
    }
  }

  if( Statec->gui != 2 ) {
    // Simple gui uses no fog.
    glEnable(GL_FOG);
  }
}

static void DrawLigand(widget_t *w)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;
  int i;

  // Enabled?
  if( !(gf->lig) ) {
    return;
  }

  // Draw ligand without fog.
  glDisable(GL_FOG);

  // Draw all atoms.
  for(i=0; i<Statec->nligand; i++) {
    DrawAtom(w, 
             Statec->ligand[i].type, 
             (Statec->vx/2.0+Statec->ligand[i].pos.s.x*Statec->res) / Statec->vx,
             (Statec->vy/2.0+Statec->ligand[i].pos.s.y*Statec->res) / Statec->vy,
             (Statec->vz/2.0+Statec->ligand[i].pos.s.z*Statec->res) / Statec->vz,
             0.005,
             1);
  }

  // Draw bonds/edges between the ligand atoms.
  if( Statec->gui != 2 ) {
    // Not low quality, so use smoothing.
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  }
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  for(i=0; i<Statec->nbonds; i++) {
    // Now draw bonds with lines.
    switch( Statec->bonds[i][2] ) {
    case 1:
      glLineWidth(2.0f);
      Cyan();
      break;
    case 2:
      glLineWidth(3.0f);
      Green();
      break;
    case 3:
      glLineWidth(4.0f);
      Red();
      break;
    case 4:
      glLineWidth(3.0f);
      Blue();
      break;
    }
    glBegin(GL_LINES);
    glVertex3f((Statec->vx/2.0+Statec->ligand[Statec->bonds[i][0]].pos.s.x*Statec->res) / Statec->vx,
               (Statec->vy/2.0+Statec->ligand[Statec->bonds[i][0]].pos.s.y*Statec->res) / Statec->vy,
               (Statec->vz/2.0+Statec->ligand[Statec->bonds[i][0]].pos.s.z*Statec->res) / Statec->vz);
    glVertex3f((Statec->vx/2.0+Statec->ligand[Statec->bonds[i][1]].pos.s.x*Statec->res) / Statec->vx,
               (Statec->vy/2.0+Statec->ligand[Statec->bonds[i][1]].pos.s.y*Statec->res) / Statec->vy,
               (Statec->vz/2.0+Statec->ligand[Statec->bonds[i][1]].pos.s.z*Statec->res) / Statec->vz);
    glEnd();
    glLineWidth(1.0f);  
  }
  glEnable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_LINE_SMOOTH);
  if( Statec->gui != 2 ) {
    // Simple gui uses no fog.
    glEnable(GL_FOG);
  }
}

static int DrawAtoms(widget_t *w, int skip)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;
  int i,c;

  static int f=1,dlO,dlN,dlC,dlH,dlother,*dl;

  // Enabled?
  if( !(gf->prot) ) {
    return skip;
  }

  // This "skip" is for a loading screen in case it takes
  // a long time to build the initial display lists.
  if( f && !skip ) {
    skip = 1;
  } else if( f && skip ) {
    skip = 0;
    f    = 0;
    // One list per channel.
    for(c=0; c<Statec->chnls; c++) {
      switch(atom_channels()[c]) {
      case 'O':
        dl = &dlO;
        break;
      case 'N':
        dl = &dlN;
        break;
      case 'C':
        dl = &dlC;
        break;
      case 'H':
        dl = &dlH;
        break;
      default:
        dl = &dlother;
        break;
      }
      *dl = glGenLists(1);
      glNewList(*dl, GL_COMPILE);
      for(i=0; i<Statec->natoms; i++) {
        if( Statec->atoms[i].type == atom_channels()[c] ) {
          DrawAtom(w, 
                   Statec->atoms[i].type, 
                   (Statec->vx/2.0+Statec->atoms[i].pos.s.x*Statec->res) / Statec->vx,
                   (Statec->vy/2.0+Statec->atoms[i].pos.s.y*Statec->res) / Statec->vy,
                   (Statec->vz/2.0+Statec->atoms[i].pos.s.z*Statec->res) / Statec->vz,
                   0.005,
                   0);
        }
      }
      glEndList();
    }
  }

  // Draw all atoms.
  if( !f ) {
    for(c=0; c<Statec->chnls; c++) {
      switch(atom_channels()[c]) {
      case 'O':
	if( !gf->O ) { continue; }
	dl = &dlO;
	break;
      case 'N':
	if( !gf->N ) { continue; }
	dl = &dlN;
	break;
      case 'C':
	if( !gf->C ) { continue; }
	dl = &dlC;
	break;
      case 'H':
	if( !gf->H ) { continue; }
	dl = &dlH;
	break;
      default:
	if( !gf->other ) { continue; }
	dl = &dlother;
	break;
      }
      glCallList(*dl);
    }
  }

  return skip;
}

static void DrawVoxel(widget_t *w, float x, float y, float z, float r, vector3_t *v)
{
  float     black[4]={0.0f,0.0f,0.0f,0.0f};
  float     color[4];

  // Set color.
  color[0] = v->a[0];
  color[1] = v->a[1];
  color[2] = v->a[2];
  color[3] = 1.0f;

  // Draw the voxel.
  glMateriali(GL_FRONT_AND_BACK,  GL_SHININESS, 128);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  color);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   color);
  glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION,  black);
  glPushMatrix();
  glTranslatef(x,y,z);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  // Draw voxel as a box.
  DrawBox(r,r,r,GL_POLYGON);
  glDisable(GL_CULL_FACE);
  glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, black);
  glPopMatrix();
}

static int DrawVoxels(widget_t *w, int skip)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;
  int       j,k,x,y,z,c,e,ge,v,f,t;
  vector3_t cv,gcv;
  float     cw;

  static int dl=-1,flags=-1;

  // Enabled?
  if( !(gf->prot) ) {
    return skip;
  }

  f = (gf->O << 4) | (gf->N << 3) | (gf->C << 2) | (gf->H << 1) | gf->other;

  // If any display flags have changed, rebuild the display list.
  if( (f != flags) && (!skip) ) {
    skip = 1;
  } else if( (f != flags) && (skip) ) {
    skip = 0;
    flags = f;
    if( dl != -1 ) {
      glDeleteLists(dl, 1);
    }
    dl = glGenLists(1);
    glNewList(dl, GL_COMPILE);
    // Draw all voxels.
    for(z=0; z<Statec->vz; z++) {
      for(y=0; y<Statec->vy; y++) {
	for(x=0; x<Statec->vx; x++) {
	  v  = 0;
	  ge = 0;
	  cw = 0.0;
	  gcv.s.x = gcv.s.y = gcv.s.z = 0.0;
	  // Consider all atom channels.
	  for(c=0; c<Statec->chnls; c++) {
	    // See if pixel is valid (passes threshold) and should be rendered.
            j = c*(Statec->vx*Statec->vy*Statec->vz) + 
                z*(Statec->vx*Statec->vy) +
                y*(Statec->vx) +
                x;
	    if( atom_color(atom_channels()[c], &cv) == -1 ) {
	      Error("DrawVoxels(): Unknown atom type.\n");
	    }
	    cw += Statec->voxels[j];
	    gcv.s.x += Statec->voxels[j] * cv.s.x;
	    gcv.s.y += Statec->voxels[j] * cv.s.y;
	    gcv.s.z += Statec->voxels[j] * cv.s.z;
	    switch(atom_channels()[c]) {
	    case 'O':
	      t = gf->O;
	      break;
	    case 'N':
	      t = gf->N;
	      break;
	    case 'C':
	      t = gf->C;
	      break;
	    case 'H':
	      t = gf->H;
	      break;
	    default:
	      t = gf->other;
	      break;
	    }
	    if( !t ) {
	      continue;
	    }
	    if( Statec->voxels[j] > VOX_TH ) {
	      v = 1;
	    }
            // Skip internal voxels.
	    k = c*(Statec->vx*Statec->vy*Statec->vz) + 
		  (z+0)*(Statec->vx*Statec->vy) +
                  (y+0)*(Statec->vx) +
		  (x-1);
	    e = 1 && ( (k >= 0) && (k < Statec->chnls*(Statec->vx*Statec->vy*Statec->vz)) && (Statec->voxels[k] > VOX_TH) );
	    k = c*(Statec->vx*Statec->vy*Statec->vz) + 
		  (z+0)*(Statec->vx*Statec->vy) +
                  (y+0)*(Statec->vx) +
		  (x+1);
	    e = e && ( (k >= 0) && (k < Statec->chnls*(Statec->vx*Statec->vy*Statec->vz)) && (Statec->voxels[k] > VOX_TH) );
	    k = c*(Statec->vx*Statec->vy*Statec->vz) + 
		  (z+0)*(Statec->vx*Statec->vy) +
                  (y-1)*(Statec->vx) +
		  (x+0);
	    e = e && ( (k >= 0) && (k < Statec->chnls*(Statec->vx*Statec->vy*Statec->vz)) && (Statec->voxels[k] > VOX_TH) );
	    k = c*(Statec->vx*Statec->vy*Statec->vz) + 
		  (z+0)*(Statec->vx*Statec->vy) +
                  (y+1)*(Statec->vx) +
		  (x+0);
	    e = e && ( (k >= 0) && (k < Statec->chnls*(Statec->vx*Statec->vy*Statec->vz)) && (Statec->voxels[k] > VOX_TH) );
	    k = c*(Statec->vx*Statec->vy*Statec->vz) + 
		  (z-1)*(Statec->vx*Statec->vy) +
                  (y+0)*(Statec->vx) +
		  (x+0);
	    e = e && ( (k >= 0) && (k < Statec->chnls*(Statec->vx*Statec->vy*Statec->vz)) && (Statec->voxels[k] > VOX_TH) );
	    k = c*(Statec->vx*Statec->vy*Statec->vz) + 
		  (z+1)*(Statec->vx*Statec->vy) +
                  (y+0)*(Statec->vx) +
		  (x+0);
	    e = e && ( (k >= 0) && (k < Statec->chnls*(Statec->vx*Statec->vy*Statec->vz)) && (Statec->voxels[k] > VOX_TH) );
	    ge = ge | e;
	  }
	  // Draw if valid (threshold) and not enclosed on all sided by others.
	  if( v && !ge ) {
	    gcv.s.x /= cw;
	    gcv.s.y /= cw;
	    gcv.s.z /= cw;
	    DrawVoxel(w,
		      ((float)x) / Statec->vx,
		      ((float)y) / Statec->vy,
		      ((float)z) / Statec->vz,
		      1.0/Statec->vz,
		      &gcv);
	  }
	}
      }
    }
    glEndList();
  }

  // Draw everything.
  if( dl != -1 ) {
    glCallList(dl);
  }

  return skip;
}

static int DrawVolumetric(widget_t *w, int skip)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;
  int        i,j,c,x,y,z,f,t;
  vector3_t  cv,gcv;
  float      cw,*data;

  static int dl=-1,flags=-1;
  static int tex_id = -1;

  // Enabled?
  if( !(gf->prot) ) {
    return skip;
  }

  f = (gf->O << 4) | (gf->N << 3) | (gf->C << 2) | (gf->H << 1) | gf->other;

  // If any display flags have changed, rebuild the display list.
  if( (f != flags) && (!skip) ) {
    skip = 1;
  } else if( (f != flags) && (skip) ) {
    skip  = 0;
    flags = f;
    // Get space to hold the 3D texture.
    if( !(data = malloc(Statec->vx*Statec->vy*Statec->vz*4*sizeof(float))) ) {
      Error("DrawVolumetric(): Allocate 3D texture for voxels failed (%d).\n",
	    Statec->vx*Statec->vy*Statec->vz*4*sizeof(float));
    }
    memset(data,0,Statec->vx*Statec->vy*Statec->vz*4*sizeof(float));
    // Fill in the texture from the voxels.
    for(z=0; z<Statec->vz; z++) {
      for(y=0; y<Statec->vy; y++) {
	for(x=0; x<Statec->vx; x++) {
	  cw = 0.0;
	  gcv.s.x = gcv.s.y = gcv.s.z = 0.0;
	  // Get a color for this voxel.
	  for(c=0; c<Statec->chnls; c++) {
	    if( atom_color(atom_channels()[c], &cv) == -1 ) {
	      Error("(): Unknown atom type.\n");
	    }
	    switch(atom_channels()[c]) {
	    case 'O':
	      t = gf->O;
	      break;
	    case 'N':
	      t = gf->N;
	      break;
	    case 'C':
	      t = gf->C;
	      break;
	    case 'H':
	      t = gf->H;
	      break;
	    default:
	      t = gf->other;
	      break;
	    }
	    if( t ) {
	      j = c*(Statec->vx*Statec->vy*Statec->vz) + 
                z*(Statec->vx*Statec->vy) +
                y*(Statec->vx) +
                x;
	      cw      += Statec->voxels[j];
	      gcv.s.x += Statec->voxels[j] * cv.s.x;
	      gcv.s.y += Statec->voxels[j] * cv.s.y;
	      gcv.s.z += Statec->voxels[j] * cv.s.z;
	    }
	  }
	  gcv.s.x /= cw;
	  gcv.s.y /= cw;
	  gcv.s.z /= cw;
	  // Store this into the data array for the texture.
	  j = z*(Statec->vx*Statec->vy) +
	      y*(Statec->vx) +
	      x;	  
	  j *= 4;
	  data[j+0] = gcv.s.x;
	  data[j+1] = gcv.s.y;
	  data[j+2] = gcv.s.z;
	  data[j+3] = cw / Statec->chnls;
	  if( data[j+0] > 1.0 ) { data[j+0] = 1.0; }
	  if( data[j+1] > 1.0 ) { data[j+1] = 1.0; }
	  if( data[j+2] > 1.0 ) { data[j+2] = 1.0; }
	  if( data[j+3] > 1.0 ) { data[j+3] = 1.0; }
	}
      }
    }
    // Now that the texture data array is filled in, turn it into a GL texture.
    if( tex_id != -1 ) {
      glDeleteTextures(1,(GLuint*)&tex_id);
    }
    glGenTextures(1,(GLuint*)&tex_id);
    glBindTexture( GL_TEXTURE_3D, tex_id );
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    if( Statec->gui == 2 ) {
      // Simple gui uses simpler nearest filtering.
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    } else {
      // Default to tri-linear filtering.
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
      glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, Statec->vx, Statec->vy, Statec->vz, 0, GL_RGBA, GL_FLOAT, (GLvoid*)data);
    glBindTexture(GL_TEXTURE_3D, 0);
    // We're done with the texture data (it's in the GPU now).
    free(data);
    // Now we bind the texture and draw the 2(.5D) slices.
    if( dl != -1 ) {
      glDeleteLists(dl, 1);
    }
    dl = glGenLists(1);
    glNewList(dl, GL_COMPILE);
    glEnable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, tex_id);
    White();
    glBegin(GL_QUADS);
    for(i=0; i<Statec->vz; i++) {
      glTexCoord3f(0.0f, 0.0f, ((float)i)/(Statec->vz-1));  glVertex3f(0.0f, 0.0f, ((float)i)/(Statec->vz-1));
      glTexCoord3f(1.0f, 0.0f, ((float)i)/(Statec->vz-1));  glVertex3f(1.0f, 0.0f, ((float)i)/(Statec->vz-1));
      glTexCoord3f(1.0f, 1.0f, ((float)i)/(Statec->vz-1));  glVertex3f(1.0f, 1.0f, ((float)i)/(Statec->vz-1));
      glTexCoord3f(0.0f, 1.0f, ((float)i)/(Statec->vz-1));  glVertex3f(0.0f, 1.0f, ((float)i)/(Statec->vz-1));
    }
    glEnd();
    // We're done drawing, so unroll any needed modes, etc.
    glBindTexture(GL_TEXTURE_3D, 0);
    glDisable(GL_TEXTURE_3D);
    glEndList();
  }

  // Draw everything.
  if( dl != -1 ) {
    // Move to same position as the default 3D spot, but with no rotation.
    glPushMatrix();
    glLoadIdentity();
    glTranslatef(gf->trnx, gf->trny, gf->zoom);
    glTranslatef(-0.5f, -0.5f, -0.5f);
    // No lighting, fog, etc. for this.
    glDisable(GL_FOG);
    glDisable(GL_LIGHTING);
    // The alpha test will clip off "empty" pixels.
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, TEX_TH);
    // Blending will make them translucent when drawn back to front.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // The rotation is applied to the 3D texture coordinates.
    glEnable(GL_TEXTURE_3D);
    glMatrixMode(GL_TEXTURE);
    glPushMatrix();
    glLoadIdentity();
    glTranslatef( 0.5f, 0.5f, 0.5f );
    glRotatef(-gf->roty, 0.0f, 1.0f, 0.0f);
    glRotatef(-gf->rotx + 5.0f, 1.0f, 0.0f, 0.0f);
    glTranslatef( -0.5f,-0.5f, -0.5f );
    // Render the slices.
    glCallList(dl);
    // We're done drawing, so unroll any needed modes, etc.
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);    
    glPopMatrix();
    glDisable(GL_ALPHA_TEST);
    glEnable(GL_LIGHTING);
    glDisable(GL_TEXTURE_3D);
    if( Statec->gui != 2 ) {
      // Simple gui uses no fog.
      glEnable(GL_FOG);
    }
  }

  return skip;
}


////////////////////////////////////////////////////////////

static void DrawAxes(widget_t *w)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;

  glDisable(GL_FOG);
  if( gf->axes ) {
    if( Statec->gui != 2 ) {
      // Not low quality, so use smoothing.
      glEnable(GL_LINE_SMOOTH);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    }
    glDisable(GL_LIGHTING);
    glPushMatrix();
    glTranslatef(0.5, 0.5, 0.5);
    // Axes.
    glBegin(GL_LINES);
    Blue();
    glVertex3f(0.0f,  0.0f, -0.5f);
    glVertex3f(0.0f,  0.0f,  0.5f);
    Green();
    glVertex3f(0.0f, -0.5f,  0.0f);
    glVertex3f(0.0f,  0.5f,  0.0f);
    Red();
    glVertex3f(-0.5f, 0.0f,  0.0f);
    glVertex3f( 0.5f, 0.0f,  0.0f);
    glEnd();
    glDisable(GL_DEPTH_TEST);
    Red();   glRasterPos3f(-0.5, -0.025,  0.0);  printGLf(w->glw->font,"%s","-x");
    Red();   glRasterPos3f(+0.5, -0.025,  0.0);  printGLf(w->glw->font,"%s","+x");
    Green(); glRasterPos3f( 0.0, -0.525,  0.0);  printGLf(w->glw->font,"%s","-y");
    Green(); glRasterPos3f( 0.0, +0.525,  0.0);  printGLf(w->glw->font,"%s","+y");
    Blue();  glRasterPos3f( 0.0, -0.025, -0.5);  printGLf(w->glw->font,"%s","-z");
    Blue();  glRasterPos3f( 0.0, -0.025, +0.5);  printGLf(w->glw->font,"%s","+z");
    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glEnable(GL_LIGHTING);
    glDisable(GL_LINE_SMOOTH);
  }
  if( Statec->gui != 2 ) {
    // Simple gui uses no fog.
    glEnable(GL_FOG);
  }
}

static void DrawBoundingBoxes(widget_t *w)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;

  glDisable(GL_FOG);
  if( gf->boxes ) {
    if( Statec->gui != 2 ) {
      // Not low quality, so use smoothing.
      glEnable(GL_LINE_SMOOTH);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    }
    glDisable(GL_LIGHTING);
    glPushMatrix();
    glTranslatef(0.5, 0.5, 0.5);
    // Draw protein bounding box.
    if( gf->prot ) {
      Purple();  
      DrawBox((Statec->vmax.s.x-Statec->vmin.s.x)/(Statec->vx),
	      (Statec->vmax.s.y-Statec->vmin.s.y)/(Statec->vy),
	      (Statec->vmax.s.z-Statec->vmin.s.z)/(Statec->vz),
	      GL_LINE_LOOP);
    }
    // Draw ligand bounding box.
    if( gf->lig ) {
      Cyan();
      glPushMatrix();
      glTranslatef((Statec->lmin.s.x+(Statec->lmax.s.x-Statec->lmin.s.x)/2.0f)*(Statec->res)/(Statec->vx),
		   (Statec->lmin.s.y+(Statec->lmax.s.y-Statec->lmin.s.y)/2.0f)*(Statec->res)/(Statec->vy),
		   (Statec->lmin.s.z+(Statec->lmax.s.z-Statec->lmin.s.z)/2.0f)*(Statec->res)/(Statec->vz));
      glDisable(GL_DEPTH_TEST);
      DrawBox((Statec->lmax.s.x-Statec->lmin.s.x)*(Statec->res)/(Statec->vx),
	      (Statec->lmax.s.y-Statec->lmin.s.y)*(Statec->res)/(Statec->vy),
	      (Statec->lmax.s.z-Statec->lmin.s.z)*(Statec->res)/(Statec->vz),
	      GL_LINE_LOOP);
      glEnable(GL_DEPTH_TEST);
      glPopMatrix();
    }
    glPopMatrix();
    glEnable(GL_LIGHTING);
    glDisable(GL_LINE_SMOOTH);
  }
  if( Statec->gui != 2 ) {
    // Simple gui uses no fog.
    glEnable(GL_FOG);
  }
}

////////////////////////////////////////////////////////////

void Frame_Draw(widget_t *w)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;
  static int skip = 0;

  GLfloat light_position[] = {  5.0f,  5.0f,  5.0f, 1.0f };
  GLfloat specular_color[] = {  1.0f,  1.0f,  1.0f, 0.1f };
  GLfloat diffuse_color[]  = {  1.0f,  1.0f,  1.0f, 1.0f };
  GLfloat ambient_color[]  = { 0.05f, 0.05f, 0.05f, 1.0f };

  // Set auto-rotate.
  if( gf->arot && (gf->md != MOUSE_RIGHT) ) {
    gf->roty += 5.0f / 3.1415f / 6.0f;
  }
  if( gf->rotx < -180.0f ) {
    gf->rotx = -180.0f;
  }
  if( gf->rotx > 180.0f ) {
    gf->rotx = 180.0f;
  }

  // Add some fog to help make depth more clear.
  GLfloat density = 0.75;
  GLfloat fogColor[4] = {0.0, 0.0, 0.0, 1.0}; 
  if( Statec->gui != 2 ) {
    // Simple gui uses no fog.
    glEnable(GL_FOG);
  }
  glFogi(GL_FOG_MODE, GL_LINEAR);
  glFogfv(GL_FOG_COLOR, fogColor);
  glFogf(GL_FOG_DENSITY, density);
  glHint(GL_FOG_HINT, GL_NICEST);
  glFogf(GL_FOG_START, gf->zoom+0.25);
  glFogf(GL_FOG_END,   gf->zoom-0.45f);

  // Save 2D state so we can restore it later
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glMatrixMode(GL_MODELVIEW);

  // Switch to 3D perspective
  ViewPort3D(w->x, w->y, ScaleX(w, w->w), ScaleY(w, w->h));
  glMatrixMode(GL_MODELVIEW);

  // Setup lighting
  glShadeModel(GL_SMOOTH);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position );
  glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient_color  );
  glLightfv(GL_LIGHT0, GL_SPECULAR, specular_color );
  glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse_color  );
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_DEPTH_TEST);

  // Set "board" positioning / rotation
  glLoadIdentity();
  glTranslatef(gf->trnx, gf->trny, gf->zoom);
  glRotatef(gf->rotx + 5.0f, 1.0f, 0.0f, 0.0f);
  glRotatef(gf->roty, 0.0f, 1.0f, 0.0f);
  glTranslatef(-0.5f, -0.5f, -0.5f);

  // Draw the main 3D content (protein/ligand).
  if( !gf->help ) {
    // Draw the scene in the correct mode/order.
    switch( gf->mode ) {
    case 0:
      // Protein in voxel form (cubes).
      skip = DrawVoxels(w,skip);
      // Draw the ligand with atom spheres and bond lines.
      DrawLigand(w);
      DrawInference(w);
      // Draw coordinate axes.
      DrawAxes(w);
      // Draw protein and ligand bounding boxes.
      DrawBoundingBoxes(w);
      break;
    case 1:
      // Protein in atom form (spheres).
      skip = DrawAtoms(w,skip);
      // Draw the ligand with atom spheres and bond lines.
      DrawLigand(w);
      DrawInference(w);
      // Draw coordinate axes.
      DrawAxes(w);
      // Draw protein and ligand bounding boxes.
      DrawBoundingBoxes(w);
      break;
    case 2:
      // Draw coordinate axes.
      DrawAxes(w);
      // Draw protein and ligand bounding boxes.
      DrawBoundingBoxes(w);
      // Draw the ligand with atom spheres and bond lines.
      DrawLigand(w);
      DrawInference(w);
      // Protein in volumetric form (cloud).
      skip = DrawVolumetric(w,skip);
      break;
    }
  }

  // Disable lighting
  glDisable(GL_LIGHTING);
  glDisable(GL_LIGHT0);

  // Disable fog.
  glDisable(GL_FOG);

  // Restore 2D mode to draw 2D stuffs
  ViewPort2D(w->glw);
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);

  if( gf->help ) {
    // Draw the help / about screen if needed.
    DrawHelpScreen(w);
  } else {
    // Draw logo in place of everything else if needed.
    if( skip ) {
      DrawLoadingScreen(w);
    }
  }

  // Outline
  Yellow();
  glBegin(GL_LINE_LOOP);
  glVertex2f(0.0f,0.0f);
  glVertex2f(0.0f,1.0f);
  glVertex2f(1.0f,1.0f);
  glVertex2f(1.0f,0.0f);
  glEnd();
}

////////////////////////////////////////////////////////////////////////////////

void Frame_MouseDown(widget_t *w, int x, int y, int b)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;

  // Check bounds
  if( (x >= ScaleX(w, w->x)) && (x <= ScaleX(w, w->x+w->w)) && 
      (y >= ScaleY(w, w->y)) && (y <= ScaleY(w, w->y+w->h))    ) {
    switch(b) {
    case MOUSE_UP:
      // Zoom out
      gf->zoom -= .1;
      if( gf->zoom < 0 ) {
        gf->zoom = 0;
      }
      break;
    case MOUSE_DOWN:
      // Zoom in
      gf->zoom += .1;
      if( gf->zoom > 2 ) {
        gf->zoom = 2;
      }
      break;
    case MOUSE_LEFT:
      // Record that the left mouse is down
      gf->md  = b;
      gf->mdx = x;
      gf->mdy = y;
      break;
    case MOUSE_RIGHT:
      // Record that the right mouse is down
      gf->md  = b;
      gf->mdx = x;
      gf->mdy = y;
      break;
    }
  }
}

void Frame_MouseUp(widget_t *w, int x, int y, int b)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;

  // Check bounds.
  if( x < 0 ) { x = 0; }
  if( y < 0 ) { y = 0; }

  if( gf->md == MOUSE_RIGHT ) {
    // Apply distance moved to the rotation
    gf->rotx += (((float)y)-gf->mdy) / 3.1415f / 2.0f;
    gf->roty += (((float)x)-gf->mdx) / 3.1415f / 6.0f;
    gf->mdx = x;
    gf->mdy = y;
  }
  if( gf->md == MOUSE_LEFT ) {
    // Apply distance moved to the translation
    gf->trnx += (((float)x)-gf->mdx)*0.001;
    gf->trny -= (((float)y)-gf->mdy)*0.001;
    gf->mdx = x;
    gf->mdy = y;
  }

  // Record that the mouse is now up
  if( gf->md == b ) {
    gf->md = 0;
  }
}

void Frame_MouseMove(widget_t *w, int x, int y)
{
  frame_gui_t *gf = (frame_gui_t*)w->wd;

  // Check bounds.
  if( x < 0 ) { x = 0; }
  if( y < 0 ) { y = 0; }

  if( gf->md == MOUSE_RIGHT ) {
    // Apply distance moved to the rotation
    gf->rotx += (((float)y)-gf->mdy) / 3.1415f;
    gf->roty += (((float)x)-gf->mdx) / 3.1415f;
    gf->mdx = x;
    gf->mdy = y;
  }
  if( gf->md == MOUSE_LEFT ) {
    // Apply distance moved to the translation
    gf->trnx += (((float)x)-gf->mdx)*0.001;
    gf->trny -= (((float)y)-gf->mdy)*0.001;
    gf->mdx = x;
    gf->mdy = y;
  }
}

////////////////////////////////////////////////////////////////////////////////

#endif // !GUI_3DVIEW_C
