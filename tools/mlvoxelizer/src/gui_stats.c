#ifndef GUI_STATS_C
#define GUI_STATS_C

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
#include "gui_stats.h"

////////////////////////////////////////////////////////////////////////////////

void Stats_Draw(widget_t *w)
{
  //stats_gui_t *gf = (stats_gui_t*)w->wd;
  char buf[1024];
  int  l=1;

  // Protein / Ligand
  Yellow();
  sprintf(buf,"PDB:");
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  White();
  sprintf(buf,"%s",Statec->pfn);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  if( Statec->smiles ) {
    sprintf(buf,"SMILES");
  } else if( Statec->lfn == NULL ) {
    sprintf(buf,"%s",Statec->pfn);
  } else {
    sprintf(buf,"%s",Statec->lfn);
  }

  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);

  sprintf(buf,"P_sz:  %d",Statec->natoms);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  sprintf(buf,"%.0fx%.0fx%.0f A",
	  Statec->amax.s.x-Statec->amin.s.x,
	  Statec->amax.s.y-Statec->amin.s.y,
	  Statec->amax.s.z-Statec->amin.s.z);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  sprintf(buf,"L_sz:  %d",Statec->nligand);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  sprintf(buf,"%.1fx%.1fx%.1f",
	  Statec->lmax.s.x-Statec->lmin.s.x,
	  Statec->lmax.s.y-Statec->lmin.s.y,
	  Statec->lmax.s.z-Statec->lmin.s.z);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  sprintf(buf,"Bonds: %d",Statec->nbonds);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  l++;
  
  // Voxel stats
  Green();
  sprintf(buf,"Voxels:");
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  White();
  sprintf(buf,"Res:    %.1f A",Statec->res);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  sprintf(buf,"Wins:   %d",Statec->nsb);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  sprintf(buf,"%.1fx%.1fx%.1f",
	  Statec->win_w, Statec->win_h, Statec->win_d);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  sprintf(buf,"Stride: %0.1f A",Statec->stride);
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);
  if( Statec->valid == -1 ) {
    sprintf(buf,"Valid:  ligand");
  } else {
    sprintf(buf,"Valid:  %0.1f",Statec->valid);
  }
  glRasterPos2f(0.075f, l*0.06f); l++;
  printGLf(w->glw->font,"%s",buf);


  // Outline
  Yellow();
  glBegin(GL_LINE_LOOP);
  glVertex2f(0.0f,0.0f);
  glVertex2f(0.0f,1.0f);
  glVertex2f(1.0f,1.0f);
  glVertex2f(1.0f,0.0f);
  glEnd();
}

#endif // !GUI_STATS_C
