#ifndef GUI_STATS_H
#define GUI_STATS_H

////////////////////////////////////////////////////////////
// For all files

#include "types.h"

typedef struct {
  int foo;  // For animation or whatever
} stats_gui_t;

////////////////////////////////////////////////////////////
// For files other than gui_stats.c
#ifndef GUI_STATS_C
extern void Stats_Draw(widget_t *w);
#endif

#endif // !GUI_STATS_H
