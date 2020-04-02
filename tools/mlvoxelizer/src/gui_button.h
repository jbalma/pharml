#ifndef GUI_BUTTON_H
#define GUI_BUTTON_H

////////////////////////////////////////////////////////////
// For all files

#include "types.h"

typedef struct {
  char   *text;     // Text drawn on button
  u32b_t   val;     // Button value
  u32b_t   sel;     // 1 == selected, 0 == not selected
  u32b_t *link;     // A pointer link to an int controlled by this button
} button_gui_t;

////////////////////////////////////////////////////////////
// For files other than gui_button.c
#ifndef GUI_BUTTON_C
extern void Button_Draw(widget_t *w);
#endif

#endif // !GUI_BUTTON_H
