#ifndef IO_BITMAP_H
#define IO_BITMAP_H


#include "types.h"


typedef struct {
  u32b_t  w;      // Width
  u32b_t  h;      // Height
  u8b_t  *d;      // Pixel data
} io_bitmap_t;


////////////////////////////////////////////////////////////
// Interface:
////////////////////////////////////////////////////////////
#ifndef BITMAP_C
extern void io_bitmap_load(char *fn, io_bitmap_t *bmp);
extern void io_bitmap_free(io_bitmap_t *bmp);
#endif //!BITMAP_C


#endif
