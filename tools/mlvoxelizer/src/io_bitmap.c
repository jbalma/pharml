#ifndef IO_BITMAP_C
#define IO_BITMAP_C


#include <stdlib.h>
#include <string.h>
#include <stdio.h>


#include "types.h"
#include "util.h"
#include "io_bitmap.h"


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////


//
// Frees the pixel data associated with the bitmap bmp
//
void io_bitmap_free(io_bitmap_t *bmp)
{
  free(bmp->d);
}


//
// Loads a 32-bit bitmap from file fn into bitmap bmp 
//
void io_bitmap_load(char *fn, io_bitmap_t *bmp)
{
  FILE   *f;  
  u32b_t  data_offset,i;  
  u16b_t  type,planes,bpp;
  u8b_t   t;

  
  // Open the file 
  if( !(f=fopen(fn, "rb")) ) {
    Error("io_bitmap_load(): Failed to open bitmap file (%s).\n",fn);
  }

  // Check for bitmap format
  if( !fread(&type, sizeof(u16b_t), 1, f) ) {
    Error("io_bitmap_load(): Error reading bitmap magic number (%s).\n",fn);
  }
  if (type != 19778) {
    Error("io_bitmap_load(): Incorrect magic number found; Cowardly refusing to continue (%s).\n",fn);
  }        

  // Read position of actual bitmap data
  fseek(f, 8, SEEK_CUR);
  if ( !fread(&data_offset, sizeof(u32b_t), 1, f) ) {
    Error("io_bitmap_load(): Failed to read pixel data offset (%s).\n",fn);
  }

  // Read the width and height of the bitmap
  fseek(f, 4, SEEK_CUR);
  if( !fread(&bmp->w, sizeof(u32b_t), 1, f) ) {
    Error("io_bitmap_load(): Error reading bitmap width (%s).\n",fn);
  }
  if( !fread(&bmp->h, sizeof(u32b_t), 1, f) ) {
    Error("io_bitmap_load(): Error reading bitmap height (%s).\n",fn);
  }

  // Verify that the number of planes is 1
  if( !fread(&planes, sizeof(u16b_t), 1, f) ) {
    Error("io_bitmap_load(): Error reading number of planes (%s).\n",fn);
  }
  if (planes != 1) {
    Error("io_bitmap_load(): Unsupported number of planes (%u!=1)(%s).\n",planes,fn);
  }

  // Read the number of bits per pixel
  if (!fread(&bpp, sizeof(u16b_t), 1, f)) {
    Error("io_bitmap_load(): Error reading bits per pixel (%s).\n",fn);
  }
  if (bpp != 24) {
    Error("io_bitmap_load(): Unsupported bits per pixel (%u!=24)(%s).\n",bpp,fn);
  }

  // Allocate space for and read pixel data
  if( !(bmp->d = malloc(bmp->w*bmp->h*3*sizeof(u8b_t))) ) {
    Error("io_bitmap_load(): Error allocating space for pixel data (%ub)(%s).\n",bmp->w*bmp->h*3,fn);
  }
  fseek(f, data_offset, SEEK_SET);
  if (!fread(bmp->d, bmp->w*bmp->h*3, 1, f)) {
    Error("io_bitmap_load(): Error reading pixel data ()%s.\n",fn);
  }

  // Swap red and blue (bgr -> rgb)
  for (i = 0; i < bmp->w*bmp->h*3; i += 3) {
    t           = bmp->d[i];
    bmp->d[i]   = bmp->d[i+2];
    bmp->d[i+2] = t;
  } 
  
  // Close the file
  fclose(f);
}


#endif
