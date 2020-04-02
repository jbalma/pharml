#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <sys/types.h>
#include <string.h>
#include <sys/time.h>

#include "types.h"
#define UTIL_C
#include "util.h"

// wallclock time
u64b_t get_time()
{
  struct timeval tv;

  gettimeofday(&tv, NULL);

  return tv.tv_sec * 1000000ull + tv.tv_usec;
}

// fprintf(stderr); exit();
void Error(const char *fmt, ...)
{
  va_list ap;

  if(!fmt) exit(1);
  va_start(ap, fmt);  
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fflush(stderr);
  _Exit(1);
}

// fprintf(stderr);
void Warn(const char *fmt, ...)
{
  va_list ap;

  if(!fmt) return;
  va_start(ap, fmt);  
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fflush(stderr);
}
