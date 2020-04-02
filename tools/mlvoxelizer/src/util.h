#ifndef UTIL_H
#define UTIL_H

#include "types.h"

////////////////////////////////////////////////////////////
// Interface:
////////////////////////////////////////////////////////////
#ifndef UTIL_C
extern void   Error(const char *fmt, ...);
extern void   Warn(const char *fmt, ...);

extern u64b_t get_time();
#endif


#endif
