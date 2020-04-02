#ifndef ROTATION_H
#define ROTATION_H

#include "matrix.h"


////////////////////////////////////////////////////////////
// Interface:
////////////////////////////////////////////////////////////
#ifndef ROTATION_C
matrix4_t* get_rotation_matricies  (int *n);  // Returns pointer to array of rotation matricies, sets n.
void       print_rotation_matricies();        // Print matricies to stdout.
#endif //!ROTATION_C


#endif
