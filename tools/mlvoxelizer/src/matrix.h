#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

// 3x3
typedef struct st_matrix3_t {
  double a[3][3];
} matrix3_t;

// 4x4
typedef struct st_matrix4_t {
  double a[4][4];
} matrix4_t;

////////////////////////////////////////////////////////////
// Interface:
////////////////////////////////////////////////////////////
#ifndef MATRIX_C
extern matrix3_t* matrix3_add_scalar (matrix3_t *m1, matrix3_t *m2, const double s);
extern matrix3_t* matrix3_mult_scalar(matrix3_t *m1, matrix3_t *m2, const double s);
extern vector3_t* matrix3_mult_vector(matrix3_t *m1, vector3_t *v1, vector3_t *v2);
extern matrix3_t* matrix3_identity   (matrix3_t *m1);
extern matrix3_t* matrix3_transpose  (matrix3_t *m1, matrix3_t *m2);

extern matrix4_t* matrix4_add_scalar (matrix4_t *m1, matrix4_t *m2, const double s);
extern matrix4_t* matrix4_mult_scalar(matrix4_t *m1, matrix4_t *m2, const double s);
extern vector4_t* matrix4_mult_vector(matrix4_t *m1, vector4_t *v1, vector4_t *v2);
extern matrix4_t* matrix4_identity   (matrix4_t *m1);
extern matrix4_t* matrix4_transpose  (matrix4_t *m1, matrix4_t *m2);
extern matrix4_t* matrix4_rotation   (double angle, vector3_t *axis, matrix4_t *m1);
#endif //!MATRIX_C

#endif
