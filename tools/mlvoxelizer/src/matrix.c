#ifndef MATRIX_C
#define MATRIX_C

#include <math.h>
#include "matrix.h"

////////////////////////////////////////////////////////////

//
// m2 = m1 + s
//
matrix3_t* matrix3_add_scalar(matrix3_t *m1, matrix3_t *m2, const double s)
{
  int i,j;

  for(i=0; i<3; i++) {
    for(j=0; j<3; j++) {
      m2->a[i][j] = m1->a[i][j] + s;
    }
  }

  return m2;
}

//
// m2 = m1 * s
//
matrix3_t* matrix3_mult_scalar(matrix3_t *m1, matrix3_t *m2, const double s)
{
  int i,j;

  for(i=0; i<3; i++) {
    for(j=0; j<3; j++) {
      m2->a[i][j] = m1->a[i][j] * s;
    }
  }

  return m2;
}

//
// v2 = m1 * v1
//
vector3_t* matrix3_mult_vector(matrix3_t *m1, vector3_t *v1, vector3_t *v2)
{
  vector3_t t;
  double    s;
  int       i,j;

  for(i=0; i<3; i++) {
    s = 0.0;
    for(j=0; j<3; j++) {
      s += m1->a[i][j] * v1->a[j];
    }
    t.a[i] = s;
  }

  vector3_copy(&t,v2);
  return v2;
}

//
// m1 = I
//
matrix3_t* matrix3_identity(matrix3_t *m1)
{
  int i,j;

  for(i=0; i<3; i++) {
    for(j=0; j<3; j++) {
      m1->a[i][j] = 0.0;
    }
  }
  for(i=0; i<3; i++) {
    m1->a[i][i] = 1.0;
  }

  return m1;
}

//
// m2 = m1^T
//
matrix3_t* matrix3_transpose(matrix3_t *m1, matrix3_t *m2)
{
  int i,j;

  for(i=0; i<3; i++) {
    for(j=0; j<3; j++) {
      m2->a[i][j] =  m1->a[j][i];
    }
  }

  return m2;
}

////////////////////////////////////////////////////////////

//
// m2 = m1 + s
//
matrix4_t* matrix4_add_scalar(matrix4_t *m1, matrix4_t *m2, const double s)
{
  int i,j;

  for(i=0; i<4; i++) {
    for(j=0; j<4; j++) {
      m2->a[i][j] = m1->a[i][j] + s;
    }
  }

  return m2;
}

//
// m2 = m1 * s
//
matrix4_t* matrix4_mult_scalar(matrix4_t *m1, matrix4_t *m2, const double s)
{
  int i,j;

  for(i=0; i<4; i++) {
    for(j=0; j<4; j++) {
      m2->a[i][j] = m1->a[i][j] * s;
    }
  }

  return m2;
}

//
// v2 = m1 * v1
//
vector4_t* matrix4_mult_vector(matrix4_t *m1, vector4_t *v1, vector4_t *v2)
{
  vector4_t t;
  double    s;
  int       i,j;

  for(i=0; i<4; i++) {
    s = 0.0;
    for(j=0; j<4; j++) {
      s += m1->a[i][j] * v1->a[j];
    }
    t.a[i] = s;
  }

  vector4_copy(&t,v2);
  return v2;
}

//
// m1 = I
//
matrix4_t* matrix4_identity(matrix4_t *m1)
{
  int i,j;

  for(i=0; i<4; i++) {
    for(j=0; j<4; j++) {
      m1->a[i][j] = 0.0;
    }
  }
  for(i=0; i<4; i++) {
    m1->a[i][i] = 1.0;
  }

  return m1;
}

//
// m2 = m1^T
//
matrix4_t* matrix4_transpose(matrix4_t *m1, matrix4_t *m2)
{
  int i,j;

  for(i=0; i<4; i++) {
    for(j=0; j<4; j++) {
      m2->a[i][j] =  m1->a[j][i];
    }
  }

  return m2;
}

//
// m1 = rotate(angle, axis)
//
matrix4_t* matrix4_rotation(double angle, vector3_t *axis, matrix4_t *m1)
{
  vector3_t a;
  double    r  = angle * (M_PI/180.0);
  double    c  = cos(r);
  double    s  = sin(r);
  double    t  = 1.0 - c;

  vector3_normalize(axis,&a);
  
  double    tx = t*a.s.x;
  double    ty = t*a.s.y;
  double    tz = t*a.s.z;
  double    sz = s*a.s.z;
  double    sy = s*a.s.y;
  double    sx = s*a.s.x;

  matrix4_identity(m1);

  m1->a[0][0] = tx * a.s.x + c;
  m1->a[0][1] = tx * a.s.y + sz;
  m1->a[0][2] = tx * a.s.z - sy;

  m1->a[1][0] = tx * a.s.y - sz;
  m1->a[1][1] = ty * a.s.y + c;
  m1->a[1][2] = ty * a.s.z + sx;

  m1->a[2][0] = tx * a.s.z + sy;
  m1->a[2][1] = ty * a.s.z - sx;
  m1->a[2][2] = tz * a.s.z + c;

  return m1;
}

////////////////////////////////////////////////////////////

#endif
