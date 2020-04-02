#ifndef VECTOR_H
#define VECTOR_H

typedef union {
  // Coords available as structure fields
  struct {
    double x;
    double y;
    double z;
  } s;
  // Coords available as an array
  double a[3];
} vector3_t;


typedef union {
  // Coords available as structure fields
  struct {
    double x;
    double y;
    double z;
    double w;
  } s;
  // Coords available as an array
  double a[4];
} vector4_t;


////////////////////////////////////////////////////////////
// Interface:
////////////////////////////////////////////////////////////
#ifndef VECTOR_C
extern int        vector3_compare        (vector3_t *v1, vector3_t *v2);
extern double     vector3_point_line_dist(vector3_t *x,  vector3_t *l1, vector3_t *l2);
extern double     vector3_length         (vector3_t *v);
extern double     vector3_dotprod        (vector3_t *v1, vector3_t *v2);
extern vector3_t* vector3_crossprod      (vector3_t *v1, vector3_t *v2, vector3_t *v3);
extern vector3_t* vector3_mult_scalar    (vector3_t *v1, vector3_t *v2, const double s);
extern vector3_t* vector3_add_scalar     (vector3_t *v1, vector3_t *v2, const double s);
extern vector3_t* vector3_add_vector     (vector3_t *v1, vector3_t *v2, vector3_t *v3);
extern vector3_t* vector3_sub_vector     (vector3_t *v1, vector3_t *v2, vector3_t *v3);
extern vector3_t* vector3_normalize      (vector3_t *v1, vector3_t *v2);
extern vector3_t* vector3_copy           (vector3_t *v1, vector3_t *v2);


extern vector4_t* vector4_mult_scalar(vector4_t *v1, vector4_t *v2, const double s);
extern vector4_t* vector4_add_scalar (vector4_t *v1, vector4_t *v2, const double s);
extern vector4_t* vector4_add_vector (vector4_t *v1, vector4_t *v2, vector4_t *v3);
extern vector4_t* vector4_sub_vector (vector4_t *v1, vector4_t *v2, vector4_t *v3);
extern vector4_t* vector4_copy       (vector4_t *v1, vector4_t *v2);
#endif //!VECTOR_C

#endif
