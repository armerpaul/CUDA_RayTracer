#ifndef __TYPES_H__
#define __TYPES_H__

/* Color struct */
typedef struct {
   double r;
   double g;
   double b;
   double f; // "filter" or "alpha"
} color_t;

typedef struct {
   double x, y, z;
} Point;

typedef struct {
   Point* center;
   double radius;
   color_t *ambient, *diffuse, *specular;
} Sphere;

typedef struct {
   Point *center, *normal;
   color_t *ambient, *diffuse, *specular;
} Plane;

typedef struct {
   Point *eye, *lookAt, *lookUp;
   Point *u, *v, *w;
} Camera;

typedef struct {
   Point *position;
   color_t *ambient, *diffuse, *specular;
} PointLight;

typedef struct {
   Point *origin, *direction;
} Ray;

#endif
