#include "glm/glm.hpp"
#ifndef __TYPES_H__
#define __TYPES_H__

typedef glm::vec3 Point;

/* Color struct */
typedef struct {
   float r;
   float g;
   float b;
   float f; // "filter" or "alpha"
} color_t;

/*typedef struct {
   float x, y, z;
} Point;
*/
typedef struct {
   Point center;
   float radius;
   color_t ambient, diffuse, specular;
} Sphere;

typedef struct {
   Point center, normal, topleft, bottomright;
   color_t ambient, diffuse, specular;
} Plane;

typedef struct {
   Point eye, lookAt, lookUp, lookRight;
  float theta_x, theta_y;
} Camera;

typedef struct {
   Point position;
   color_t ambient, diffuse, specular;
} PointLight;

typedef struct {
   Point origin, direction;
} Ray;

#endif
