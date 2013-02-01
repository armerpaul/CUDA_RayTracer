/*
  CPE 471 Lab 1 
  Base code for Rasterizer
  Example code using B. Somers' image code - writes out a sample tga
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <glm/glm.hpp>
#include <math.h>
#include "Image.h"
#include "types.h"
#include "VanExLib.h"

__constant__ Sphere s[NUM_SPHERES];
__constant__ Plane * f;
__constant__ PointLight * l;
__constant__ Camera *cam;

Camera* CameraInit();
PointLight* LightInit();
Sphere* CreateSpheres();
Point* CreatePoint(double x, double y, double z);
color_t* CreateColor(double r, double g, double b);

__global__ void CUDARayTrace(color_t * pixelList);

__device__ color_t RayTrace(Ray* r, Sphere* s, Plane* f, PointLight* l);
__device__ color_t SphereShading(int sNdx, Ray* r, Point* p, Sphere* sphereList, PointLight* l);
__device__ double SphereRayIntersection(Sphere* s, Ray* r);
__device__ double dot(Point p1, Point p2);
__device__ Point subtractPoints(Point p1, Point p2);
__device__ Point normalize(Point p);

int main(void) 
{
   // set up for random num generator
   srand ( time(NULL) );
   Image img(WINDOW_WIDTH, WINDOW_HEIGHT);
   Camera* camera = CameraInit();
   PointLight* light = LightInit();
	 color_t * pixel_device = NULL;
   double aspectRatio = WINDOW_WIDTH; 
	 aspectRatio /= WINDOW_HEIGHT;
  
  	//SCENE SET UP
  	// (floor)
   Plane* floor = new Plane();
   //floor->center = CreatePoint(0, -1 * WINDOW_HEIGHT / 2, -1 * WINDOW_WIDTH / 2);
   //floor->color = CreateColor(200, 200, 200);
   //floor->normal = CreatePoint(0, 0, -1 * WINDOW_WIDTH / 2);
   // (spheres)
   Sphere* spheres = CreateSpheres();

   cudaMemcpyToSymbol(f, floor, sizeof(Plane));
   cudaMemcpyToSymbol(l, light, sizeof(PointLight));
   cudaMemcpyToSymbol(cam, camera, sizeof(Camera));
   cudaMemcpyToSymbol(s, spheres, sizeof(Sphere)*NUM_SPHERES);
   
   cudaMemcpy(pixel_device, new color_t[WINDOW_WIDTH * WINDOW_HEIGHT], WINDOW_WIDTH * WINDOW_HEIGHT, cudaMemcpyHostToDevice);
   //MEMCPY'S
   
   CUDARayTrace<<< (WINDOW_WIDTH * WINDOW_HEIGHT + 1023) / 1024, 1024 >>>(pixel_device);
   
   for (int i=0; i < WINDOW_WIDTH; i++) {
		for (int j=0; j < WINDOW_HEIGHT; j++) {
         //Looping over the Rays
     		img.pixel(i, j, pixel_device[j*WINDOW_WIDTH + i]);
		    }
  	}
  	
	// IMAGE OUTPUT
	//
  	// write the targa file to disk
  	img.WriteTga((char *)"vanilla.tga", true); 
  	// true to scale to max color, false to clamp to 1.0
} 
Camera* CameraInit() {
   
   Camera* temp = new Camera();
   temp->eye = CreatePoint(0, 0, 0);
   temp->lookAt = CreatePoint(0, 0, SCREEN_DISTANCE);
   temp->lookUp = CreatePoint(0, 1, 0);

   temp->u = CreatePoint(1, 0, 0);
   temp->v = CreatePoint(0, 1, 0);
   temp->w = CreatePoint(0, 0, 1);
   
   return temp;
}

PointLight* LightInit() {
   PointLight* temp = new PointLight();

   temp->ambient = CreateColor(0.2, 0.2, 0.2);
   temp->diffuse = CreateColor(0.6, 0.6, 0.6);
   temp->specular = CreateColor(0.9, 0.9, 0.9);

   //temp->position = CreatePoint(0, 200, 0);
   temp->position = CreatePoint(-1 * WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, -1 * WINDOW_WIDTH / 2);

   return temp;
}

Point* CreatePoint(double x, double y, double z) {
   Point* p = new Point();
   
   p->x = x;
   p->y = y;
   p->z = z;
   
   return p;
}

color_t* CreateColor(double r, double g, double b) {
   color_t* c = new color_t();

   c->r = r;
   c->g = g;
   c->b = b;
   c->f = 1.0;

   return c;
}

Sphere* CreateSpheres() {
	Sphere* spheres = new Sphere[NUM_SPHERES]();
   
   	for (int i=0; i<NUM_SPHERES; i++) {
	   	spheres[i].radius = rand() % 25 + 75;
	   	spheres[i].center = CreatePoint(175 * i - 200, 
	   									-1 * WINDOW_HEIGHT / 2 + 2 * spheres[i].radius,
	   									pow(-15, i) - 500);
	   	spheres[i].ambient = CreateColor(i * .3, .5 + i * .2, 1 - i * .3);
	   	spheres[i].diffuse = CreateColor(i * .3, .5 + i * .2, 1 - i * .3);
	   	spheres[i].specular = CreateColor(1., 1., 1.);
	   	
	   	//fprintf(stderr, "radius = %lf\n", spheres[i].radius);
	   	//fprintf(stderr, "r = %lf, g = %lf, b = %lf\n", spheres[i].color.r, spheres[i].color.g, spheres[i].color.b);
	   	//fprintf(stderr, "x = %lf, y = %lf, z = %lf\n\n", spheres[i].center.x, spheres[i].center.y, spheres[i].center.z);
	}
	
	return spheres;
}

__global__ void CUDARayTrace(color_t * pixelList)
{
    double tanVal = tan(FOV/2);
    double aspectRatio = WINDOW_WIDTH / WINDOW_HEIGHT;
    int row = (blockIdx.x *blockDim.x + threadIdx.x ) / WINDOW_WIDTH;
    int col = (blockIdx.x *blockDim.x + threadIdx.x ) % WINDOW_WIDTH;
    color_t returnColor;
    Ray * r = new Ray();
   
    //INIT RAY VALUES
	  r->origin = cam->eye;
    r->direction = cam->lookAt;
    r->direction->x = -1 * aspectRatio * tanVal + (2 * tanVal / WINDOW_HEIGHT) * col;
    r->direction->y = tanVal - (2 * tanVal / WINDOW_HEIGHT) * row;
    
    returnColor = RayTrace(r, s, f, l);

    int index = row *WINDOW_WIDTH + col;
    pixelList[index].r = returnColor.r;
    pixelList[index].g = returnColor.g;
    pixelList[index].b = returnColor.b;
    pixelList[index].f = returnColor.f;
 
}
__device__ color_t RayTrace(Ray* r, Sphere* s, Plane* f, PointLight* l) {
    color_t* black = CreateColor(0, 0, 0); 
    double t, smallest;
   	Point* p;
   	int i = 0, closestSphere = -1;

    
    while (i < NUM_SPHERES) {
		t = SphereRayIntersection(s + i, r);

		if (t > 0 && (t < smallest || closestSphere < 0)) {
			smallest = t;
			closestSphere = i;
		}
      	i++;
   }

   if (closestSphere > -1) {

   		//fprintf(stderr, "r = %lf, g = %lf, b = %lf\n", s[closestSphere].color.r, s[closestSphere].color.g, s[closestSphere].color.b);
      	p = CreatePoint(r->direction->x * smallest, r->direction->y * smallest, r->direction->z * smallest);
      	return SphereShading(closestSphere, r, p, s, l);
   }
   
   return *black;
}

__device__ double SphereRayIntersection(Sphere* s, Ray* r) {
	double a, b, c, d, t1, t2;
    
    a = dot(*(r->direction), *(r->direction));
    b = dot(subtractPoints(*(r->origin), *(s->center)), *(r->direction));
    c = dot(subtractPoints(*(r->origin), *(s->center)), subtractPoints(*(r->origin), *(s->center)))
            - (s->radius * s->radius);
    d = (b * b) - (a * c);
    

    if (d >= 0) {
		//fprintf(stderr, "a = %lf, b = %lf, c = %lf\n", a, b, c);
        
		t1 = (-1 * b - sqrt(d)) / a;
		t2 = (-1 * b + sqrt(d)) / a;
	
		if (t2 > t1 && t2 > 0) {
			//fprintf(stderr, "d = %lf, t2 = %lf\n\n", d, t2);
			return t2;
		} else if (t1 > 0) {
			//fprintf(stderr, "d = %lf, t1 = %lf\n\n", d, t1);
			return t1;
		}
	}
	return d;
}

__device__ color_t SphereShading(int sNdx, Ray* r, Point* p, Sphere* sphereList, PointLight* l) {
	color_t a, d, s, total;
	double reflectTemp, NdotL, RdotV;
	Point viewVector, lightVector, reflectVector, normalVector;

   //printf("r->%lf g->%lf b->%lf\n", l->ambient->r, l->ambient->g, l->ambient->b);
   //printf("r->%lf g->%lf b->%lf\n", l->diffuse->r, l->diffuse->g, l->diffuse->b);
   //printf("r->%lf g->%lf b->%lf\n\n", l->specular->r, l->specular->g, l->specular->b);

	viewVector = normalize(subtractPoints(*(r->origin), *p));
	lightVector = normalize(subtractPoints(*p, *(l->position)));
	normalVector = normalize(subtractPoints(*p, *(sphereList[sNdx].center)));
	reflectVector = subtractPoints(normalVector, lightVector);

   NdotL = dot(lightVector, normalVector);

	reflectTemp = 2 * NdotL;
	reflectVector.x *= reflectTemp;
	reflectVector.y *= reflectTemp;
	reflectVector.z *= reflectTemp;
	
	a.r = l->ambient->r * sphereList[sNdx].ambient->r;
	a.g = l->ambient->g * sphereList[sNdx].ambient->g;
	a.b = l->ambient->b * sphereList[sNdx].ambient->b;

   if (NdotL > 0. ) {

      //printf("%lf\n", NdotL);
      //printf("%lf %lf %lf\n", sphereList[sNdx].diffuse->r, sphereList[sNdx].diffuse->g, sphereList[sNdx].diffuse->b);
      //printf("%lf %lf %lf\n", l->diffuse->r, l->diffuse->g, l->diffuse->b);

      // Diffuse
      d.r = NdotL * l->diffuse->r * sphereList[sNdx].diffuse->r;
      d.g = NdotL * l->diffuse->g * sphereList[sNdx].diffuse->g;
      d.b = NdotL * l->diffuse->b * sphereList[sNdx].diffuse->b;

      // Specular
      RdotV = pow(dot(reflectVector, viewVector), 2.0);
      s.r = RdotV * l->specular->r * sphereList[sNdx].specular->r;
      s.g = RdotV * l->specular->g * sphereList[sNdx].specular->g;
      s.b = RdotV * l->specular->b * sphereList[sNdx].specular->b;

      //printf("%lf %lf %lf\n\n", d.r, d.g, d.b);
	} else {
      d.r = 0;
      d.g = 0;
      d.b = 0;

      s.r = 0;
      s.g = 0;
      s.b = 0;
   }
   
	total.r = a.r + d.r + s.r;
	total.g = a.g + d.g + s.g;
	total.b = a.b + d.b + s.b;

	//fprintf(stderr, "LIGHT A  r = %lf, g = %lf, b = %lf\n", l->ambient->r, l->ambient->g, l->ambient->b);
	//fprintf(stderr, "LIGHT D  r = %lf, g = %lf, b = %lf\n", l->diffuse->r, l->diffuse->g, l->diffuse->b);
	//fprintf(stderr, "LIGHT S  r = %lf, g = %lf, b = %lf\n", l->specular->r, l->specular->g, l->specular->b);

	//fprintf(stderr, "SPHERE r = %lf, g = %lf, b = %lf\n", total.r, total.g, total.b);	
	//fprintf(stderr, "PHONG  r = %lf, g = %lf, b = %lf\n\n", total.r, total.g, total.b);
	
	return total;
}

__device__ Point normalize(Point p) {
	Point n = p;
	double d = sqrt(dot(p, p));
	
	n.x /= d;
	n.y /= d;
	n.z /= d;
	
	return n;
}

__device__ double dot(Point p1, Point p2) {
   return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

// This is essentially p1 - p2:
__device__ Point subtractPoints(Point p1, Point p2) {
   Point p3;

   p3.x = p1.x - p2.x;
   p3.y = p1.y - p2.y;
   p3.z = p1.z - p2.z;

   return p3;
}
