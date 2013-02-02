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

/*__constant__ Sphere s[NUM_SPHERES];
__constant__ Plane * f;
__constant__ PointLight * l;
__constant__ Camera *cam;*/

Camera* CameraInit();
PointLight* LightInit();
Sphere* CreateSpheres();
__host__ __device__ Point CreatePoint(double x, double y, double z);
__host__ __device__ color_t CreateColor(double r, double g, double b);

__global__ void CUDARayTrace(color_t * pixelList);
__global__ void CUDADummy(Camera * cam);//, Plane * f, PointLight *l, Sphere * s);

__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, PointLight* l);
__device__ color_t SphereShading(int sNdx, Ray r, Point p, Sphere* sphereList, PointLight* l);
__device__ double SphereRayIntersection(Sphere* s, Ray r);
__device__ double dot(Point p1, Point p2);
__device__ Point subtractPoints(Point p1, Point p2);
__device__ Point normalize(Point p);

/* 
 *  Handles CUDA errors, taking from provided sample code on clupo site
 */

static void HandleError( cudaError_t err, const char * file, int line)
{
    if(err !=cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
            exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main(void) 
{
   // set up for random num generator
   srand ( time(NULL) );
   
   Image img(WINDOW_WIDTH, WINDOW_HEIGHT);
   Camera* camera = CameraInit(), * cam_d;
   PointLight* light = LightInit(), *l_d ;
   color_t * pixel_device = NULL;
   double aspectRatio = WINDOW_WIDTH; 
   aspectRatio /= WINDOW_HEIGHT;
   
   pixel_device = new color_t[WINDOW_WIDTH * WINDOW_HEIGHT];
  	
  	//SCENE SET UP
  	// (floor)
   Plane* floor = new Plane(), *f_d;
   //floor->center = CreatePoint(0, -1 * WINDOW_HEIGHT / 2, -1 * WINDOW_WIDTH / 2);
   //floor->color = CreateColor(200, 200, 200);
   //floor->normal = CreatePoint(0, 0, -1 * WINDOW_WIDTH / 2);
   // (spheres)
   Sphere* spheres = CreateSpheres(), *s_d;


   //HANDLE_ERROR( cudaMemcpyToSymbol(cam, camera, sizeof(Camera)) );


   color_t * pixel_deviceD;
   HANDLE_ERROR( cudaMalloc(&pixel_deviceD,sizeof(color_t) * WINDOW_WIDTH * WINDOW_HEIGHT) );

   HANDLE_ERROR( cudaMalloc((void**)&cam_d, sizeof(Camera)) );
  /* HANDLE_ERROR( cudaMalloc(&f_d, sizeof(Plane)) );
   HANDLE_ERROR( cudaMalloc(&l_d, sizeof(PointLight)) );
   HANDLE_ERROR( cudaMalloc(&s_d,  sizeof(Sphere)*NUM_SPHERES));
  */ 
   //HANDLE_ERROR( cudaMemcpy(l_d, light, sizeof(PointLight), cudaMemcpyHostToDevice) );
   HANDLE_ERROR( cudaMemcpy(cam_d, camera,sizeof(Camera), cudaMemcpyHostToDevice) );
  /* HANDLE_ERROR( cudaMemcpy(f_d, floor,sizeof(Plane), cudaMemcpyHostToDevice) );
   HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
  */ 
   
   //MEMCPY'S
   // The Kernel Call
   //CUDARayTrace<<< (WINDOW_WIDTH * WINDOW_HEIGHT + 1023) / 1024, 1024 >>>(cam_d, f_d, l_d, s_d, pixel_deviceD);
   CUDADummy<<<1, 1>>>(cam_d);//, f_d, l_d, s_d);
   // Coming Back

   HANDLE_ERROR( cudaMemcpy(pixel_device, pixel_deviceD,sizeof(color_t) * WINDOW_WIDTH * WINDOW_HEIGHT, cudaMemcpyDeviceToHost) );
   fflush(stdout);
   
   for (int i=0; i < WINDOW_WIDTH; i++) {
		for (int j=0; j < WINDOW_HEIGHT; j++) {
         //Looping over the Rays
     		img.pixel(i, j, pixel_device[j*WINDOW_WIDTH + i]);
		    }
  	}
  	
	// IMAGE OUTPUT
	//
  	// write the targa file to disk
  	img.WriteTga((char *)"raytraced.tga", true); 
  	// true to scale to max color, false to clamp to 1.0
   cudaFree(pixel_deviceD);
} 


Camera* CameraInit() {
   
   Camera* c = new Camera();
   
   c->eye = CreatePoint(0, 0, 0);
   c->lookAt = CreatePoint(0, 0, SCREEN_DISTANCE);
   c->lookUp = CreatePoint(0, 1, 0);

   c->u = CreatePoint(1, 0, 0);
   c->v = CreatePoint(0, 1, 0);
   c->w = CreatePoint(0, 0, 1);
   
   return c;
}

PointLight* LightInit() {
   PointLight* l = new PointLight();

   l->ambient = CreateColor(0.2, 0.2, 0.2);
   l->diffuse = CreateColor(0.6, 0.6, 0.6);
   l->specular = CreateColor(0.9, 0.9, 0.9);

   l->position = CreatePoint(-1 * WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, -1 * WINDOW_WIDTH / 2);

   return l;
}

__host__  __device__ Point CreatePoint(double x, double y, double z) {
   Point p;
   
   p.x = x;
   p.y = y;
   p.z = z;

   return p;
}

__host__ __device__ color_t CreateColor(double r, double g, double b) {
   color_t c;

   c.r = r;
   c.g = g;
   c.b = b;
   c.f = 1.0;

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
__global__ void CUDADummy(Camera * cam)//, Plane * f ,PointLight * l,Sphere * s)
{
  printf("C addr: %f\n", cam);//, F addr: %f, L addr: %f, Sphere addr: %s", cam, f, l, s); 
}
__global__ void CUDARayTrace(color_t * pixelList)
{
    double tanVal = tan(FOV/2);
    double aspectRatio = WINDOW_WIDTH / WINDOW_HEIGHT;
    int row = (blockIdx.x *blockDim.x + threadIdx.x ) / WINDOW_WIDTH;
    int col = (blockIdx.x *blockDim.x + threadIdx.x ) % WINDOW_WIDTH;
    color_t returnColor;
    Ray r;
    
    //BOUNDARY CHECK
    if(row > WINDOW_HEIGHT)
      return;

    //INIT RAY VALUES
//	  r.origin = cam->eye;
//    r.direction = cam->lookAt;
    r.direction.y = tanVal - (2 * tanVal / WINDOW_HEIGHT) * row;
    r.direction.x = -1 * aspectRatio * tanVal + (2 * tanVal / WINDOW_HEIGHT) * col;


//    returnColor = RayTrace(r, s, f, l);
    int index = row *WINDOW_WIDTH + col;
    
    //printf("%d %f\n",index,pixelList[index].r);
    if(index == 0)
      printf("I SHOULD RUN IF WORKS");
    pixelList[index].r = returnColor.r;
    pixelList[index].g = returnColor.g;
    pixelList[index].b = returnColor.b;
    pixelList[index].f = returnColor.f;
    
    
    /*pixelList[index].r = index%256;
    pixelList[index].g = 0;
    pixelList[index].b = 0;
    pixelList[index].f = 0;
*/
}

__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, PointLight* l) {
    color_t black = CreateColor(0, 0, 0); 
    double t, smallest;
   	Point p;
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
      	p = CreatePoint(r.direction.x * smallest, r.direction.y * smallest, r.direction.z * smallest);
      	return SphereShading(closestSphere, r, p, s, l);
   }
   
   return black;
}

__device__ double SphereRayIntersection(Sphere* s, Ray r) {
	double a, b, c, d, t1, t2;
    
    a = dot((r.direction), (r.direction));
    b = dot(subtractPoints((r.origin), (s->center)),(r.direction));
    c = dot(subtractPoints((r.origin), (s->center)), subtractPoints((r.origin), (s->center)))
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

__device__ color_t SphereShading(int sNdx, Ray r, Point p, Sphere* sphereList, PointLight* l) {
	color_t a, d, s, total;
	double reflectTemp, NdotL, RdotV;
	Point viewVector, lightVector, reflectVector, normalVector;

   //printf("r->%lf g->%lf b->%lf\n", l->ambient->r, l->ambient->g, l->ambient->b);
   //printf("r->%lf g->%lf b->%lf\n", l->diffuse->r, l->diffuse->g, l->diffuse->b);
   //printf("r->%lf g->%lf b->%lf\n\n", l->specular->r, l->specular->g, l->specular->b);

	viewVector = normalize(subtractPoints((r.origin), p));
	lightVector = normalize(subtractPoints(p, (l->position)));
	normalVector = normalize(subtractPoints(p, (sphereList[sNdx].center)));
	reflectVector = subtractPoints(normalVector, lightVector);

  NdotL = dot(lightVector, normalVector);
	
  reflectTemp = 2 * NdotL;
	reflectVector.x *= reflectTemp;
	reflectVector.y *= reflectTemp;
	reflectVector.z *= reflectTemp;
	
  a.r = l->ambient.r * sphereList[sNdx].ambient.r;
	a.g = l->ambient.g * sphereList[sNdx].ambient.g;
	a.b = l->ambient.b * sphereList[sNdx].ambient.b;
  
   if (NdotL > 0 ) {
   //if(true){
      //printf("%lf\n", NdotL);
      //printf("%lf %lf %lf\n", sphereList[sNdx].diffuse->r, sphereList[sNdx].diffuse->g, sphereList[sNdx].diffuse->b);
      //printf("%lf %lf %lf\n", l->diffuse->r, l->diffuse->g, l->diffuse->b);

      // Diffuse
      d.r = .5; //NdotL * l->diffuse.r * sphereList[sNdx].diffuse.r;
      d.g = .5; //NdotL * l->diffuse.g * sphereList[sNdx].diffuse.g;
      d.b = .5; //NdotL * l->diffuse.b * sphereList[sNdx].diffuse.b;
      
      // Specular
      RdotV = dot(reflectVector, viewVector) * dot(reflectVector, viewVector);
      s.r = RdotV * l->specular.r * sphereList[sNdx].specular.r;
      s.g = RdotV * l->specular.g * sphereList[sNdx].specular.g;
      s.b = RdotV * l->specular.b * sphereList[sNdx].specular.b;
      
      //printf("%lf %lf %lf\n\n", d.r, d.g, d.b);
	} else {
      d.r = 0;
      d.g = 0;
      d.b = 0;

      s.r = 0;
      s.g = 0;
      s.b = 0;
   }
   
	//total.r = a.r + d.r + s.r;
	//total.g = a.g + d.g + s.g;
	//total.b = a.b + d.b + s.b;

	//fprintf(stderr, "LIGHT A  r = %lf, g = %lf, b = %lf\n", l->ambient->r, l->ambient->g, l->ambient->b);
	//fprintf(stderr, "LIGHT D  r = %lf, g = %lf, b = %lf\n", l->diffuse->r, l->diffuse->g, l->diffuse->b);
	//fprintf(stderr, "LIGHT S  r = %lf, g = %lf, b = %lf\n", l->specular->r, l->specular->g, l->specular->b);

	//fprintf(stderr, "SPHERE r = %lf, g = %lf, b = %lf\n", total.r, total.g, total.b);	
	//fprintf(stderr, "PHONG  r = %lf, g = %lf, b = %lf\n\n", total.r, total.g, total.b);
	
	return total;
}

__device__ Point normalize(Point p) {
	double d = sqrt(dot(p, p));
  //double d = 1;
  p.x /= d;
	p.y /= d;
	p.z /= d;
	
	return p;
}

__device__ double dot(Point p1, Point p2) {
  return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
  //return 5;
}

// This is essentially p1 - p2:
__device__ Point subtractPoints(Point p1, Point p2) {
   Point p3;

   p3.x = p1.x - p2.x;
   p3.y = p1.y - p2.y;
   p3.z = p1.z - p2.z;
   
   return p3;

}
