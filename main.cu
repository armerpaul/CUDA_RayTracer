/*
  CPE 471 Lab 1 
  Base code for Rasterizer
  Example code using B. Somers' image code - writes out a sample tga
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "glm/glm.hpp"
#include <math.h>
#include <algorithm>
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
Plane* CreatePlanes();
__host__ __device__ Point CreatePoint(float x, float y, float z);
__host__ __device__ color_t CreateColor(float r, float g, float b);

__global__ void CUDARayTrace(Camera * cam, Plane * f, PointLight *l, Sphere * s, color_t * pixelList);
__global__ void CUDADummy(Camera * cam);//, Plane * f, PointLight *l, Sphere * s);

__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, PointLight* l);
__device__ color_t Shading(Ray r, Point p, Point normalVector, PointLight* l, color_t diffuse, color_t ambient, color_t specular); 
__device__ float SphereRayIntersection(Sphere* s, Ray r);
__device__ float PlaneRayIntersection(Plane* s, Ray r);
//__device__ float glm::dot(Point p1, Point p2);
//__device__ Point subtractPoints(Point p1, Point p2);
//__device__ Point glm::normalize(Point p);



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
   //srand ( time(NULL) );
   srand ( 0 );
   Image img(WINDOW_WIDTH, WINDOW_HEIGHT);
   Camera* camera = CameraInit(), * cam_d;
   PointLight* light = LightInit(), *l_d ;
   color_t * pixel_device = NULL;
   float aspectRatio = WINDOW_WIDTH; 
   aspectRatio /= WINDOW_HEIGHT;
   cudaEvent_t start, stop; 
   pixel_device = new color_t[WINDOW_WIDTH * WINDOW_HEIGHT];
  	
  	//SCENE SET UP
  	// (floor)
   Plane* floor = CreatePlanes(), *f_d;
   //floor->center = CreatePoint(0, -1 * WINDOW_HEIGHT / 2, -1 * WINDOW_WIDTH / 2);
   //floor->color = CreateColor(200, 200, 200);
   //floor->normal = CreatePoint(0, 0, -1 * WINDOW_WIDTH / 2);
   // (spheres)
   Sphere* spheres = CreateSpheres(), *s_d;


  // HANDLE_ERROR( cudaMemcpyToSymbol(cam, camera, sizeof(Camera)) );


   color_t * pixel_deviceD;
   HANDLE_ERROR( cudaMalloc(&pixel_deviceD,sizeof(color_t) * WINDOW_WIDTH * WINDOW_HEIGHT) );

   HANDLE_ERROR( cudaMalloc((void**)&cam_d, sizeof(Camera)) );
   HANDLE_ERROR( cudaMalloc(&f_d, sizeof(Plane)) );
   HANDLE_ERROR( cudaMalloc(&l_d, sizeof(PointLight)) );
   HANDLE_ERROR( cudaMalloc(&s_d,  sizeof(Sphere)*NUM_SPHERES));
   
   HANDLE_ERROR( cudaMemcpy(l_d, light, sizeof(PointLight), cudaMemcpyHostToDevice) );
   HANDLE_ERROR( cudaMemcpy(cam_d, camera,sizeof(Camera), cudaMemcpyHostToDevice) );
   HANDLE_ERROR( cudaMemcpy(f_d, floor,sizeof(Plane), cudaMemcpyHostToDevice) );
   HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
   
   //CUDA Timing 
   HANDLE_ERROR( cudaEventCreate(&start) );
   HANDLE_ERROR( cudaEventCreate(&stop) );
   HANDLE_ERROR( cudaEventRecord(start, 0));

   // The Kernel Call
   CUDARayTrace<<< (WINDOW_WIDTH * WINDOW_HEIGHT + 383) / 384, 384  >>>(cam_d, f_d, l_d, s_d, pixel_deviceD);
   //CUDARayTrace<<< 100, 575 >>>(cam_d, f_d, l_d, s_d, pixel_deviceD);

   //CUDADummy<<<1, 1>>>(cam_d);//, f_d, l_d, s_d);
   // Coming Back
   HANDLE_ERROR(cudaEventRecord( stop, 0));
   HANDLE_ERROR(cudaEventSynchronize( stop ));
   float elapsedTime;
   HANDLE_ERROR(cudaEventElapsedTime( &elapsedTime, start, stop));

   printf("GPU computation time: %.1f ms\n", elapsedTime);

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
   l->specular = CreateColor(0.4, 0.4, 0.4);

   l->position = CreatePoint(600, 0, -200);

   return l;
}

__host__  __device__ Point CreatePoint(float x, float y, float z) {
   Point p;
   
   p.x = x;
   p.y = y;
   p.z = z;

   return p;
}

__host__ __device__ color_t CreateColor(float r, float g, float b) {
   color_t c;

   c.r = r;
   c.g = g;
   c.b = b;
   c.f = 1.0;

   return c;
}

Sphere* CreateSpheres() {
   Sphere* spheres = new Sphere[NUM_SPHERES]();
   int i=0, j=0, k=0, num=0;

   while (num < NUM_SPHERES) {
      for (i=0; i < 6 && num < NUM_SPHERES; i++) {
         for (j=0; j < 5 && num < NUM_SPHERES; j++) {
            spheres[num].radius = 40. - rand() % 10;
            spheres[num].center = CreatePoint(WINDOW_WIDTH/10 + 100 - rand() % 300,
                                              100 - rand() % 200,
                                              -200. - rand() %200); //- rand() % 20);
            spheres[num].ambient = CreateColor(std::min((i + j) * .15, 1.),
                                               std::min((j + k) * .15, 1.),
                                               std::max(1. - (k + i) * .15, 0.));
            spheres[num].diffuse = CreateColor(std::min((i + j) * .15, 1.),
                                               std::min((j + k) * .15, 1.),
                                               std::max(1. - (k + i) * .15, 0.));
            spheres[num].specular = CreateColor(1., 1., 1.);
            num++;
         }
      }
      k++;
   }

   return spheres;

}

#define NUM_PLANES 1

Plane* CreatePlanes() {
   Plane* planes = new Plane[NUM_PLANES]();
   int num=0;
   while (num < NUM_PLANES) {
            planes[num].normal = CreatePoint(0,0,0) ;
            planes[num].center = CreatePoint(0,200,0);
            planes[num].ambient = CreateColor(.5,.5,.5);
            planes[num].diffuse =  CreateColor(.5,.5,.5);
            planes[num].specular = CreateColor(1.,1.,1.);
            num++;
   }
   return planes;
}

__global__ void CUDADummy(Camera * cam)//, Plane * f ,PointLight * l,Sphere * s)
{
  printf("C addr: %f\n", cam->lookAt.z);//, F addr: %f, L addr: %f, Sphere addr: %s", cam, f, l, s); 
}

__global__ void CUDARayTrace(Camera * cam,Plane * f,PointLight * l, Sphere * s, color_t * pixelList)
{
    float tanVal = tan(FOV/2);
    float aspectRatio = WINDOW_WIDTH / WINDOW_HEIGHT;
    int row = (blockIdx.x *blockDim.x + threadIdx.x ) / WINDOW_WIDTH;
    int col = (blockIdx.x *blockDim.x + threadIdx.x ) % WINDOW_WIDTH;
    color_t returnColor;
    Ray r;
    
    //BOUNDARY CHECK
    if(row > WINDOW_HEIGHT)
      return;

    //INIT RAY VALUES
	  r.origin = cam->eye;
    r.direction = cam->lookAt;
    r.direction.y = tanVal - (2 * tanVal / WINDOW_HEIGHT) * row;
    r.direction.x = -1 * aspectRatio * tanVal + (2 * tanVal / WINDOW_HEIGHT) * col;


    returnColor = RayTrace(r, s, f, l);
    int index = row *WINDOW_WIDTH + col;
    
    //if(index == 0)
    //  printf("I RAN, I WORKED\n");
    pixelList[index].r = returnColor.r;
    pixelList[index].g = returnColor.g;
    pixelList[index].b = returnColor.b;
    pixelList[index].f = returnColor.f;
    
}

__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, PointLight* l) {
   // color_t black = CreateColor(1.f, 0, 0); 
    float t, smallest;
   	Point p, normalVector;
   	int i = 0, closestSphere = -1, closestPlane = -1;

    
    while (i < NUM_SPHERES) {
      t = SphereRayIntersection(s + i, r);


      if (t > 0 && (closestSphere < 0 || t < smallest)) {
        smallest = t;
        closestSphere = i;
      }
      i++;
    }
i=0;
    while (i < NUM_PLANES) {
      t = PlaneRayIntersection(f, r);


      if (t > 0 && (closestSphere < 0 || t < smallest)) {
        smallest = t;
        closestPlane = i;
      }
      i++;
    }

    if(closestPlane > -1)
    {
      //plane closer than sphere
      p = CreatePoint(r.direction.x * smallest, r.direction.y * smallest, r.direction.z * smallest);
	    normalVector = glm::normalize(f[closestPlane].normal-f[closestPlane].center);
      return Shading(r, p, normalVector, l, f[closestPlane].diffuse,
      f[closestPlane].ambient,f[closestPlane].specular);
    }

    else   if (closestSphere > -1) {
      p = CreatePoint(r.direction.x * smallest, r.direction.y * smallest, r.direction.z * smallest);
	    normalVector = glm::normalize((s[closestSphere].center)-p);
      return Shading(r, p, normalVector, l, s[closestSphere].diffuse,
      s[closestSphere].ambient,s[closestSphere].specular);
    }

    return CreateColor(0,0,0);
}

__device__ float PlaneRayIntersection(Plane *p, Ray r)
{
  float t;
  Point N = p->normal - p->center;
  float denominator = glm::dot(r.direction,N);
  if(denominator!=0)
  {
    t = (glm::dot(p->center-r.origin,N)) / denominator;
    return t;
  }
  else
  {
    return -1;
  }
}


__device__ float SphereRayIntersection(Sphere* s, Ray r) {
	float a, b, c, d, t1, t2;
    
    a = glm::dot((r.direction), (r.direction));
    b = glm::dot((r.origin)- (s->center),(r.direction));
    c = glm::dot((r.origin)-(s->center), (r.origin)- (s->center))
            - (s->radius * s->radius);
    d = (b * b) - (a * c);
    
    if (d >= 0) {

		t1 = (-1 * b - sqrt(d)) / a;
		t2 = (-1 * b + sqrt(d)) / a;
    
		if (t2 < t1 && t2 > 0) {
			return t2;
		} else if (t1 > 0) {
			return t1;
		}
	}
	return d;
}

__device__ color_t Shading(Ray r, Point p, Point normalVector,
PointLight* l, color_t diffuse, color_t ambient, color_t specular) {
	color_t a, d, s, total;
	float NdotL, RdotV;
	Point viewVector, lightVector, reflectVector;

   //printf("r->%lf g->%lf b->%lf\n", l->ambient->r, l->ambient->g, l->ambient->b);
   //printf("r->%lf g->%lf b->%lf\n", l->diffuse->r, l->diffuse->g, l->diffuse->b);
   //printf("r->%lf g->%lf b->%lf\n\n", l->specular->r, l->specular->g, l->specular->b);

	viewVector = glm::normalize((r.origin)-p);
	
	lightVector = glm::normalize((l->position) -p);
	
  NdotL = glm::dot(lightVector, normalVector);
//  reflectVector = normalVector - lightVector;
  reflectVector = (2.f *normalVector*NdotL) -lightVector;
 // reflectVector = glm::reflect(-lightVector,normalVector);
	/*
  reflectTemp = 2 * NdotL;
	reflectVector.x *= reflectTemp;
	reflectVector.y *= reflectTemp;
	reflectVector.z *= reflectTemp;
	*/

  a.r = l->ambient.r * ambient.r;
	a.g = l->ambient.g * ambient.g;
	a.b = l->ambient.b * ambient.b;
  
  // Diffuse
  d.r = NdotL * l->diffuse.r * diffuse.r * (NdotL > 0);
  d.g = NdotL * l->diffuse.g * diffuse.g * (NdotL > 0);
  d.b = NdotL * l->diffuse.b * diffuse.b * (NdotL > 0);
      
  // Specular
  RdotV = glm::pow(glm::dot(glm::normalize(reflectVector), viewVector), 100.f);
  //RdotV = glm::dot(reflectVector,viewVector) *glm::dot(reflectVector,viewVector) ;
  s.r = RdotV * l->specular.r * specular.r * (NdotL > 0) *(RdotV>0);
  s.g = RdotV * l->specular.g * specular.g * (NdotL > 0) *(RdotV>0);
  s.b = RdotV * l->specular.b * specular.b * (NdotL > 0) *(RdotV>0);
/*	
  total.r =  -s.r;
	total.g =  -s.g;
	total.b =  -s.b;*/
  total.r = glm::min(1.f, a.r + d.r + s.r);
	total.g = glm::min(1.f, a.g + d.g + s.g);
	total.b = glm::min(1.f, a.b + d.b + s.b);
  total.f = 1.f;
	return total;
}
/*
__device__ Point glm::normalize(Point p) {
	float d = sqrt(glm::dot(p, p));
  
  p.x /= d;
	p.y /= d;
	p.z /= d;
	
	return p;
}
*/
/*
__device__ float glm::dot(Point p1, Point p2) {
  return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
//	return glm::dot(p1,p2);
}
*/
// This is essentially p1 - p2:
/*
__device__ Point subtractPoints(Point p1, Point p2) {
   Point p3;

   p3.x = p1.x - p2.x;
   p3.y = p1.y - p2.y;
   p3.z = p1.z - p2.z;
   
  return p1-p2;

}*/
