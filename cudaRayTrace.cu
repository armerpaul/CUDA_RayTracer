/*
 * CPE 570 && CPE 458 Duet
 * Ray Tracer
 * Professor Christopher Lupo and Professor Zoe" Wood
 * Paul Armer(parmer), Bryan Ching(bcching), Matt Crussell(macrusse)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "glm/glm.hpp"
#include <math.h>
#include <algorithm>
#include "Image.h"
#include "types.h"
#include "cudaRayTrace.h"

Camera* CameraInit();
PointLight* LightInit();
Sphere* CreateSpheres();

__host__ __device__ Point CreatePoint(float x, float y, float z);
__host__ __device__ color_t CreateColor(float r, float g, float b);

__global__ void CUDARayTrace(Camera * cam, Plane * f, PointLight *l, Sphere * s, color_t * pixelList);

__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, PointLight* l);
__device__ color_t SphereShading(int sNdx, Ray r, Point p, Sphere* sphereList, PointLight* l);
__device__ float SphereRayIntersection(Sphere* s, Ray r);



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
   float aspectRatio = WINDOW_WIDTH; 
   aspectRatio /= WINDOW_HEIGHT;
   cudaEvent_t start, stop; 
   pixel_device = new color_t[WINDOW_WIDTH * WINDOW_HEIGHT];
  	
  	//SCENE SET UP
  	// (floor)
   Plane* floor = new Plane(), *f_d;
   
   
   // (spheres)
   Sphere* spheres = CreateSpheres(), *s_d;




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
  	
    // write the targa file to disk
  	img.WriteTga((char *)"raytraced.tga", true); 
  	// true to scale to max color, false to clamp to 1.0

    //FREE ALLOCS
    cudaFree(pixel_deviceD);
} 

/*
 * Initializes camera at point (X,Y,Z)
 */
Camera* CameraInit() {
   
   Camera* c = new Camera();
   
   c->eye = CreatePoint(0, 0, 0);//(X,Y,Z)
   c->lookAt = CreatePoint(0, 0, SCREEN_DISTANCE);
   c->lookUp = CreatePoint(0, 1, 0);

   c->u = CreatePoint(1, 0, 0);
   c->v = CreatePoint(0, 1, 0);
   c->w = CreatePoint(0, 0, 1);
   
   return c;
}
/*
 * Initializes light at hardcoded (X,Y,Z) coordinates
 */
PointLight* LightInit() {
   PointLight* l = new PointLight();

   l->ambient = CreateColor(0.2, 0.2, 0.2);
   l->diffuse = CreateColor(0.6, 0.6, 0.6);
   l->specular = CreateColor(0.4, 0.4, 0.4);

   l->position = CreatePoint(50, 0, -150);

   return l;
}
/*
 * Creates a point, for GLM Point has been defined as vec3
 */
__host__  __device__ Point CreatePoint(float x, float y, float z) {
   Point p;
   
   p.x = x;
   p.y = y;
   p.z = z;

   return p;
}
/*
 * Creates a color_t type color based on input values
 */
__host__ __device__ color_t CreateColor(float r, float g, float b) {
   color_t c;

   c.r = r;
   c.g = g;
   c.b = b;
   c.f = 1.0;

   return c;
}
/*
 * Creates NUM_SPHERES # of Spheres, with randomly chosen values on color, location, and size
 */
Sphere* CreateSpheres() {
   Sphere* spheres = new Sphere[NUM_SPHERES]();
   float randr, randg, randb;
   int num = 0;
   while (num < NUM_SPHERES) {
            randr = (rand()%1000) /1000.f ;
            randg = (rand()%1000) /1000.f ;
            randb = (rand()%1000) /1000.f ;
            spheres[num].radius = 11. - rand() % 10;
            spheres[num].center = CreatePoint(WINDOW_WIDTH/8 - rand() % 200,
                                              100 - rand() % 200,
                                              -200. - rand() %200);
            spheres[num].ambient = CreateColor(randr, randg, randb);
            spheres[num].diffuse = CreateColor(randr, randg, randb);
            spheres[num].specular = CreateColor(1., 1., 1.);
            num++;
   }

   return spheres;

}
/*
 * CUDA global function which performs ray tracing. Responsible for initializing and writing to output vector
 */
__global__ void CUDARayTrace(Camera * cam,Plane * f,PointLight * l, Sphere * s, color_t * pixelList)
{
    float tanVal = tan(FOV/2);
    float aspectRatio = WINDOW_WIDTH / WINDOW_HEIGHT;

    //CALCULATE ABSOLUTE ROW,COL
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


    //RAY TRACE
    returnColor = RayTrace(r, s, f, l);
    
    //CALC OUTPUT INDEX
    int index = row *WINDOW_WIDTH + col;
    
    //PLACE DATA IN INDEX
    pixelList[index].r = returnColor.r;
    pixelList[index].g = returnColor.g;
    pixelList[index].b = returnColor.b;
    pixelList[index].f = returnColor.f;
    
}
/*
 * Performs Ray tracing over all spheres for any ray r
 */
__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, PointLight* l) {
    color_t color = CreateColor(0, 0, 0); 
    float t, smallest;
   	Point p;
   	int i = 0, closestSphere = -1, sphereInShadow = false;
    int closestShadow = -1, closestSphereS = -1;
    
    //FIND CLOSEST SPHERE ALONG RAY R
    while (i < NUM_SPHERES) {
      t = SphereRayIntersection(s + i, r);

      if (t > 0 && (closestSphere < 0 || t < smallest)) {
        smallest = t;
			  closestSphere = i;
		  }
      i++;
    }

    //SETUP FOR SHADOW CALCULATIONS
    i = 0;
    Ray shadowRay;
    p = CreatePoint(r.direction.x * smallest, r.direction.y * smallest, r.direction.z * smallest);
    shadowRay.origin = p;
    shadowRay.direction = l->position-p;

    //DETERMINE IF SPHERE IS BLOCKING RAY FROM LIGHT TO SPHERE
    while (i <NUM_SPHERES){ 
      t = SphereRayIntersection(s + i, shadowRay);
      if(t > 0 && t < 1  && i != closestSphere && (closestSphereS <0|| t < closestShadow)){
        closestShadow = t;
        closestSphereS = i;
        sphereInShadow = true;
      }
      i++;
    }
    
    //IF SHADOWED, ONLY SHOW AMBIENT LIGHTING
    if(!sphereInShadow && closestSphere > -1)
    {
      return SphereShading(closestSphere, r, p, s, l);
    }
    else if(closestSphere > -1) 
    {
      color.r = l->ambient.r * s[closestSphere].ambient.r;
      color.g = l->ambient.g * s[closestSphere].ambient.g;
      color.b = l->ambient.b * s[closestSphere].ambient.b;
    }
    return color;
}
/*
 * Determines distance of intersection of Ray with Sphere, -1 returned if no intersection
 */
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
    
		if (t2 > t1 && t1 > 0) {
			return t1;
		
    } else if (t2 > 0) {
			return t2;
		
    }
	}
	return -1;
}
/*
 * Calculates Ambient, Diffuse, and Specular Shading for a single Ray
 */
__device__ color_t SphereShading(int sNdx, Ray r, Point p, Sphere* sphereList, PointLight* l) {
	  color_t a, d, s, total;
	  float NdotL, RdotV;
	  Point viewVector, lightVector, reflectVector, normalVector;

	  viewVector = glm::normalize((r.origin)-p);
	
	  lightVector = glm::normalize((l->position) -p);
	  normalVector = glm::normalize(p-(sphereList[sNdx].center));
	
    NdotL = glm::dot(lightVector, normalVector);
    reflectVector = (2.f *normalVector*NdotL)-lightVector;

    // Ambient
    a.r = l->ambient.r * sphereList[sNdx].ambient.r;
	  a.g = l->ambient.g * sphereList[sNdx].ambient.g;
	  a.b = l->ambient.b * sphereList[sNdx].ambient.b;
  
    // Diffuse
    d.r = NdotL * l->diffuse.r * sphereList[sNdx].diffuse.r * (NdotL > 0);
    d.g = NdotL * l->diffuse.g * sphereList[sNdx].diffuse.g * (NdotL > 0);
    d.b = NdotL * l->diffuse.b * sphereList[sNdx].diffuse.b * (NdotL > 0);
      
    // Specular
    RdotV = glm::pow(glm::dot(glm::normalize(reflectVector), viewVector), 100.f);
    s.r = RdotV * l->specular.r * sphereList[sNdx].specular.r * (NdotL > 0);
    s.g = RdotV * l->specular.g * sphereList[sNdx].specular.g * (NdotL > 0);
    s.b = RdotV * l->specular.b * sphereList[sNdx].specular.b * (NdotL > 0);
  
    total.r = glm::min(1.f, a.r + d.r+ s.r);
	  total.g = glm::min(1.f, a.g + d.g+ s.g);
	  total.b = glm::min(1.f, a.b + d.b+ s.b);
    total.f = 1.f;
	  return total;
}