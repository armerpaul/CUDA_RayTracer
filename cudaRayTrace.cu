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

Camera * camera, *cam_d;
PointLight *light, *l_d;
Plane * planes, *p_d;
Sphere * spheres, *s_d;
float theta, stheta;

Camera* CameraInit();
PointLight* LightInit();
Sphere* CreateSpheres();
Plane* CreatePlanes();
__host__ __device__ Point CreatePoint(float x, float y, float z);
__host__ __device__ color_t CreateColor(float r, float g, float b);

__global__ void CUDARayTrace(Camera * cam, Plane * f, PointLight *l, Sphere * s, uchar4 * position);

__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, PointLight* l);
__device__ color_t SphereShading(int sNdx, Ray r, Point p, Sphere* sphereList, PointLight* l);
__device__ color_t Shading(Ray r, Point p, Point normalVector, PointLight* l, color_t diffuse, color_t ambient, color_t specular); 
__device__ float SphereRayIntersection(Sphere* s, Ray r);
__device__ float PlaneRayIntersection(Plane* s, Ray r);

static void HandleError( cudaError_t err, const char * file, int line)
{
	if(err !=cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/* 
 *  Handles CUDA errors, taking from provided sample code on clupo site
 */
extern "C" void setup_scene()
{
	HANDLE_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	camera = CameraInit();
	light = LightInit();
	spheres = CreateSpheres();
	planes = CreatePlanes(); 

	HANDLE_ERROR( cudaMalloc((void**)&cam_d, sizeof(Camera)) );
	HANDLE_ERROR( cudaMalloc(&p_d, sizeof(Plane)*NUM_PLANES) );
	HANDLE_ERROR( cudaMalloc(&l_d, sizeof(PointLight)) );
	HANDLE_ERROR( cudaMalloc(&s_d,  sizeof(Sphere)*NUM_SPHERES));

	HANDLE_ERROR( cudaMemcpy(l_d, light, sizeof(PointLight), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(cam_d, camera,sizeof(Camera), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(p_d, planes,sizeof(Plane)*NUM_PLANES, cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
	theta = 0;
	stheta = 0;
}

extern "C" void ijklMove(unsigned char key)
{
	float sin_theta_x, cos_theta_x, sin_theta_y,cos_theta_y;
	switch(key){
	case('i'):
		camera->theta_x+=.05;
		break; 
	case('k'):
		camera->theta_x-=.05;
		break;
	case('j'):
		camera->theta_y-=.05;
		break;
	case('l'):
		camera->theta_y+=.05;
		break;
	}
	sin_theta_x = sin(camera->theta_x);
	sin_theta_y = sin(camera->theta_y);
	cos_theta_x = cos(camera->theta_x);
	cos_theta_y = cos(camera->theta_y);

	camera->lookAt = glm::normalize(CreatePoint(sin_theta_y ,sin_theta_x , -1*cos_theta_x*cos_theta_y));
	camera->lookRight = glm::normalize(CreatePoint(cos_theta_y , 0 , sin_theta_y));
	camera->lookUp = glm::normalize(CreatePoint(0,cos_theta_x, sin_theta_x));
}



extern "C" void wasdMove(unsigned char key)
{
	Point move;
	switch(key){
	case('w'):
		move = 10.f * camera->lookAt;
		break; 
	case('s'):
		move = -10.f *camera->lookAt;
		break;
	case('a'):
		move = -10.f * camera->lookRight;
		break;
	case('d'):
		move = 10.f * camera->lookRight;
		break;
	}
	camera->eye += move;
}
extern "C" void misc(unsigned char key)
{
	Point center;
	switch(key){
	case('q'):
		{
			camera = CameraInit();
			break;
		}
	case('r'):
		{
			spheres = CreateSpheres();
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('-'):
		{
			for(int i = 0; i < NUM_SPHERES; i++)
				spheres[i].radius = glm::max(0.f, spheres[i].radius-1);
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('='):
		{
			for(int i = 0; i < NUM_SPHERES; i++)
				spheres[i].radius = glm::min(100.f, spheres[i].radius+1);
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('o'):
		{
			Point center = *new Point(0,0,-2400);
			center = *new Point(0,0,-2400);
			for(int i = 0; i < NUM_SPHERES; i++)
			{
				Point c_dir = glm::normalize(spheres[i].center - center);
				Point move_dir = glm::cross(c_dir, *new Point(0,1,0));
				spheres[i].center += 5.f*move_dir;
				spheres[i].center -= 5.f*c_dir;

			}
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('p'):
		{
			center = *new Point(0,0,-2400);
			for(int i = 0; i < NUM_SPHERES; i++)
			{
				Point c_dir = glm::normalize(spheres[i].center - center);
				Point move_dir = glm::cross(c_dir, *new Point(0,1,0));
				spheres[i].center -= 10.f*move_dir;
				spheres[i].center += 10.f*c_dir;

			}
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('['):
		{
			center = camera->eye;
			for(int i = 0; i < NUM_SPHERES; i++)
			{
				Point c_dir = glm::normalize(spheres[i].center - center);
				Point move_dir = glm::cross(c_dir, *new Point(0,1,0));
				spheres[i].center += 10.f*move_dir;
				//spheres[i].center -= 10.f*c_dir;

			}
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}

	case(']'):
		{
			center = camera->eye;
			for(int i = 0; i < NUM_SPHERES; i++)
			{
				Point c_dir = glm::normalize(spheres[i].center - center);
				Point move_dir = glm::cross(c_dir, *new Point(0,1,0));
				spheres[i].center -= 10.f*move_dir;
				//spheres[i].center += 10.f*c_dir;

			}
			HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );
			break;
		}
	case('9'):
		{

			for(int i = 0; i < NUM_PLANES; i++) {
				planes[i].ambient.r = glm::max(planes[i].ambient.r - .05f, 0.f);
				planes[i].ambient.g = glm::max(planes[i].ambient.g - .05f, 0.f);
				planes[i].ambient.b = glm::max(planes[i].ambient.b - .05f, 0.f);
				planes[i].diffuse.r = glm::max(planes[i].diffuse.r - .05f, 0.f);
				planes[i].diffuse.g = glm::max(planes[i].diffuse.g - .05f, 0.f);
				planes[i].diffuse.b = glm::max(planes[i].diffuse.b - .05f, 0.f);
			}

			HANDLE_ERROR( cudaMemcpy(p_d, planes,sizeof(Plane)*NUM_PLANES, cudaMemcpyHostToDevice) );
			break;
		}
	case('0'):
		{
			for(int i = 0; i < NUM_PLANES; i++) {
				planes[i].ambient.r = glm::min(planes[i].ambient.r + .05f, 1.f);
				planes[i].ambient.g = glm::min(planes[i].ambient.g + .05f, 1.f);
				planes[i].ambient.b = glm::min(planes[i].ambient.b + .05f, 1.f);
				planes[i].diffuse.r = glm::min(planes[i].diffuse.r + .05f, 1.f);
				planes[i].diffuse.g = glm::min(planes[i].diffuse.g + .05f, 1.f);
				planes[i].diffuse.b = glm::min(planes[i].diffuse.b + .05f, 1.f);
			}

			HANDLE_ERROR( cudaMemcpy(p_d, planes,sizeof(Plane)*NUM_PLANES, cudaMemcpyHostToDevice) );
			break;
		}
	}
}
extern "C" void launch_kernel(uchar4* pos, unsigned int image_width, 
		unsigned int image_height, float time)
{
	Point move;

	light->position.x -= 2 *sin(theta += .01);	

	spheres[NUM_SPHERES-1].radius=5;
	spheres[NUM_SPHERES-1].center=light->position;
	spheres[NUM_SPHERES-1].ambient=CreateColor(1,0,0);
	spheres[NUM_SPHERES-1].diffuse=CreateColor(1,1,1);
	spheres[NUM_SPHERES-1].specular=CreateColor(1,1,1);

	HANDLE_ERROR( cudaMemcpy(l_d, light, sizeof(PointLight), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(cam_d, camera,sizeof(Camera), cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMemcpy(s_d, spheres,sizeof(Sphere)*NUM_SPHERES, cudaMemcpyHostToDevice) );

	// The Kernel Call
	dim3 gridSize((WINDOW_WIDTH+15)/16, (WINDOW_HEIGHT+15)/16);
	dim3 blockSize(16,16);
	CUDARayTrace<<< gridSize, blockSize  >>>(cam_d, p_d, l_d, s_d, pos);
	cudaThreadSynchronize();
} 

/*
 * Initializes camera at point (X,Y,Z)
 */
Camera* CameraInit() {

	Camera* c = new Camera();

	c->eye = CreatePoint(0, 0, 0);//(X,Y,Z)
	c->lookAt = CreatePoint(0, 0, SCREEN_DISTANCE);
	c->lookUp = CreatePoint(0, 1, 0);
	c->lookRight = CreatePoint(1, 0, 0);
	c->theta_x = 0;
	c->theta_y = 0;
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

	l->position = CreatePoint(50, 50, -400);

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
	while (num < NUM_SPHERES-1) {
		randr = (rand()%1000) /1000.f ;
		randg = (rand()%1000) /1000.f ;
		randb = (rand()%1000) /1000.f ;
		spheres[num].radius = 80. - rand() % 60;
		spheres[num].center = CreatePoint(2400. - rand() % 4800,
				700 - rand() % 1100,
				-0. - rand() %4800);
		spheres[num].ambient = CreateColor(randr, randg, randb);
		spheres[num].diffuse = CreateColor(randr, randg, randb);
		spheres[num].specular = CreateColor(1., 1., 1.);
		num++;
	}


	spheres[NUM_SPHERES-1].radius=5;
	spheres[NUM_SPHERES-1].center=light->position;
	spheres[NUM_SPHERES-1].ambient=CreateColor(1,1,1);
	spheres[NUM_SPHERES-1].diffuse=CreateColor(1,1,1);
	spheres[NUM_SPHERES-1].specular=CreateColor(1,1,1);

	return spheres;
}

/*
 * Creates NUM_PLANES NUMBER OF PLANES, CURRENTLY THIS IS HARDCODED
 */
Plane* CreatePlanes() {
	Plane* planes = new Plane[NUM_PLANES]();
	planes[0].normal = CreatePoint(0,1,0);
	planes[0].center = CreatePoint(0,-500,0);
	planes[0].ambient = planes[0].diffuse = planes[0].specular = CreateColor(1,1,1);

	planes[1].normal = CreatePoint(0,-1,0) ;
	planes[1].center = CreatePoint(0,800,0);
	planes[1].ambient = planes[1].diffuse = planes[1].specular = CreateColor(1,1,1);

	planes[2].normal = CreatePoint(0,0, 1) ;
	planes[2].center = CreatePoint(0,0,-5000);
	planes[2].ambient = planes[2].diffuse = planes[2].specular = CreateColor(1,1,1);

	planes[3].normal = CreatePoint(1,0,0) ;
	planes[3].center = CreatePoint(-2400,0,0);
	planes[3].ambient = planes[3].diffuse = planes[3].specular = CreateColor(1,1,1);

	planes[4].normal = CreatePoint(-1,0,0) ;
	planes[4].center = CreatePoint(2400,0, 0);
	planes[4].ambient = planes[4].diffuse = planes[4].specular = CreateColor(1,1,1);

	planes[5].normal = CreatePoint(0,0,-1) ;
	planes[5].center = CreatePoint(0,0, 1000);
	planes[5].ambient = planes[5].diffuse = planes[5].specular = CreateColor(1,1,1);

	return planes;
}

/*
 * CUDA global function which performs ray tracing. Responsible for initializing and writing to output vector
 */
__global__ void CUDARayTrace(Camera * cam,Plane * f,PointLight * l, Sphere * s, uchar4 * pos)
{
	float tanVal = tan(FOV/2);

	//CALCULATE ABSOLUTE ROW,COL
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	color_t returnColor;
	Ray r;

	//BOUNDARY CHECK
	if(row >= WINDOW_HEIGHT || col >= WINDOW_WIDTH)
		return;

	float rvaly = tanVal - (2 * tanVal / WINDOW_HEIGHT) * row;
	float rvalx = -1 * WINDOW_WIDTH / WINDOW_HEIGHT * tanVal + (2 * tanVal / WINDOW_HEIGHT) * col;
	//INIT RAY VALUES
	r.origin = cam->eye;
	r.direction = cam->lookAt;
	r.direction += (rvalx * cam->lookRight);
	r.direction += (rvaly * cam->lookUp);
	r.direction = glm::normalize(r.direction);
	//r.direction.y += tanVal - (2 * tanVal / WINDOW_HEIGHT) * row;
	//r.direction.x += -1 * WINDOW_WIDTH / WINDOW_HEIGHT * tanVal + (2 * tanVal / WINDOW_HEIGHT) * col;

	//RAY TRACE
	returnColor = RayTrace(r, s, f, l);

	//CALC OUTPUT INDEX
	int index = row *WINDOW_WIDTH + col;

	//PLACE DATA IN INDEX
	pos[index].x = 0xFF * returnColor.r;
	pos[index].y = 0xFF * returnColor.g;
	pos[index].z = 0xFF * returnColor.b;
	pos[index].w = 0xFF * returnColor.f;

}

/*
 * Performs Ray tracing over all spheres for any ray r
 */
__device__ color_t RayTrace(Ray r, Sphere* s, Plane* f, PointLight* l) {
	color_t color = CreateColor(0, 0, 0); 
	float t, smallest;
	int i = 0, closestSphere = -1, closestPlane = -1,  inShadow = false;
	//r.direction += r.origin; //Set back to normal
	Point normalVector;
	//FIND CLOSEST SPHERE ALONG RAY R
	while (i < NUM_SPHERES) {
		t = SphereRayIntersection(s + i, r);

		if (t > 0 && (closestSphere < 0 || t < smallest)) {
			smallest = t;
			closestSphere = i;
		}
		i++;
	}
	//r.direction -= r.origin;
	i=0;
	while (i < NUM_PLANES) {
		t = PlaneRayIntersection(f + i, r);
		if (t > 0 && ( (closestSphere < 0 && closestPlane < 0) || t < smallest)) {//POSSIBLE LOGIC FIX CLOSESTSPHERE >1
			smallest = t;
			closestSphere = -1;
			closestPlane = i;
		}
		i++;
	}

	//SETUP FOR SHADOW CALCULATIONS
	i = 0;
	Ray shadowRay;

	//r.direction += r.origin;//Smallest needs to be calculated differently
	shadowRay.origin = CreatePoint(r.direction.x * smallest, r.direction.y * smallest, r.direction.z * smallest);
	shadowRay.origin += r.origin;
	shadowRay.direction = l->position - shadowRay.origin;

	//DETERMINE IF SPHERE IS BLOCKING RAY FROM LIGHT TO SPHERE
	if(closestSphere > -1 || closestPlane > -1)
	{
		while (i <NUM_SPHERES-1 && !inShadow){ 
			t = SphereRayIntersection(s + i, shadowRay);
			if(i != closestSphere && t < 1 && t > 0){
				//	printf("%f\n",t);
				inShadow = true;
			}
			i++;
		}
		i = 0;
		while(i < NUM_PLANES && !inShadow){
			t = PlaneRayIntersection(f + i, shadowRay);
			if(i != closestPlane && t < 1 && t > 0){
				inShadow = true;
			}
			i++;
		}
	}


	//inShadow = false; 
	if(closestPlane > -1 && !inShadow)
	{
		//plane closer than sphere
		//normalVector = glm::normalize(f[closestPlane].normal-f[closestPlane].center);
		return Shading(r, shadowRay.origin, f[closestPlane].normal, l, f[closestPlane].diffuse,
				f[closestPlane].ambient,f[closestPlane].specular);
	}
	if(closestPlane > -1)
	{
		color.r = l->ambient.r * f[closestPlane].ambient.r;
		color.g = l->ambient.g * f[closestPlane].ambient.g;
		color.b = l->ambient.b * f[closestPlane].ambient.b;
		//return CreateColor(1,1,1);
		return color;
	}

	//IF SHADOWED, ONLY SHOW AMBIENT LIGHTING
	if(closestSphere > -1 && !inShadow)
	{

		normalVector = glm::normalize(shadowRay.origin-(s[closestSphere].center));
		return Shading(r, shadowRay.origin, normalVector, l, s[closestSphere].diffuse,
				s[closestSphere].ambient,s[closestSphere].specular);
	}
	if(closestSphere > -1)
	{
		color.r = l->ambient.r * s[closestSphere].ambient.r;
		color.g = l->ambient.g * s[closestSphere].ambient.g;
		color.b = l->ambient.b * s[closestSphere].ambient.b;
	}
	return color;
}

/*
 * Determines distance of intersection of Ray with Plane, -1 returned if no intersection
 */
__device__ float PlaneRayIntersection(Plane *p, Ray r)
{
	float t;
	//Point N = glm::normalize(p->normal);
	float denominator = glm::dot(r.direction,p->normal);
	if(denominator!=0)
	{
		t = (glm::dot(p->center-r.origin,p->normal)) / denominator;
		if (t>1000000)
			return -1;
		return t;
	}
	else
	{
		return -1;
	}
}


/*
 * Determines distance of intersection of Ray with Sphere, -1 returned if no intersection
 * http://sci.tuomastonteri.fi/programming/sse/example3
 */
__device__ float SphereRayIntersection(Sphere* s, Ray r) {
	float a, b, c, d, t1, t2;

	a = glm::dot((r.direction), (r.direction));

	b = glm::dot((r.origin)-(s->center),(r.direction));
	c = glm::dot((s->center),(s->center)) +glm::dot(r.origin,r.origin) -2.0f*glm::dot(r.origin, s->center)
		- (s->radius * s->radius);
	d = (b * b) - (a * c);

	if (d >= 0) {

		t1 = (-1 * b - sqrt(d)) / (a);
		t2 = (-1 * b + sqrt(d)) / (a);

		if (t2 > t1 && t1 > 0) {
			return t1;

		} else if (t2 > 0) {
			return t2;

		}
	}
	return -1;
}
/*
 * Determines Ambient, Diffuse, and Specular lighting on the plane
 */ 
__device__ color_t Shading(Ray r, Point p, Point normalVector,
		PointLight* l, color_t diffuse, color_t ambient, color_t specular) {
	color_t a, d, s, total;
	float NdotL, RdotV;
	Point viewVector, lightVector, reflectVector;

	viewVector = glm::normalize((r.origin)-p);

	lightVector = glm::normalize((l->position) -p);

	NdotL = glm::dot(lightVector, normalVector);
	reflectVector = (2.f *normalVector*NdotL) -lightVector;

	a.r = l->ambient.r * ambient.r;
	a.g = l->ambient.g * ambient.g;
	a.b = l->ambient.b * ambient.b;

	// Diffuse
	d.r = NdotL * l->diffuse.r * diffuse.r * (NdotL > 0);
	d.g = NdotL * l->diffuse.g * diffuse.g * (NdotL > 0);
	d.b = NdotL * l->diffuse.b * diffuse.b * (NdotL > 0);

	// Specular
	RdotV = glm::pow(glm::dot(glm::normalize(reflectVector), viewVector), 100.f);
	s.r = RdotV * l->specular.r * specular.r * (NdotL > 0) *(RdotV>0);
	s.g = RdotV * l->specular.g * specular.g * (NdotL > 0) *(RdotV>0);
	s.b = RdotV * l->specular.b * specular.b * (NdotL > 0) *(RdotV>0);

	total.r = glm::min(1.f, a.r + d.r + s.r);
	total.g = glm::min(1.f, a.g + d.g + s.g);
	total.b = glm::min(1.f, a.b + d.b + s.b);
	total.f = 1.f;
	return total;
}
