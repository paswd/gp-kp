// nvcc opengl.cu -lGL -lGLU -lGLEW -lglut

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../lib/cuPrintf.cu"

using namespace std;

const int32_t GLOBAL_WIDTH = 1024;
const int32_t GLOBAL_HEIGHT = 648;
const uint32_t VAR_COUNT = 2;
const uint32_t FUNC_MAXIMUMS_CNT = 4;
const uint32_t POINTS_COUNT = 2000;
const double SCALE_CHANGE_SPEED = 1.05;
const double PI = 3.1415926;
const double POINT_RADIUS = 10.;
const double DOUBLE_GEN_ACCURACY = 1000.;
const double POINTS_GEN_WIDTH = 10.;
const double POINTS_GEN_HEIGHT = 10.;

const double INERTIA = .5;
const double PARAM_A1 = .5;
const double PARAM_A2 = .5;

/*const double FUNC_A[FUNC_MAXIMUMS_CNT][VAR_COUNT] = {
	{2.54, 6.35},
	{7.56, 3.35},
	{7.35, 3.65}
};*/

dim3 blocks2D(32, 32), threads2D(16, 16);
dim3 blocks1D(1024), threads1D(256);

#define CSC(call) {														\
	 cudaError err = call;												\
	 if(err != cudaSuccess) {											\
		  fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
				__FILE__, __LINE__, cudaGetErrorString(err));			\
		  exit(1);														\
	 }																	\
} while (0)


const double 	xc = 0.0f,
				yc = 0.0f,
				sx = 5.0f,
				sy = sx * GLOBAL_HEIGHT / GLOBAL_WIDTH,
				FUNC_MIN = 0.,
				FUNC_MAX = .84;


struct Position {
	double X;
	double Y;
};

struct Point {
	Position Pos;

	double Angle;
	double Speed;
	double LocalMin;
	uchar4 Pixel;
	//bool PixelGradient;
};

struct GlobalData {
	Point *PointsArr;
	double Min;
};

GlobalData *GLOBAL;

__host__ double fRand(double fMin, double fMax)
{
    double f = (double) (rand() % (int32_t) fMax);
    return fMin + f * (fMax - fMin);
}

__host__ void setGlobalData() {
	CSC(cudaMalloc((void**) &GLOBAL, sizeof(GlobalData)));
	Point *tmpPointsArr;
	CSC(cudaMalloc((void**) &tmpPointsArr, sizeof(Point) * POINTS_COUNT));
	
	GlobalData globalData;
	globalData.PointsArr = tmpPointsArr;
	globalData.Min = 0;

	CSC(cudaMemcpy(GLOBAL, &globalData, sizeof(GlobalData), cudaMemcpyHostToDevice));

}

__host__ void destroyGlobalData() {
	GlobalData globalData;
	CSC(cudaMemcpy(&globalData, GLOBAL, sizeof(GlobalData), cudaMemcpyDeviceToHost));
	Point *tmpPointsArr = globalData.PointsArr;

	CSC(cudaFree(tmpPointsArr));
	CSC(cudaFree(GLOBAL));
}

__device__ double distance(double x1, double y1, double x2, double y2) {
	return sqrt(pow(abs(x1 - x2), 2.) + pow(abs(y1 - y2), 2.));
}

__device__ int32_t distance(int32_t x1, int32_t y1, int32_t x2, int32_t y2) {
	return (abs(x1 - x2) + abs(y1 - y2));
}
	
__device__ double fun(double x, double y, double t) {
	//return sin(x * x + t) + cos(y * y + t * 0.6) + sin(x * x + y * y + t * 0.3);
	//x /= 10.;
	//y /= 10.;
	double func_a[FUNC_MAXIMUMS_CNT][VAR_COUNT] = {
		{.054, 1.035},
		{3.956, .135},
		{.535, 1.065},
		{1.032, .121}
	};
	double summ = 0.;
	for (uint32_t i = 0; i < FUNC_MAXIMUMS_CNT; i++) {
		summ += 1. / (pow((x + t) - func_a[i][0], 2.) + pow((y + t) - func_a[i][1], 2.));
	}
	//summ = 2.5;
	//cout << summ << endl;
	//cuPrintf("%lf\n", summ);
	return summ;
}

__device__ double getCoordinateX(int32_t i, double scale) {
	return (2.0f * i / (double)(GLOBAL_WIDTH - 1) - 1.0f) * scale;
}

__device__ double getCoordinateY(int32_t j, double scale) {
	return (2.0f * j / (double)(GLOBAL_HEIGHT - 1) - 1.0f) * scale;
}

__device__ int32_t getPixelX(double x, double scale) {
	return (x / (2.0f * scale) + 0.5f) * (double)(GLOBAL_WIDTH - 1);
}

__device__ int32_t getPixelY(double y, double scale) {
	return (y / (2.0f * scale) + 0.5f) * (double)(GLOBAL_HEIGHT - 1);
}

__device__ double fun(int32_t i, int32_t j, double t, double scale)  {
	double x = getCoordinateX(i, scale);
	double y = getCoordinateY(j, scale);

	return fun(x * sx + xc, -y * sy + yc, t);	 
}

__device__ uchar4 get_color(float f) {
	//f /= 10;
	//f += .1;
	float k = 1.0 / 6.0;
	if (f < k)
		return make_uchar4((int)(f * 255 / k), 0, 0, 0);
	if (f < 2 * k)
		return make_uchar4(255, (int)((f - k) * 255 / k), 0, 0);
	if (f < 3 * k)
		return make_uchar4(255, 255, (int)((f - 2 * k) * 255 / k), 0);
	if (f < 4 * k)
		return make_uchar4(255 - (int)((f - 3 * k) * 255 / k), 255, 255, 0);
	if (f < 5 * k)
		return make_uchar4(0, 255 - (int)((f - 4 * k) * 255 / k), 255, 0);
	if (f < 6 * k)
		return make_uchar4(0, 0, 255 - (int)((f - 5 * k) * 255 / k), 0);
	return make_uchar4(0, 0, 0, 0);
}

__device__ uchar4 getPixel(double t) {
	return make_uchar4((int)(255 * cos(t + 2.)), (int)(255 * cos(t)), (int)(255 * sin(t)), 0);
}

/*__device__ uchar4 getPixel(GlobalData *Global, int32_t i, int32_t j, double f, double t, double scale) {
	for (int32_t n = 0; n < POINTS_COUNT; n++) {
		if (distance(getCoordinateX(i, scale), getCoordinateY(j, scale),
				Global->PointsArr[n].Pos.X, Global->PointsArr[n].Pos.Y) < POINT_RADIUS) {
			return getPixel(t);
		}
		if (distance(getCoordinateX(i, scale), getCoordinateY(j, scale),
				Global->PointsArr[n].Pos.X, Global->PointsArr[n].Pos.Y) < 2. * POINT_RADIUS) {
			return make_uchar4(0, 0, 0, 0);
		}
	}
	cuPrintf("UNIT %d :: %d\n", i, getPixelX(getCoordinateX(i, scale), scale));
	return get_color(f);
}*/


__global__ void drawMap(GlobalData *Global, uchar4* data, double t, double scale) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t offsetx = blockDim.x * gridDim.x;
	int32_t offsety = blockDim.y * gridDim.y;
	int32_t i, j;

	for(i = idx; i < GLOBAL_WIDTH; i += offsetx) {
		for(j = idy; j < GLOBAL_HEIGHT; j += offsety) {
			double f = (fun(i, j, t, scale) - FUNC_MIN) / (FUNC_MAX - FUNC_MIN);
			//data[j * GLOBAL_WIDTH + i] = getPixel(Global, i, j, f, t, scale);
			data[j * GLOBAL_WIDTH + i] = get_color(f); //make_uchar4(0, 0, (int)(f * 255), 255);
		}
	}
	__syncthreads();
}

__global__ void drawPoints(GlobalData *Global, uchar4 *data, double t, double scale) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t offsetx = blockDim.x * gridDim.x;
	int32_t i, j;

	for (int32_t n = idx; n < POINTS_COUNT; n += offsetx) {
		i = getPixelX(Global->PointsArr[n].Pos.X, scale);
		j = getPixelY(Global->PointsArr[n].Pos.Y, scale);

		if (i > 0 && j > 0 && i < GLOBAL_WIDTH && j < GLOBAL_HEIGHT) {
			data[j * GLOBAL_WIDTH + i] = getPixel(t);
		}
		//cuPrintf("Unit: %lf :: %lf :: %d\n", getCoordinateX(i, scale), Global->PointsArr[n].Pos.X, i);
	}
}

__device__ void generatePoint(Point *point, int32_t *rand_arr, int32_t num) {
	//curandCreateGenerator(&rand_gen ,CURAND_RNG_PSEUDO_DEFAULT);

	//point->Pos.X = cuRand() % GLOBAL_WIDTH;
	//point->Pos.Y = cuRand() % GLOBAL_HEIGHT;
	/*curandGenerateUniformDouble(rand_gen, &(point->Pos.X), 1);
	curandGenerateUniformDouble(rand_gen, &(point->Pos.Y), 1);
	curandGenerateUniformDouble(rand_gen, &(point->Angle), 1);*/
	

	point->Pos.X = (double)(rand_arr[3 * num]) / DOUBLE_GEN_ACCURACY * POINTS_GEN_WIDTH;
	point->Pos.Y = (double)(rand_arr[3 * num + 1]) / DOUBLE_GEN_ACCURACY * POINTS_GEN_HEIGHT;
	point->Angle = (double)(rand_arr[3 * num + 2]) / DOUBLE_GEN_ACCURACY * 2. * PI;

	if (rand_arr[3 * num] % 2) {
		point->Pos.X = -point->Pos.X;
	}
	if (rand_arr[3 * num + 1] % 2) {
		point->Pos.Y = -point->Pos.Y;
	}
	point->Speed = 1.;
	point->LocalMin = 0.;
	point->Pixel = make_uchar4(0, 255, 0, 0);


	//curandDestroyGenerator(rand_gen);
}

__global__ void setGlobalDataValues(GlobalData *Global, int32_t *rand_arr) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t offsetx = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < POINTS_COUNT; i += offsetx) {
		generatePoint(&(Global->PointsArr[i]), rand_arr, i);
	}
}

/*__device__ void changeParams(GlobalData *Global, int32_t n) {
	int32_t i = getPixelX(Global->PointsArr[n].Pos.X, scale);
	int32_t j = getPixelY(Global->PointsArr[n].Pos.Y, scale);

	Global
}
__global__ void movePoints(GlobalData *Global) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t offsetx = blockDim.x * gridDim.x;

	for (int32_t n = idx; n < POINTS_COUNT; n += offsetx) {
		changeParams(Global, n);
	}
}*/

__host__ void generateRandValues() {
	curandGenerator_t rand_gen;
	curandCreateGenerator(&rand_gen ,CURAND_RNG_PSEUDO_DEFAULT);
	curandDestroyGenerator(rand_gen);
	int32_t rand_arr[POINTS_COUNT * 3];
	int32_t *cuda_rand_arr;
	CSC(cudaMalloc((void**) &cuda_rand_arr, sizeof(int32_t) * POINTS_COUNT * 3));
	//curandGenerateUniformDouble(rand_gen, rand_arr, POINTS_COUNT * 3);
	for (uint32_t i = 0; i < POINTS_COUNT * 3; i++) {
		rand_arr[i] = rand() % (int32_t)DOUBLE_GEN_ACCURACY;
	}
	CSC(cudaMemcpy(cuda_rand_arr, rand_arr, sizeof(int32_t) * POINTS_COUNT * 3, cudaMemcpyHostToDevice));

	setGlobalDataValues<<<blocks1D, threads1D>>>(GLOBAL, cuda_rand_arr);
	CSC(cudaFree(cuda_rand_arr));
}

struct cudaGraphicsResource *res;
GLuint vbo;

double GLOBAL_SCALE = 1.;

void update() {
	static double t = 0.0;
	uchar4* dev_data;
	size_t size;
	CSC(cudaGraphicsMapResources(1, &res, 0));
	cudaPrintfInit();
	CSC(cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res));
	drawMap<<<blocks2D, threads2D>>>(GLOBAL, dev_data, t, GLOBAL_SCALE);
	drawPoints<<<blocks1D, threads1D>>>(GLOBAL, dev_data, t, GLOBAL_SCALE);
	cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
	CSC(cudaGetLastError());
	CSC(cudaGraphicsUnmapResources(1, &res, 0));
	glutPostRedisplay();
	t += 0.05;
}

void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(GLOBAL_WIDTH, GLOBAL_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);	
	glutSwapBuffers();
}

void keys(unsigned char key, int32_t x, int32_t y) {
	if (key == 27) {
		CSC(cudaGraphicsUnregisterResource(res));
		glBindBuffer(1, vbo);
		glDeleteBuffers(1, &vbo);
		destroyGlobalData();
		exit(0);
	}
	//cout << key << endl;
	if (key == '+') {
		GLOBAL_SCALE /= SCALE_CHANGE_SPEED;
		return;
	}
	if (key == '-') {
		GLOBAL_SCALE *= SCALE_CHANGE_SPEED;
		return;
	}
}


int main(int argc, char** argv) {
	
	setGlobalData();
	generateRandValues();


	glutInit(&argc, argv); 							
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);	
	glutInitWindowSize(GLOBAL_WIDTH, GLOBAL_HEIGHT);
	glutCreateWindow("Hot map");
	
	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(keys);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble) GLOBAL_WIDTH, 0.0, (GLdouble) GLOBAL_HEIGHT);

	glewInit();

	glGenBuffers(1, &vbo);								
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);		
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, GLOBAL_WIDTH * GLOBAL_HEIGHT * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

	CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));

	glutMainLoop();	
	return 0;
}