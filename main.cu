// nvcc opengl.cu -lGL -lGLU -lGLEW -lglut

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "../lib/cuPrintf.cu"

using namespace std;

const int32_t GLOBAL_WIDTH = 1024;
const int32_t GLOBAL_HEIGHT = 648;
const uint32_t VAR_COUNT = 2;
const uint32_t FUNC_MAXIMUMS_CNT = 4;
const uint32_t POINTS_COUNT = 500;
const double SCALE_CHANGE_SPEED = 1.05;

/*const double FUNC_A[FUNC_MAXIMUMS_CNT][VAR_COUNT] = {
	{2.54, 6.35},
	{7.56, 3.35},
	{7.35, 3.65}
};*/

dim3 blocks(32, 32), threads(16, 16);

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


struct TVector {
	double X;
	double Y;
};

struct Point {
	double Angle;
	double Speed;
	double LocalMin;
};

struct GlobalData {
	Point *PointsArr;
	double Min;
};

GlobalData *GLOBAL;

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

__device__ double fun(int32_t i, int32_t j, double t, double scale)  {
	double x = 2.0f * i / (double)(GLOBAL_WIDTH - 1) - 1.0f;
	double y = 2.0f * j / (double)(GLOBAL_HEIGHT - 1) - 1.0f;
	x *= scale;
	y *= scale;	
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

__global__ void kernel(GlobalData *Global, uchar4* data, double t, double scale) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	int32_t offsetx = blockDim.x * gridDim.x;
	int32_t offsety = blockDim.y * gridDim.y;
	int32_t i, j;

	for(i = idx; i < GLOBAL_WIDTH; i += offsetx)
		for(j = idy; j < GLOBAL_HEIGHT; j += offsety) {
			double f = (fun(i, j, t, scale) - FUNC_MIN) / (FUNC_MAX - FUNC_MIN);
			data[j * GLOBAL_WIDTH + i] = get_color(f);//make_uchar4(0, 0, (int)(f * 255), 255);
		}
}

__device__ void generatePoint(Point *point) {

}

__global__ void setGlobalDataValues(GlobalData *Global, Point *points_arr) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t offsetx = blockDim.x * gridDim.x;

	for (int32_t i = idx; i < POINTS_COUNT; i += offsetx) {
		generatePoint(&(Global->PointsArr[i]));
	}
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
	kernel<<<blocks, threads>>>(GLOBAL, dev_data, t, GLOBAL_SCALE);
	cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
	CSC(cudaGetLastError());
	CSC(cudaGraphicsUnmapResources(1, &res, 0));
	glutPostRedisplay();
	//t += 0.05;
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