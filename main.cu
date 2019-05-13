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
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include "../lib/cuPrintf.cu"

using namespace std;

const int32_t GLOBAL_WIDTH = 1024;
const int32_t GLOBAL_HEIGHT = 648;
const uint32_t VAR_COUNT = 2;
const uint32_t FUNC_MAXIMUMS_CNT = 3 ;
const uint32_t FUNC_MAXIMUMS_CNT_LIM = 5;
const uint32_t POINTS_COUNT = 500.;
const double SCALE_CHANGE_SPEED = 1.05;
const double PI = 3.1415926;
const double EPS = .00001;
const double DOUBLE_GEN_ACCURACY = 1000.;
const double POINTS_GEN_WIDTH = 10.;
const double POINTS_GEN_HEIGHT = 10.;
const double GRAVITY_PARAM = 10.;
const double DIST_LIM = 10.;
const double MOVING_PARAM_X = 5.;
const double MOVING_PARAM_Y = 5.;

const double INERTIA = .08;
const double PARAM_A_GLOBAL = .3;
const double PARAM_A_LOCAL = .2;

const double SHIFT_SPEED_X = .5;
const double SHIFT_SPEED_Y = .5;

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


struct Comparator {
	__host__ __device__ bool operator()(double a, double b) {
		return a < b;
	}
};

struct Position {
	double X;
	double Y;
};

__device__ __host__ void setPosition(Position *pos, double x, double y) {
	pos->X = x;
	pos->Y = y;
}

struct Point {
	Position Pos;

	double Angle;
	double Speed;
	double LocalMin;
	Position LocalMinPos;
	uchar4 Pixel;
	//bool PixelGradient;
};

struct GlobalData {
	Point *PointsArr;
	double Min;
	Position MinPos;
	double PointSelectCoeff;
	Position CurrCenter;
};

GlobalData *GLOBAL;
double *MAX_ARR;
double *POS_X;
double *POS_Y;

bool GLOBAL_MOVE = true;
bool MOVE_FUNCTION = false;

__host__ double fRand(double fMin, double fMax)
{
    double f = (double) (rand() % (int32_t) fMax);
    return fMin + f * (fMax - fMin);
}

__host__ void setGlobalData() {
	CSC(cudaMalloc((void**) &GLOBAL, sizeof(GlobalData)));
	CSC(cudaMalloc((void**) &MAX_ARR, sizeof(double) * POINTS_COUNT));
	CSC(cudaMalloc((void**) &POS_X, sizeof(double) * POINTS_COUNT));
	CSC(cudaMalloc((void**) &POS_Y, sizeof(double) * POINTS_COUNT));
	CSC(cudaMemset(MAX_ARR, 0., sizeof(double) * POINTS_COUNT));

	Point *tmpPointsArr;
	CSC(cudaMalloc((void**) &tmpPointsArr, sizeof(Point) * POINTS_COUNT));
	
	GlobalData globalData;
	globalData.PointsArr = tmpPointsArr;
	globalData.Min = 0.;
	globalData.CurrCenter.X = 0.;
	globalData.CurrCenter.Y = 0.;

	CSC(cudaMemcpy(GLOBAL, &globalData, sizeof(GlobalData), cudaMemcpyHostToDevice));
	//CURR_CENTER.X = 0.;
	//CURR_CENTER.Y = 0.;

}

__host__ void destroyGlobalData() {
	GlobalData globalData;
	CSC(cudaMemcpy(&globalData, GLOBAL, sizeof(GlobalData), cudaMemcpyDeviceToHost));
	Point *tmpPointsArr = globalData.PointsArr;

	CSC(cudaFree(tmpPointsArr));
	CSC(cudaFree(GLOBAL));
	CSC(cudaFree(MAX_ARR));
	CSC(cudaFree(POS_X));
	CSC(cudaFree(POS_Y));
}

__device__ __host__ double distance(double x1, double y1, double x2, double y2) {
	return sqrt(pow(abs(x1 - x2), 2.) + pow(abs(y1 - y2), 2.));
}

__device__ __host__ int32_t distance(int32_t x1, int32_t y1, int32_t x2, int32_t y2) {
	return (abs(x1 - x2) + abs(y1 - y2));
}
	
__device__ __host__ double func(double x, double y, double t) {
	/*if (x < EPS && y < EPS) {
		x += EPS;
		y += EPS;
	}*/
	//return sin(x * x + t) + cos(y * y + t * 0.6) + sin(x * x + y * y + t * 0.3);
	//x /= 10.;
	//y /= 10.;
	double func_a[FUNC_MAXIMUMS_CNT_LIM][VAR_COUNT] = {
		//{0., 0.},
		{.054, 1.035},
		{3.956, .135},
		{.535, 1.065},
		{1.032, .121},
		{1.032, .121}
	};
	double summ = 0.;
	for (uint32_t i = 0; i < FUNC_MAXIMUMS_CNT; i++) {
		if (x - func_a[i][0] < EPS) {
			x += EPS;
		}
		if (y - func_a[i][1] < EPS) {
			y += EPS;
		}
		summ += 1. / (pow(x + MOVING_PARAM_X * cos(t * SHIFT_SPEED_X) - func_a[i][0], 2.) +
			pow(y + MOVING_PARAM_Y * sin(t * SHIFT_SPEED_Y) - func_a[i][1], 2.));
	}
	//summ = 2.5;
	//cout << summ << endl;
	//cuPrintf("%lf\n", summ);
	return summ;
}

/*__host__ double funcDbg(double x, double y, double t) {
	
	//return sin(x * x + t) + cos(y * y + t * 0.6) + sin(x * x + y * y + t * 0.3);
	//x /= 10.;
	//y /= 10.;
	double func_a[FUNC_MAXIMUMS_CNT_LIM][VAR_COUNT] = {
		//{0., 0.},
		{.054, 1.035},
		{3.956, .135},
		{.535, 1.065},
		{1.032, .121}
	};
	double summ = 0.;
	cout << "X = " << x << endl;
	cout << "XMod = " << (x - t * SHIFT_SPEED_X) << endl;
	cout << "XPrePow = " << (x - t * SHIFT_SPEED_X) - func_a[0][0] << endl;
	cout << "XPow = " << pow((x - t * SHIFT_SPEED_X) - func_a[0][0], 2.) << endl;
	cout << "Y = " << y << endl;
	cout << "YMod = " << (y - t * SHIFT_SPEED_Y) << endl;
	cout << "YPrePow = " << (y - t * SHIFT_SPEED_Y) - func_a[0][1] << endl;
	cout << "YPow = " << pow((y - t * SHIFT_SPEED_Y) - func_a[0][1], 2.) << endl;
	for (uint32_t i = 0; i < FUNC_MAXIMUMS_CNT; i++) {
		summ += 1. / (pow((x - t * SHIFT_SPEED_X) - func_a[i][0], 2.) + pow((y + t * SHIFT_SPEED_Y) - func_a[i][1], 2.));
	}
	//summ = 2.5;
	//cout << summ << endl;
	//cuPrintf("%lf\n", summ);
	return summ;
}*/

__device__ __host__ double getCoordinateX(int32_t i, double scale, Position shift) {
	return (2.0f * i / (double)(GLOBAL_WIDTH - 1) - 1.0f) * scale * sx + shift.X;
}

__device__ __host__ double getCoordinateY(int32_t j, double scale, Position shift) {
	return (2.0f * j / (double)(GLOBAL_HEIGHT - 1) - 1.0f) * scale * sy - shift.Y;
}

__device__ __host__ int32_t getPixelX(double x, double scale, Position shift) {
	return ((x - shift.X) / (2.0f * scale * sx) + 0.5f) * (double)(GLOBAL_WIDTH - 1);
}

__device__ __host__ int32_t getPixelY(double y, double scale, Position shift) {
	return ((y + shift.Y) / (2.0f * scale * sy) + 0.5f) * (double)(GLOBAL_HEIGHT - 1);
}

__device__ __host__ double func(int32_t i, int32_t j, double t, double scale, Position shift)  {
	return func(
		getCoordinateX(i, scale, shift),
		getCoordinateY(j, scale, shift),
		t);	 
}

__device__ __host__ bool isVisible(int32_t i, int32_t j) {
	return i > 0 && j > 0 && i < GLOBAL_WIDTH && j < GLOBAL_HEIGHT;
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

	for (i = idx; i < GLOBAL_WIDTH; i += offsetx) {
		for (j = idy; j < GLOBAL_HEIGHT; j += offsety) {
			double f = (func(i, j, t, scale, Global->CurrCenter) - FUNC_MIN) / (FUNC_MAX - FUNC_MIN);
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

	/*if (idx == 0) {
		int32_t i0 = getPixelX(0., scale);
		int32_t j0 = getPixelY(0., scale);
		int32_t i1 = getPixelX(1., scale);
		int32_t j1 = getPixelY(1., scale);

		data[j0 * GLOBAL_WIDTH + i0] = getPixel(t);
		data[j1 * GLOBAL_WIDTH + i1] = getPixel(t);
	}*/

	for (int32_t n = idx; n < POINTS_COUNT; n += offsetx) {
		i = getPixelX(Global->PointsArr[n].Pos.X, scale, Global->CurrCenter);
		j = getPixelY(Global->PointsArr[n].Pos.Y, scale, Global->CurrCenter);

		if (isVisible(i, j)) {
			data[j * GLOBAL_WIDTH + i] = getPixel(t);
		}
		//cuPrintf("Unit: %lf :: %lf :: %d\n", getCoordinateX(i, scale), Global->PointsArr[n].Pos.X, i);
	}

	__syncthreads();
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
	point->LocalMinPos.X = 0.;
	point->LocalMinPos.Y = 0.;
	point->Pixel = make_uchar4(0, 255, 0, 0);


	//curandDestroyGenerator(rand_gen);
}

__global__ void setGlobalDataValues(GlobalData *Global, int32_t *rand_arr) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t offsetx = blockDim.x * gridDim.x;

	Global->PointSelectCoeff = rand_arr[0] / DOUBLE_GEN_ACCURACY;
	Global->MinPos.X = 0.;
	Global->MinPos.Y = 0.;

	for (int32_t i = idx; i < POINTS_COUNT; i += offsetx) {
		generatePoint(&(Global->PointsArr[i]), rand_arr, i);
	}
}

__device__ void calculateLocalMin(GlobalData *Global, int32_t n, double t, double scale, double *max_arr) {
	int32_t i = getPixelX(Global->PointsArr[n].Pos.X, scale, Global->CurrCenter);
	int32_t j = getPixelY(Global->PointsArr[n].Pos.Y, scale, Global->CurrCenter);
	double x = Global->PointsArr[n].Pos.X;
	double y = Global->PointsArr[n].Pos.Y;

	//if (func(i, j, t, scale))
	Global->PointsArr[n].LocalMin *= INERTIA;
	max_arr[n] *= INERTIA;
	double funcRes = func(i, j, t, scale, Global->CurrCenter);
	if (funcRes >= Global->PointsArr[n].LocalMin) {
		Global->PointsArr[n].LocalMin = funcRes;
		max_arr[n] = funcRes;
		Global->PointsArr[n].LocalMinPos.X = x;
		Global->PointsArr[n].LocalMinPos.Y = y;

	}
}

__device__ void getCollisionVector(GlobalData *Global, Position *res, int32_t n) {
	res->X = 0.;
	res->Y = 0.;
	//Position curr;
	for (int32_t i = 0; i < n; i++) {
		if (i == n) {
			continue;
		}
		double dist = distance(Global->PointsArr[i].Pos.X, Global->PointsArr[i].Pos.Y,
			Global->PointsArr[n].Pos.X, Global->PointsArr[n].Pos.Y);
		/*curr.X = -(Global->PointsArr[i].Pos.X - Global->PointsArr[n].Pos.X) / pow(dist, 4.);
		curr.Y = -(Global->PointsArr[i].Pos.Y - Global->PointsArr[n].Pos.Y) / pow(dist, 4.);

		curr.X /= pow(dist, 4.);
		curr.Y /= pow(dist, 4.);*/

		res->X += -(Global->PointsArr[i].Pos.X - Global->PointsArr[n].Pos.X) / pow(dist, 4.);
		res->Y += -(Global->PointsArr[i].Pos.Y - Global->PointsArr[n].Pos.Y) / pow(dist, 4.);
	}
	res->X /= (double)(POINTS_COUNT - 1) * GRAVITY_PARAM;
	res->Y /= (double)(POINTS_COUNT - 1) * GRAVITY_PARAM;

	__syncthreads();
}

__device__ void changeParams(GlobalData *Global, int32_t n, double t, double scale) {
	double x = Global->PointsArr[n].Pos.X;
	double y = Global->PointsArr[n].Pos.Y;

	/*if (Global->PointsArr[n].LocalMin < Global->Min) {
		Global->Min = Global->PointsArr[n].LocalMin;
		Global->MinPos = Global->PointsArr[n].LocalMinPos;
	}*/

	Global->PointsArr[n].Speed = Global->PointsArr[n].Speed * INERTIA +
		PARAM_A_LOCAL * Global->PointSelectCoeff * distance(x, y,
			Global->PointsArr[n].LocalMinPos.X, Global->PointsArr[n].LocalMinPos.Y) +
		PARAM_A_GLOBAL * (1. - Global->PointSelectCoeff) * distance(x, y,
			Global->MinPos.X, Global->MinPos.Y);
	/*Global->PointsArr[n].Speed = Global->PointsArr[n].Speed * INERTIA +
		PARAM_A_LOCAL * 0. * distance(x, y,
			Global->PointsArr[n].LocalMinPos.X, Global->PointsArr[n].LocalMinPos.Y) +
		PARAM_A_GLOBAL * (1. - 0.) * distance(x, y,
			Global->MinPos.X, Global->MinPos.Y);*/

	Position currLocalMin, currGlobalMin, resPos, collision;
	setPosition(&currLocalMin, Global->PointsArr[n].LocalMinPos.X - x,
		Global->PointsArr[n].LocalMinPos.Y - y);
	setPosition(&currGlobalMin, Global->MinPos.X - x,
		Global->MinPos.Y - y);

	resPos.X = PARAM_A_LOCAL * Global->PointSelectCoeff * currLocalMin.X +
		PARAM_A_GLOBAL * (1. - Global->PointSelectCoeff) * currGlobalMin.X;
	resPos.Y = PARAM_A_LOCAL * Global->PointSelectCoeff * currLocalMin.Y +
		PARAM_A_GLOBAL * (1. - Global->PointSelectCoeff) * currGlobalMin.Y;

	double dist = distance(x, y, resPos.X, resPos.Y);
	resPos.X = resPos.X / dist * Global->PointsArr[n].Speed;
	resPos.Y = resPos.Y / dist * Global->PointsArr[n].Speed;

	getCollisionVector(Global, &collision, n);

	Global->PointsArr[n].Pos.X += resPos.X + collision.X;
	Global->PointsArr[n].Pos.Y += resPos.Y + collision.Y;
	
	//cuPrintf("Local: %lf ~ %lf : %lf\n", Global->PointsArr[n].LocalMin, Global->PointsArr[n].LocalMinPos.X, Global->PointsArr[n].LocalMinPos.Y);
}

__global__ void calculateLocalMinimums(GlobalData *Global, double t, double scale, double *max_arr) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t offsetx = blockDim.x * gridDim.x;

	for (int32_t n = idx; n < POINTS_COUNT; n += offsetx) {
		calculateLocalMin(Global, n, t, scale, max_arr);
	}
	__syncthreads();
	//cuPrintf("Global:\n");
}

__global__ void movePoints(GlobalData *Global, double t, double scale, double *max_arr, int32_t max_pos,
		double *pos_x, double *pos_y) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t offsetx = blockDim.x * gridDim.x;

	if (idx == 0) {
		Global->Min *= INERTIA;
	}

	if (idx == 0 && max_arr[max_pos] > Global->Min) {
		Global->Min = max_arr[max_pos];
		Global->MinPos.X = Global->PointsArr[max_pos].Pos.X;
		Global->MinPos.Y = Global->PointsArr[max_pos].Pos.Y;
	}

	__syncthreads();

	for (int32_t n = idx; n < POINTS_COUNT; n += offsetx) {
		if (distance(
				Global->PointsArr[n].Pos.X, Global->PointsArr[n].Pos.Y,
				Global->CurrCenter.X, Global->CurrCenter.Y) < DIST_LIM) {
			pos_x[n] = Global->PointsArr[n].Pos.X;
			pos_y[n] = Global->PointsArr[n].Pos.Y;
		} else {
			pos_x[n] = 0.;
			pos_y[n] = 0.;
		}
		changeParams(Global, n, t, scale);
	}
	//cuPrintf("Global:\n");
}

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
	//CSC(cudaFree(cuda_rand_arr));
}

struct cudaGraphicsResource *res;
GLuint vbo;

double GLOBAL_SCALE = 1.;

__host__ int32_t findGlobalMaximum() {
	Comparator cmp;
	thrust::device_ptr <double> begin = thrust::device_pointer_cast(MAX_ARR);
	thrust::device_ptr <double> max = thrust::max_element(
		begin,
		begin + POINTS_COUNT, cmp);
	return max - begin;
}

__global__ void calculateSumm(double *arr) {
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t offsetx = blockDim.x * gridDim.x;

	for (int32_t width = 2; width < POINTS_COUNT; width *= 2) {
		for (int32_t i = idx * width; i < POINTS_COUNT; i += offsetx * width) {
			if (i + width / 2 < POINTS_COUNT) {
				arr[i] += arr[i + width / 2];
			}
		}
		__syncthreads();
	}
}
__host__ void calculateSummTest(double *arr) {
	int32_t idx = 0;
	int32_t offsetx = 1;

	cout << "START" << endl;
	for (int32_t width = 2; width < POINTS_COUNT; width *= 2) {
		for (int32_t id = 0; id < POINTS_COUNT; id++) {
			cout << arr[id] << " ";
		}
		cout << endl;
		for (int32_t i = idx * width; i < POINTS_COUNT; i += offsetx * width) {
			if (i + width / 2 < POINTS_COUNT) {
				cout << i << " <- " << i + width / 2 << endl;
				cout << arr[i] << " + " << arr[i + width / 2];
				arr[i] += arr[i + width / 2];
				cout << " = " << arr[i] << endl;
				cout << "------------------" << endl;
			}
		}
		cout << "===================" << endl;
	}
	cout << "END" << endl << endl;
}

void updateCenter() {
	//thrust::device_ptr <double> begin_x = thrust::device_pointer_cast(POS_X);
	//thrust::device_ptr <double> begin_y = thrust::device_pointer_cast(POS_Y);
	//calculateSumm<<<blocks1D, threads1D>>>(POS_X);
	//calculateSumm<<<blocks1D, threads1D>>>(POS_Y);
	/*double pos_x[POINTS_COUNT];
	double pos_y[POINTS_COUNT];

	CSC(cudaMemcpy(pos_x, POS_X, sizeof(double) * POINTS_COUNT, cudaMemcpyDeviceToHost));
	CSC(cudaMemcpy(pos_y, POS_Y, sizeof(double) * POINTS_COUNT, cudaMemcpyDeviceToHost));*/

	/*calculateSummTest(pos_x);
	calculateSummTest(pos_y);*/
	cout << "TEST" << endl;
	thrust::device_vector<double> vect_x(POS_X, POS_X + POINTS_COUNT);
	thrust::device_vector<double> vect_y(POS_Y, POS_Y + POINTS_COUNT);
	double summ_x = thrust::reduce(vect_x.begin(), vect_x.end());
	double summ_y = thrust::reduce(vect_y.begin(), vect_y.end());
	//summ_x = 0.;
	//summ_y = 0.;
	//cout << summ_x << " " << summ_y << endl;
	//CSC(cudaMemcpy(&summ_x, POS_X, sizeof(double), cudaMemcpyDeviceToHost));
	//CSC(cudaMemcpy(&summ_y, POS_Y, sizeof(double), cudaMemcpyDeviceToHost));

	Position new_center;
	setPosition(
		&new_center,
		summ_x / POINTS_COUNT,
		summ_y / POINTS_COUNT
		);
	cout << new_center.X << " : " << new_center.Y << endl;
	GlobalData globalData;
	CSC(cudaMemcpy(&globalData, GLOBAL, sizeof(GlobalData), cudaMemcpyDeviceToHost));
	if (!isVisible(
		getPixelX(new_center.X, GLOBAL_SCALE, globalData.CurrCenter),
		getPixelY(new_center.Y, GLOBAL_SCALE, globalData.CurrCenter)
		)) {
		globalData.CurrCenter = new_center;
		CSC(cudaMemcpy(GLOBAL, &globalData, sizeof(GlobalData), cudaMemcpyHostToDevice));
		cout << "INVISIBLE" << endl;
	} else {
		cout << "  VISIBLE" << endl;
	}
}

void Test() {
	GlobalData globalData;
	CSC(cudaMemcpy(&globalData, GLOBAL, sizeof(GlobalData), cudaMemcpyDeviceToHost));
	Point *tmpPointsArr = globalData.PointsArr;
	Point pointArr[POINTS_COUNT];

	CSC(cudaMemcpy(pointArr, tmpPointsArr, sizeof(Point) * POINTS_COUNT, cudaMemcpyDeviceToHost));

	//double localMax[POINTS_COUNT];

	//CSC(cudaMemcpy(localMax, MAX_ARR, sizeof(double) * POINTS_COUNT, cudaMemcpyDeviceToHost));
	cout << "GLOBAL: " << globalData.Min << " ~ " << globalData.MinPos.X << " : " << globalData.MinPos.Y << endl;
	for (int32_t i = 0; i < POINTS_COUNT; i++) {
		cout << pointArr[i].Pos.X << " :: " << pointArr[i].Pos.Y << " - " << func(pointArr[i].Pos.X, pointArr[i].Pos.Y, 0)<< " ~~~ " << pointArr[i].LocalMinPos.X << "\t" << pointArr[i].LocalMinPos.Y << "\t" <<
			distance(pointArr[i].LocalMinPos.X, pointArr[i].LocalMinPos.Y, 0., 0.) << "\t" << pointArr[i].LocalMin << endl;
	}
}

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
	if (GLOBAL_MOVE) {
		if (MOVE_FUNCTION) {
			t += 0.05;
		}
		calculateLocalMinimums<<<blocks1D, threads1D>>>(GLOBAL, t, GLOBAL_SCALE, MAX_ARR);
		//Test();
		uint32_t max_pos = findGlobalMaximum();
		movePoints<<<blocks1D, threads1D>>>(GLOBAL, t, GLOBAL_SCALE, MAX_ARR, max_pos, POS_X, POS_Y);
		updateCenter();
	}
	GlobalData globalData;
	CSC(cudaMemcpy(&globalData, GLOBAL, sizeof(GlobalData), cudaMemcpyDeviceToHost));
	//cout << globalData.Min << " ~ " << globalData.MinPos.X << " : " << globalData.MinPos.Y << endl; 
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
	if (key == 't') {
		Test();
		return;
	}
	if (key == 'p') {
		GLOBAL_MOVE = !GLOBAL_MOVE;
		return;
	}
	if (key == 'm') {
		MOVE_FUNCTION = !MOVE_FUNCTION;
		return;
	}
	//cout << (int) key << endl;
}


int main(int argc, char** argv) {
	//srand(time(NULL));
	//cout <<  "MAX = " << func(.054, 1.035, 0.) << " ~ " << .054 << " : " << 1.035 << endl;
	/*int n;
	cin >> n;*/
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