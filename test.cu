// nvcc opengl.cu -lGL -lGLU -lGLEW -lglut

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

const int w = 1024;
const int h = 648;

dim3 blocks(32, 32), threads(16, 16);

#define CSC(call) {														\
	 cudaError err = call;												\
	 if(err != cudaSuccess) {											\
		  fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
				__FILE__, __LINE__, cudaGetErrorString(err));			\
		  exit(1);														\
	 }																	\
} while (0)


const double xc = 0.0f, yc = 0.0f, sx = 5.0f, sy = sx * h / w, minf = -3.0, maxf = 3.0;

__device__ double fun(double x, double y, double t) {
	return sin(x * x + t) + cos(y * y + t * 0.6) + sin(x * x + y * y + t * 0.3);
}

__device__ double fun(int i, int j, double t)  {
	double x = 2.0f * i / (double)(w - 1) - 1.0f;
	double y = 2.0f * j / (double)(h - 1) - 1.0f;	
	return fun(x * sx + xc, -y * sy + yc, t);	 
}

__device__ uchar4 get_color(float f) {
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

__global__ void kernel(uchar4* data, double t) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int i, j;
	for(i = idx; i < w; i += offsetx)
		for(j = idy; j < h; j += offsety) {
			double f = (fun(i, j, t) - minf) / (maxf - minf);
			data[j * w + i] = get_color(f);//make_uchar4(0, 0, (int)(f * 255), 255);
		}
}

struct cudaGraphicsResource *res;
GLuint vbo;

void update() {
	static double t = 0.0;
	uchar4* dev_data;
	size_t size;
	CSC(cudaGraphicsMapResources(1, &res, 0));
	CSC(cudaGraphicsResourceGetMappedPointer((void**) &dev_data, &size, res));
	kernel<<<blocks, threads>>>(dev_data, t);
	CSC(cudaGetLastError());
	CSC(cudaGraphicsUnmapResources(1, &res, 0));
	glutPostRedisplay();
	t += 0.05;
}

void display() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);	
	glutSwapBuffers();
}

void keys(unsigned char key, int x, int y) {
	if (key == 27) {
		CSC(cudaGraphicsUnregisterResource(res));
		glBindBuffer(1, vbo);
		glDeleteBuffers(1, &vbo);
		exit(0);
	}
}


int main(int argc, char** argv) {
	glutInit(&argc, argv); 							
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);	
	glutInitWindowSize(w, h);
	glutCreateWindow("Hot map");
	
	glutIdleFunc(update);
	glutDisplayFunc(display);
	glutKeyboardFunc(keys);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble) w, 0.0, (GLdouble) h);

	glewInit();

	glGenBuffers(1, &vbo);								
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vbo);		
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);

	CSC(cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsMapFlagsWriteDiscard));

	glutMainLoop();	
	return 0;
}