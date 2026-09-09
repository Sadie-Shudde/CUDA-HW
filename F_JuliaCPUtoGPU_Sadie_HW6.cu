// Name: Sadie Shudde
// Simple Julia CPU.
// nvcc F_JuliaCPUtoGPU_Sadie_HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
 Use __device__ for the escapeOrNotColor function
*/

/*
 Purpose:
 To apply your new GPU skills to do  something cool!
*/

/*
 Explain what you did to fix the code:
 1. Lines 45-48 Added Global variables for block and grid size and pixel pointers for both CPU and GPU
 2. Lines 56-62 Added Function Prototypes
 3. Lines 79-87 Added setUpDevices()
 4. Lines 90-98 Added allocateMemory()
 5. Lines 101 Changed escapeOrNotColor() to __device__
 6. Lines 130-151 Added __gloabal__ colorPixels() to color one pixel this runs on GPU - PINK
 7. Lines 153-174 Changed display() function now calls GPU functions
 8. Lines 177-185 Added cleanUp function
 9. Lines 189, 193, 203 In main() just added the function calls needed
 More in depth comments can be found next to the altered code
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *Pixels_CPU; //CPU pointer
float *Pixels_GPU; //GPU pointer

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
void setUpDevices();
void allocateMemory();
__device__ float escapeOrNotColor(float, float);
__global__ void colorPixels(float*, float, float, float, float, int, int);//
void display(void);
void cleanUp();

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
// Kept as 1D. Each thread will figure out which row/column its id is
void setUpDevices()
{
	BlockSize.x = 1024;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1024; 
	GridSize.y = 1;
	GridSize.z = 1;
}

//Allocating the memory we will be using
void allocateMemory()
{
	//Hosst or CPU memory
	Pixels_CPU = (float*)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	
	//Device GPU memory
	cudaMalloc(&Pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

//Runs on the GPU. This will decide if an (x,y) point escapes to infinitiy or stays bounded
__device__ float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;
	
	int maxCount = MAXITERATIONS;
	float maxMag = MAXMAG;
	
	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < maxMag && count < maxCount) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		return(0.0);//is bounded
	}
	else
	{
		return(1.0);//goes to infinity
	}
}

//This is the kernal/function that runs on the GPU
__global__ void colorPixels(float *pixels, float xMin, float yMin, float stepSizeX, float stepSizeY, int width, int height)
{
	//Threads poistion in the list of pixels
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	//Convert 1D id into a 2D row//col
	int row = id/width; //Which row this pixel is in - how many full rows we have passed = which row the pixel is in
	int col = id%width; //which column this pixel is in - how far into the current row we are = which column
	//Convert the pixels row/col into an (x,y) ordered pair
	float x = xMin + col*stepSizeX;
	float y = yMin + row*stepSizeY;
	//Find this pixel's spot in the r,g,b array
	int k = 3 * (row*width+col); //Find this pixel's spot int the flat r,g,b array
	
	//1.0 if ecaped 0.0 if not
	float color = escapeOrNotColor(x,y);
	
	//PINK
	pixels[k] = color*1.0; //Red
	pixels[k+1] = color*0.4; //Green
	pixels[k+2] = color*0.7; //Blue
}

void display(void) 
{ 
	float stepSizeX, stepSizeY;
	
	stepSizeX = (XMax-XMin)/((float)WindowWidth);
	stepSizeY = (YMax-YMin)/((float)WindowHeight);
	
	//Coloring the pixels on the GPU
	colorPixels<<<GridSize, BlockSize>>>(Pixels_GPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copy Memory from GPU to CPU
	cudaMemcpy(Pixels_CPU, Pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	
	//Making sure the GPU and CPU wait until each other are at the same place
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, Pixels_CPU); 
	glFlush(); 
}

// Cleaning up memory after we are finished.
void cleanUp()
{
	// Freeing host "CPU" memory.
	free(Pixels_CPU); 

	// Free divice "GPU" memory.
	cudaFree(Pixels_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
}

int main(int argc, char** argv)
{ 
	//Setting up the GPU
	setUpDevices();
	
	//Allocating the memory you will need
	allocateMemory();
	
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
   	
   	
   	cleanUp();
   	return(0);
}

