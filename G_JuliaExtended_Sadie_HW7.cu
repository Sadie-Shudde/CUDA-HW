// Name: Sadie Shudde
// Not simple Julia Set on the GPU
// nvcc G_JuliaExtended_Sadie_HW7_Art_Move.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 However, it currently only runs on a 1024x1024 window.

 Your tasks:
 - Modify the code so it works on any given window size. 
   I will pick these on the fly unsigned int WindowWidth, WindowHeight; 
   float XMin, XMax, YMin, YMax; and your code should work. You will be graded on this.
   
 - But you can set these values to whatever you want for the art compitition.
 - Add color to the fractal — be creative! You will be judged on your artistic flair.
 - Don't cut off your ear or anything, but try to make Vincent wish he'd had a GPU.
 - This is a competition with a prize!!!
*/

/*
 Purpose:
 To have some fun with your new GPU skills!
*/

/*
 Explain what you did to fix the code:
 Dr. Wyatt I would apologize in advance for how over the top I went; however, I know that neither of us are surprised and so therefore I will not be offering an apology. 
 You at least get to look at the pretty picture and did not have to sit here for hours trying to figure out how stuff works. 
 Here is the original picture I found I blame this creator https://www.shadertoy.com/view/WdtBRS?ref=makerforce.io for my insanity. Enjoy code is also commented below
 
 WINDOW SIZE
 1. Kept thread ID 1D using 2D blocks and grids with rows and cols like last week
 2. id and buffer offset are kept separate as id and k = 3*id so multiplying by 3 does not corrupt the row and col math
 3. blockSize.x is queried from the GPU instead of hardcoded so it adapts to whatever device it runs on
 4. gridSize is computed from width*height directly with a check on if(id<totalPixels) so it works even when it doesnt nicely divide into blocks
 5. XMin/XMax get scaled by aspect ratio so the fractal isn't stretched or cut on non square screens
 
 Orbit
 1. Include math.h library
 2. Get rid of MAXMAG and replace the other #define values
 3. Add trapDist() function to find distance between 2 points will come into play for orbiter function
 4. Replace escapeOrNotColor() with juliaOrbitColor() this function will color a point depending on how long it spent near a certain fixed point - Yes I know this is overly complicated
 
 Movement
 1. Added a Time variable that will increase a little each frame and moved Pixels_CPU/GPU to global scope and added allocateMemory() so the memory isn't reallocated 60 times a second
 2. Added update() as a GLUT timer callback. It increments Time and calls glutPostRedisplay() to trigger a redraw. It then reschedules itself every 16ms 
 3. Passed Time into the kernal function where it drives a small oscillation added to C's real aprt. This slight wobble in C is what makes the whole fractal shape morph continusously
 
 Once again no apology because this is cool and pretty! And if you hate it, Hayden made me do it.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <math.h>
// Defines
#define MAXITERATIONS 500 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.77568377 //Real part of C
#define B   0.134646737 //Imaginary part of C

// Global variables
unsigned int WindowWidth = 1600; //For art set to 1600
unsigned int WindowHeight = 900; //For art set to 900
float aspect = (float)WindowWidth/(float)WindowHeight;// make sure it fits in the screen
float XMin = -0.9*aspect; //For art set all to 0.95 with respective + - signs
float XMax =  0.9*aspect;
float YMin = -0.9;
float YMax =  0.9;

//Getting it to move is I know this is over the top I am not apologizing
float Time = 0.0f; // animation clock
float *Pixels_CPU;
float *Pixels_GPU;

// Function prototypes
void cudaErrorCheck(const char*, int);
__device__ float trapDist(float, float, float, float);
__device__ void juliaOrbitColor(float, float, float, float, float*, float*, float*);
void allocateMemory();
__global__ void colorPixels(float, float, float, float, float, int, int, float);
void display(void);
void update(int);

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
//Allocate memory here now to get it to move
void allocateMemory()
{
	Pixels_CPU = (float*)malloc(WindowWidth*WindowHeight*3*sizeof(float));
	cudaMalloc(&Pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}
//Distance formula to help in orbit color
__device__ float trapDist(float ax, float ay, float bx, float by)
{
	float dx = ax-bx;
	float dy = ay-by;
	return sqrtf(dx*dx+dy*dy);
}

__device__ void juliaOrbitColor(float x, float y, float cx, float cy, float *r, float *g, float *b)
{
	//Colors assigned to each trap point copied from reference 
	const float col1r = 1.0f, col1g = 0.3f, col1b = 0.4f; //Pink color
	const float col2r = 0.4f, col2g = 1.0f, col2b = 0.2f; //Green color
	const float col3r = 0.3f, col3g = 0.4f, col3b = 1.0f; //Blue color
	
	//These track the closet the orbit ever gets to each trap point across all iterations;
	//Just start them at a huge number so that the first distance will be smaller
	float mdist1 = 1e20f, mdist2 = 1e20f, mdist3 = 1e20f;
	
	//We will need these values after the loop ends
	for(int i = 0; i< MAXITERATIONS; i++)
	{
		if(x*x+y*y >4.0f) break; 
		//Actual Julia set formula z = z^2+c
		//(x,y) is z as a complex number squaring and adding c creates fractal structure
		//This was in escapeOrNotColor just a bit different
		float newX = x*x-y*y +cx;
		float newY = 2.0f*x*y +cy;
		x = newX;
		y = newY;
		//After updating point check how close it is to trap points
		float d1 = trapDist(x, y, 0.0f, 0.0f);
		float d2 = trapDist(x, y, 0.0f, 0.5f);
		float d3 = trapDist(x, y, 0.5f, 0.0f);
		//keep only the smallest distance
		if(d1 <mdist1) mdist1 = d1;
		if(d2 <mdist2) mdist2 = d2;
		if(d3 <mdist3) mdist3 = d3;
	}

	//Blend the three colors together weighted by how close the orbit got to each trap point
	float rr = mdist1*col1r + mdist2*col2r + mdist3*col3r;
	float gg = mdist1*col1g + mdist2*col2g + mdist3*col3g;
	float bb = mdist1*col1b + mdist2*col2b + mdist3*col3b;
	
	//Squaring pushes the small values even smaller and larger valeus less so -> increases contrast between areas that got close to trap points
	rr *= rr;
	gg *= gg;
	bb *= bb;
	
	//Invert this flips things so that small summed distances become bright colors
	rr = 1.0f - rr;
	gg = 1.0f - gg;
	bb = 1.0f - bb;
	
	//Clamp to [0,1] so its in a valid color range
	*r = fminf(fmaxf(rr, 0.0f), 1.0f);
	*g = fminf(fmaxf(gg, 0.0f), 1.0f);
	*b = fminf(fmaxf(bb, 0.0f), 1.0f);
}
__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, int width, int height, float time)//Add width, height, and time to function argument 
{
	float x,y;
	int id = threadIdx.x + blockDim.x*blockIdx.x; // global ID
	
	//Get total number of pixels
	int totalPixels = width*height;
	
	
	
	if(id < totalPixels)//If totalPixels doesn't divide nicely into blocks don't go poop in people's yards
	{
		int row = id/width; //Which row pixel is in
		int col = id%width; //Which col pixel is in i.e. the remainder after the above division
		
		//Asigning each thread its x and y value of its pixel.
		x = xMin + dx*col;
		y = yMin + dy*row;
		//Need 3 for r,g,b
		int k = 3*id;
		float s = 0.13f*sinf(time*0.3f); // Oscillates smoothly back and forth over time
		float cx = A + s*s; //wobble the real part of C based on time
		float cy = B;
		float r, g, b;
		juliaOrbitColor(x, y, cx, cy, &r, &g, &b);
		
		pixels[k] = r; //Setting the red
		pixels[k+1] = g; //Setting the green
		pixels[k+2] = b; //Setting the blue 
	}
}

void display(void) 
{ 
	dim3 blockSize, gridSize;
	float stepSizeX, stepSizeY;
	float totalPixels = WindowWidth*WindowHeight;//Get total number of pixels
	
	stepSizeX = (XMax - XMin)/((float)WindowWidth);
	stepSizeY = (YMax - YMin)/((float)WindowHeight);
	
	//Find the blockSize from GPU
	int maxThreadsPerBlock;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); 
	cudaErrorCheck(__FILE__, __LINE__);
	maxThreadsPerBlock = prop.maxThreadsPerBlock;
	
	blockSize.x = maxThreadsPerBlock;
	blockSize.y = 1;
	blockSize.z = 1;
	
	//Blocks in a grid - Enough blocks to cover every pixel round up
	gridSize.x = (totalPixels + blockSize.x - 1)/blockSize.x;
	gridSize.y = 1;
	gridSize.z = 1;
	
	colorPixels<<<gridSize, blockSize>>>(Pixels_GPU, XMin, YMin, stepSizeX, stepSizeY, WindowWidth, WindowHeight, Time);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Copying the pixels that we just colored back to the CPU.
	cudaMemcpyAsync(Pixels_CPU, Pixels_GPU, WindowWidth*WindowHeight*3*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	//NSYNC
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	//Putting pixels on the screen.
	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, Pixels_CPU); 
	glFlush(); 
	
}

//Update animation tick function
void update(int value)
{
	Time += 0.02f;
	glutPostRedisplay();
	glutTimerFunc(16, update, 0);
}

int main(int argc, char** argv)
{ 
	//Allocate Memory
	allocateMemory();
	
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutTimerFunc(16, update, 0); //kicks off animation loop
   	glutMainLoop();
}


