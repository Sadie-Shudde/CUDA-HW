// Name: Sadie Shudde
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc S_NBodyCPUToGPU1Block_Sadie_HW21.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the CPU. Rewrite it, keeping the same general format, 
 but offload the compute-intensive parts of the code to the GPU for acceleration.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate, (We will keep the number of bodies under 1024 for this HW so it can be run on one block.)
 2. Whether to draw sub-arrangements of the bodies during the simulation (1), or only the first and last arrangements (0).
*/

/*
 Purpose:
 To learn how to move an Nbody CPU simulation to an Nbody GPU simulation.
*/

/*
 Explain what you did to fix the code:
 1. Added GPU pointers and the dimension stuff for the block
 2. Set up one block with N threads (1 thread per body) and allocated the memory in setUp()
 3. Added the kernal so each thread loops over every other body and adds the total force on its own body
 4. synced the threads cause equality after the force calculations to make sure no one moves until every thread has the old positions
 5. Each thread then updates its own velocity and position
 6. Synced threads again cause equality or whatever so the bodies finish moving before the next time step starts
 7. Replaced the nBody() function so it loops with a host loop that copies the data to the GPU, launches the kernal for DRAW_RATE steps at a time, and copies position back
 8. Copied final positions and velocities back 
 9. Added a check that N is between 1 and 1024 so its in a single block
 10. Added things to make it zoom and rotate 
 
 
 I have no DWTS to talk about today so instead here is a list of the craziest plot lines from each season of the TV show Riverdale.
 Also for referance this is based on the Archie comics
 
 Season 1 - Cheryl Blossom kept her dead twins corpse in the basement chapel
 Season 2 - There is a serial killer named the Black Hood that is a main characters father 
 Season 3 - The town gets consumed by a live action role playing game and there is now a cult called "The Farm" brainwashes half the town and attempts a mass ascension via a homemade rocket ship before being exposed as a front for organ harvesting. Also Archie escapes from prison and gets mauled by a grizzly bear
 Season 4 - Jughead fakes his own death, hides out in a subterranean bunker, and forces his grieving friends to host a fake wake for him
 Season 5 - After a time jump ages the cast into young adults, Jughead hits rock bottom and in a rabies-induced, alcohol-fueled bender, he hallucinates falling into the New York City sewers and entertaining a literal Rat King by reading him his novel, only to wake up swarmed by real rats.
 Season 6 - A bomb blast under Archie’s bed catapults the gang into Rivervale, a parallel supernatural dimension where Archie is sacrificed Midsommar-style and Veronica outsmarts the Devil. When they return to regular reality, the explosion gives the teens actual superpowers to fight an immortal villain and destroy a comet.
 Season 7 - Destroying the comet sends the entire town hurtling back in time to 1955, where the characters are trapped as high schoolers with wiped memories. To cap off the surreal ride, the series finale reveals that instead of choosing between Archie, Betty, Veronica, or Jughead, the teens magically remember all timelines and decide to date all four of them in a 1950s polyamorous group arrangement.

 
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). (p < q) p has to be less than q.
// In this code we will keep it a p = 2 and q = 4 problem. The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float *M; 
float GlobeRadius, Diameter, Radius;
float Damp;
float3 *P_GPU, *V_GPU; //Body positions and velocities on the GPU
float *M_GPU; //Body mass on GPU
dim3 BlockSize, GridSize;
//You spin me round right baby right round - Flo Rida
float CameraZ;
float RotateY = 0.0f;

// Function prototypes
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody();
int main(int, char**);
void cudaErrorCheck(const char*, int);
__global__ void nBodyGPU(float3 *, float3 *, float *, float, float, int, int, int);

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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

//One thread per body all in one block so don't have to worry about sharing memory
__global__ void nBodyGPU(float3 *p, float3 *v, float *m, float damp, float dt, int n, int steps, int first)
{
	int i = threadIdx.x; //ID
	float3 f; //This will be the total force
	float dx, dy, dz, d, d2, forceMag; 
	
	//Each pass of this loop is one time step
	for(int s = 0; s < steps; s++)
	{
		//Each thread adds up the total force on its own body from every other body so start at 0
		f.x = 0.0f;
		f.y = 0.0f;
		f.z = 0.0f;
		
		if(i<n)//Only threads that have a body do work
		{
			for(int j = 0; j < n; j++)//Add up the forces from every other body
			{
				if(j != i)//if its not equal to the thread id cause a body cant pull on itself
				{	//Distance
					dx = p[j].x - p[i].x;
					dy = p[j].y - p[i].y;
					dz = p[j].z - p[i].z;
					d2 = dx*dx + dy*dy + dz*dz;
					d = sqrt(d2);
					
					forceMag = (G*m[i]*m[j])/((float)d2) - (H*m[i]*m[j])/((float)d2*d2);//Good old John Lennard Jones small bottoms make big tops
					f.x += forceMag*dx/d;
					f.y += forceMag*dy/d;
					f.z += forceMag*dz/d;
					
				}
			}
		}
		//EQUALITY!!!!!! We wait for those who can't finish fast enough
		__syncthreads();
		
		if(i<n)
		{
			if(first == 1 && s == 0)//Very first step of the run uses a half step and no damping leap frog!!!!
			{
				v[i].x += (f.x/m[i])*0.5f*dt;
				v[i].y += (f.y/m[i])*0.5f*dt;
				v[i].z += (f.z/m[i])*0.5f*dt;
			}
			else
			{
				v[i].x += ((f.x-damp*v[i].x)/m[i])*dt;
				v[i].y += ((f.y-damp*v[i].y)/m[i])*dt;
				v[i].z += ((f.z-damp*v[i].z)/m[i])*dt;
			}
			//New position
			p[i].x += v[i].x*dt;
			p[i].y += v[i].y*dt;
			p[i].z += v[i].z*dt;
		}
		//EQUALITY!!!!!! We wait for those who can't finish fast enough even through they totally should have finished already
		__syncthreads();
		
	}
}

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		timer();
	}
	if(key == '+' || key == '=') CameraZ -= 0.1*GlobeRadius;
	if(key == '-' || key == '_') CameraZ += 0.1*GlobeRadius;
	if(key =='a') RotateY -= 10.0f;//Left Arrow
	if(key =='d') RotateY += 10.0f;//Right Arrow
	
	if(CameraZ < 0.5*GlobeRadius) CameraZ = 0.5*GlobeRadius; //Woah give the balls some space
	if(CameraZ > 4.0*GlobeRadius) CameraZ = 4.0*GlobeRadius; //Unlike Matt Damen in The Martian don't get lost in space
	if(key == 'q')
	{
		cudaFree(P_GPU); 
		cudaErrorCheck(__FILE__, __LINE__);
		cudaFree(V_GPU); 
		cudaErrorCheck(__FILE__, __LINE__);
		cudaFree(M_GPU);
		cudaErrorCheck(__FILE__, __LINE__);
		exit(0);
	}
	
	glutPostRedisplay(); //Redraw
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glLoadIdentity(); //Rests the Camera then place it using the current zoom and spin
	gluLookAt(0.0, 0.0, CameraZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glRotatef(RotateY, 0.0f, 1.0f, 0.0f);
	
	for(i=0; i<N; i++)
	{
		double shade = (double)i/N;
		glColor3d(1.0, 0.2+0.6*shade, 0.5+0.4*shade);
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    		nBody();
    	gettimeofday(&end, NULL);
    	drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
    	float randomAngle1, randomAngle2, randomRadius;
    	float d, dx, dy, dz;
    	int test;
    	
    	Damp = 0.5;
    	
    	M = (float*)malloc(N*sizeof(float));
    	P = (float3*)malloc(N*sizeof(float3));
    	V = (float3*)malloc(N*sizeof(float3));
    	F = (float3*)malloc(N*sizeof(float3));
    	
	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the initial velosity, initial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;
	}
	BlockSize.x = N; //This will not work for the N = 1500 run because a Blocks max size is 1024 or 2^10
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;
	GridSize.y = 1;
	GridSize.z = 1;
	
	// Device "GPU" Memory
	cudaMalloc(&P_GPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&V_GPU,N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&M_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	printf("\n To start timing type s.\n");
}

void nBody()
{
	int totalSteps = (int)(RUN_TIME/DT + 0.5);
	int chunks = totalSteps/DRAW_RATE; //GPU runs RAW_RATE many steps we might come back to draw
	
	//Copy current memory state to the GPU
	cudaMemcpyAsync(P_GPU, P, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(V_GPU, V, N*sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(M_GPU, M, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	for(int c = 0; c < chunks; c++)
	{	//Run DRAW_RATE time steps on the GPU
		nBodyGPU<<<GridSize, BlockSize>>>(P_GPU, V_GPU, M_GPU, Damp, DT, N, DRAW_RATE, (c==0));
		cudaErrorCheck(__FILE__, __LINE__);
		
		if(DrawFlag)
		{	//Only copy positions back and draw if asked to see the animation
			cudaMemcpy(P, P_GPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
			cudaErrorCheck(__FILE__, __LINE__);
			drawPicture();
		}
	}
	
	//Bring the final positions and velocites back to the CPU
	cudaMemcpy(P, P_GPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(V, V_GPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move (1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}
	if(N > 1024 || N < 1)
	{
		printf("\n The number of bodies must be between 1 and 1024 so its in 1 block  \n");
		exit(0);
	}
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	CameraZ = 2.0f*GlobeRadius;
	float3 eye = {0.0f, 0.0f, CameraZ};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
}





