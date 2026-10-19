// Name: Sadie Shudde
// Two streams overlapped smartly 
// nvcc O_SmartStreams_Sadie_HW17.cu -o temp
/*
 What to do:
 Read about CUDA stream cooperation.

This code provides most of the setup needed to create two CUDA streams. 
Complete the implementation by replacing all the ???s.

Once the two streams are working, overlap them in a smart way to improve performance.
*/

/*
 Purpose:
 To learn how to use CUDA streams intelligently.
*/

/*
 Explain what you did to fix the code:
 1. In the loop it can now handle 2 chunks per iteration one per stream
 2. Used A_CPU +i for stream0 and A_CPU+i+DATA_CHUNKS for stream1 so each stream works in its own chunk and repeated this for B and C
 3. Copied stream 0 and 1 up for A and then same for B
 4. Launched the kernal once per stream
 5. Copied C0 and C1 back to the CPU in their streams
 6. Overlapped the streams by issuing the work breadth first. A copies for both streams , then B copies, then both kernals, then both C copies. This keeps the copy engine
 and the kernals busy at the same time. 
 
*/
/* DRAMA on DANCING WITH THE STARS
Important people to know about
Judge - Carrie Ann Inaba - she has been a judge for all 35 seasons (And she is very honest, and a bit biased)
Pro dancer - Jenna Johnson - she has won the show twice and is currently partnered with a Grays Anatomy star named Harry
Pro Dancer - Val Chmerkovskiy - He has been a pro for a very long time is a staple on the show and a fan favortie. He is also married to Jenna Johnson. He is partnered with Actress Jenna Dewan this season.

Important things to know about the show
Several different types of dances are not allowed to have lifts. This is very important because Carrie Ann consistantly deducts points for lifts being in dances where they are not allowed, but because she is biased she sometimes doesn't.
Another things to know is that both Val and Jenna J have partners with past dance experience. Like both Harry and Jenna D were backup dancers for major pop stars. This angered a lot of fans cause it felt like Val and Jenna always get 
better partners.
Lift definition - "any movement where one dancer has both feet off the floor simultaneously with partner assistance."

Okay so we have some actual drama to catch up on this week! So on last weeks show Jenna J and Harry did a samba which cannot have lifts. However they had this move where backup dancers (not Jenna J) helped Harry do a flip.
Carrie Ann made some very blunt comments during judging saying Jenna J needed to "Add more content" and "make it a proper samba" Jenna J was clearly shocked at these lowkey rude comments but then right at the end of her talking time Carrie Ann
said "Oh and there was a lift" Jenna J is taken back and responds with "a lift?" Carrie Ann then reads the definition of a lift to her (Which is a bit belittling). But the formal definition says partner lifts the person and Jenna J did
not lift Harry backup dancers did.

Anyways the next day on instagram Val, who is Jenna J's husband, posts on his story "You almost saw Fresh Prince enter the ballroom last night" this was a reference to when Will Smith slapped Chris Rock at the Oscars for making a comment
about his wife. Val meant this as a reference to Will saying "Keep my wife's name out your f***ing mouth!" not the slapping part. However, fans freaked out and Carrie Ann, who most fans hate, used this to her advantage to garner sympathy and 
basically everyone freaked out thinking Val was threating to slap Carrie Ann. Val has now been suspended for 1 week of the show. This next week is dedication night which is where the stars dedicate their dance to someone in their life. It is a 
pretty big night to miss. I do not think Val's partner Jenna D will be voted off cuase a lot of fans are mad about the suspention, but we will see. You will find out with Thursdays Homework(Now that is actually a threat).

*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define DATA_CHUNKS (1024*1024) 
#define ENTIRE_DATA_SET (20*DATA_CHUNKS)
#define MAX_RANDOM_NUMBER 1000
#define BLOCK_SIZE 256

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A0_GPU, *B0_GPU, *C0_GPU, *A1_GPU, *B1_GPU, *C1_GPU; //GPU pointers
cudaEvent_t StartEvent, StopEvent;
cudaStream_t Stream0, Stream1;

//Function prototypes
void cudaErrorCheck(const char *, int);
void setUpCudaDevices();
void allocateMemory();
void loadData();
void cleanUp();
__global__ void trigAdditionGPU(float *, float *, float *, int );

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

//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaDeviceProp prop;
	int whichDevice;
	
	cudaGetDevice(&whichDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaGetDeviceProperties(&prop, whichDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(prop.deviceOverlap != 1)
	{
		printf("\n GPU will not handle overlaps so no speedup from streams");
		printf("\n Good bye.");
		exit(0);
	}
	
	cudaStreamCreate(&Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamCreate(&Stream1);
	cudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0)
	{
		printf("\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye.");
		exit(0);
	}
	GridSize.x = DATA_CHUNKS/BLOCK_SIZE;
	GridSize.y = 1;
	GridSize.z = 1;	
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A0_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B0_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C0_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&A1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C1_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Allocate page locked Host (CPU) Memory
	cudaHostAlloc(&A_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&B_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&C_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
}

void loadData()
{
	time_t t;
	srand((unsigned) time(&t));
	
	for(int i = 0; i < ENTIRE_DATA_SET; i++)
	{		
		A_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;
		B_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(A0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C0_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(A1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C1_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaFreeHost(A_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(B_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(C_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaStreamDestroy(Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaStreamDestroy(Stream1);
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void trigAdditionGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n)
	{
		c[id] = sin(a[id]) + cos(b[id]);
	}
}

int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	loadData();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS*2)
	{	
		//Copy A up for both streams first then B then run both kernals
		cudaMemcpyAsync(A0_GPU, A_CPU+i, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(A1_GPU, A_CPU+i+ DATA_CHUNKS, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);
		
		cudaMemcpyAsync(B0_GPU, B_CPU+i, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B1_GPU, B_CPU+i+DATA_CHUNKS, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);
		
		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream0>>>(A0_GPU, B0_GPU, C0_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);
		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream1>>>(A1_GPU, B1_GPU, C1_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);
		
		//copy C back for both
		cudaMemcpyAsync(C_CPU+i, C0_GPU, DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(C_CPU+i+DATA_CHUNKS, C1_GPU, DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream1);
		cudaErrorCheck(__FILE__, __LINE__);
	}
	
	// Make the CPU wait until the Streams have finishd before it continues.
	cudaStreamSynchronize(Stream0);
	cudaStreamSynchronize(Stream1);
	
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	// Make the CPU wiat until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU = %3.1f milliseconds", timeEvent);
	
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
