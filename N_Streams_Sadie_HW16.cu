// Name: Sadie Shudde
// Setting up a stream
// nvcc N_Streams_Sadie_HW16.cu -o temp

/*
 What to do:
 Read about CUDA streams. Look at all the ???s in the code and remove the ???s that need to be removed so the code will run.
*/

/*
 Purpose:
 To learn how to setup and work with CUDA streams.
*/

/*
 Explain what you did to fix the code:
 1. Added cudaMemcpyAsync in Stream0 to copy a chunk of A from the CPU to the GPU 
 2. Repeated for B
 3. Launched trigAdditionGPU in Stream0 for the current chunk the third parameter
    is 0 for shared memory and the fourth is the stream
 4. Added cudaMemcpyAsync in Stream0 to copy the chunk of C back from the GPU to the CPU
 5. Used A_CPU+i, B_CPU+i, C_CPU+i to point to the current chunk while the GPU buffers hold one chunk at a time
 6. Drama from Dancing with the Stars - You will be getting my weekly opinions from now 

*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define DATA_CHUNKS (1024*1024) 
#define ENTIRE_DATA_SET (20*DATA_CHUNKS)
#define MAX_RANDOM_NUMBER 1000
#define BLOCK_SIZE 256
#define PINK "\033[38;2;255;105;180m"
#define RESET "\033[0m"

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
cudaEvent_t StartEvent, StopEvent;
// ??? Notice that we have to define a stream
//Defying all odds is Ezra Frech. He is a paralympian with only 1 leg and one of his hands has only 2 fingers, but OH MY WORD he is my favorite star absolutely amazing!!!! Nothing bad to say love him his dances are amazing 
cudaStream_t Stream0;

//Function prototypes
void cudaErrorCheck(const char*, int);
void setUpCudaDevices();
void allocateMemory();
void loadData();
void cleanUp();
__global__ void trigAdditionGPU(float *, float *, float *, int );

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf(PINK "\n CUDA ERROR: message = %s, File = %s, Line = %d\n" RESET, cudaGetErrorString(error), file, line);
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
		printf( PINK "\n GPU will not handle overlaps so no speedup from streams");
		printf("\n Good bye." RESET);
		exit(0);
	}
	
	// ??? Notice that we have to create the stream
	//Content creator/comedian Connor Woods deserves more!!!! The judges are so rude to him for like no reason
	//I am not saying he is perfect but he is the only one they give harsh feedback to
	//He is just a little boy and I want to protect him. JUSTICE FOR CONNOR!
	cudaStreamCreate(&Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0)
	{
		printf(PINK "\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye." RESET);
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
	cudaMalloc(&A_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//??? Notice that we are using host page locked memory
	//Speaking of Hosts we should get rid of Julianne Houghs weird skybox interviews. I know she doesn't create the
	//questions but they are so awkward I would rather have the dancers do confessionals instead
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
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	// ??? Notice that we have to free this memory with cudaFreeHost
	//A lot controversy this year has come from Guillermo Rodriguez from Jimmy Kimmel Live.
	//His partner Whitney is pregnant and he is the stereotype older contestent who should leave week 1-2
	//However he made it to week 5 and Whitney is 5 months pregnant. He always had the lowest score but he is super 
	//entertaining and so stayed by fan votes each week. Most people were saying "Free Whitney" so she could stop
	//competeing but she was lowkey doing better 5 months pregnant than some of the other pros
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
	
	// ??? Notice that we have to kill the stream.
	// I do not want to kill him, but someone I do have bad things to say about is 
	//Jackson Olsen from the Savannah bananas. Vote him off the show he is boring,
	//dances aren't great, and he has zero and i mean ZERO chemistry with his pro partner. He also made so 
	//rude comments towards guillermo online and so therefore he sucks
	cudaStreamDestroy(Stream0);
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
	
	for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS)
	{
		cudaMemcpyAsync(A_GPU, A_CPU+i, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B_GPU, B_CPU+i, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
		
		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream0>>>(A_GPU, B_GPU, C_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);
		
		cudaMemcpyAsync(C_CPU+i, C_GPU, DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
	}
	
	// ??? Notice that we have to make the CPU wait until the GPU has finished stream0
	// People in sync would probably be Harry and Jenna or Amber and Pasha neither of them I like a whole lot
	//But they are pretty good. Although I am wondering if people will stop caring about Harry soon. 
	cudaStreamSynchronize(Stream0); 
	
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	// Make the CPU wait until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(PINK "\n Time on GPU = %3.1f milliseconds" RESET, timeEvent);
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
//Dancing with the Stars cast ranking as of Week 5 - not their dances just who I like the most
/*
1. Ezra Frech
2. Connor Wood
3. Guillermo Rodriquez - Eliminated last night don't talk to me about it
4. Tatyana Ali
5. Maura Higgins
6. Tyler Cameron
7. Amber GLenn
8. Jenna Dewan
9. Julia Stiles
10. Harry Shum Jr.
11. Jackson Olsen
*/
