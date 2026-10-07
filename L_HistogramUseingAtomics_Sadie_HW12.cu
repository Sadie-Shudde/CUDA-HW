// Name: Sadie Shudde
// Histogram useing atomics in global memory and shared memory.
// nvcc L_HistogramUseingAtomics_Sadie_HW12.cu -o temp

/*
 What to do:
 This code generates a series of random numbers and places them into bins based on size ranges using the CPU.

 Your task:
 - Create a binning scheme that utilizes the GPU.
 - Take advantage of both global and shared memory atomic operations.
 - The function call has already been provided.
 - Set the block size to **twice** the number of multiprocessors on the GPU.
*/

/*
 Purpose:
 To learn how to use atomic operations at both the shared and global memory levels.
 Along the way, you'll also learn a bit about generating random numbers using `srand`,
 which will come in handy when we use `curand` in a later assignment.
*/

/*
 Explain what you did to fix the code:
 1. Changed block size to be double the number of multiprocessors
 2. Added shared memory and zeroed out the blocks
 3. Researched conspiracy theories
 4. Used atomic adds for each thread to go in the right bin
 5. used atomic adds for the blocks
 
 What's the worst thing about getting hit by a truck? The paperwork
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

/*
 Note: The Max int value is 2,147,483,647
 The length of the sequence of random number that srand generates is 2^32,
 that is 4,294,967,296 which is bigger than the largest int but the max for an unsigned int.
*/

// Defines
#define NUMBER_OF_RANDOM_NUMBERS 2147483
#define NUMBER_OF_BINS 10
#define MAX_RANDOM_NUMBER 100.0f
#define PINK "\033[38;2;255;105;180m"
#define RESET "\033[0m"

// Global variables
float *RandomNumbersGPU;
int *HistogramGPU;
float *RandomNumbersCPU;
int *HistogramCPU;
int *HistogramCPUTemp; // Use it to hod the GPU histogram past back so we can compair to CPU histogram.
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//Function prototypes
void cudaErrorCheck(const char *, int);
void SetUpCudaDevices();
void AllocateMemory();
void Innitialize();
void CleanUp();
void fillHistogramCPU();
__global__ void fillHistogramGPU(float *, int *);
int main();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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
void SetUpCudaDevices()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = 2*prop.multiProcessorCount; //Twice the number of multiprocessors on the GPU
	if(prop.maxThreadsDim[0] < BlockSize.x)
	{
		printf(PINK "\n You are trying to create more threads (%d) than your GPU can support on a block (%d).\n Good Bye\n" RESET, BlockSize.x, prop.maxThreadsDim[0]);
		exit(0);
	}
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (NUMBER_OF_RANDOM_NUMBERS - 1)/BlockSize.x + 1; //Makes enough blocks to deal with the whole vector.
	if(prop.maxGridSize[0] < GridSize.x)
	{
		printf(PINK "\n You are trying to create more blocks (%d) than your GPU can suppport (%d).\n Good Bye\n" RESET, GridSize.x, prop.maxGridSize[0]);
		exit(0);
	}
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&RandomNumbersGPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&HistogramGPU, NUMBER_OF_BINS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate Host (CPU) Memory
	RandomNumbersCPU = (float*)malloc(NUMBER_OF_RANDOM_NUMBERS*sizeof(float));
	HistogramCPU = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	HistogramCPUTemp = (int*)malloc(NUMBER_OF_BINS*sizeof(int));
	
	//Setting the the histograms to zero.
	cudaMemset(HistogramGPU, 0, NUMBER_OF_BINS*sizeof(int));
	cudaErrorCheck(__FILE__, __LINE__);
	memset(HistogramCPU, 0, NUMBER_OF_BINS*sizeof(int));
}

//Loading random numbers.
void Innitialize()
{
	time_t t;
	srand((unsigned) time(&t));
	
	// rand() returns an int in [0, RAND_MAX] "end points included".
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{		
		RandomNumbersCPU[i] = MAX_RANDOM_NUMBER*(float)rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void CleanUp()
{
	cudaFree(RandomNumbersGPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(HistogramGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	free(RandomNumbersCPU); 
	free(HistogramCPU);
	free(HistogramCPUTemp);
	//printf("\n Cleanup Done.");
}

void fillHistogramCPU()
{
	float breakPoint;//We're breaking free! Soaring Flying Theres not a star in heaven that we can't reach!!!!
	//That was just for me to have my high school musical moment that was not for you you can move on now
	int k, done;
	float stepSize = MAX_RANDOM_NUMBER/(float)NUMBER_OF_BINS;
	
	for(int i = 0; i < NUMBER_OF_RANDOM_NUMBERS; i++)
	{
		breakPoint = stepSize;
		k = 0;
		done = 0;
		while(done == 0)
		{
			if(RandomNumbersCPU[i] < breakPoint)
			{
				HistogramCPU[k]++; 
				done = 1;
			}
			
			if(NUMBER_OF_BINS < k)
			{
				printf(PINK "\n k is too big\n" RESET);
				exit(0);
			}
			k++;
			breakPoint += stepSize;
		}
	}
}

//This is the kernel. It is the function that will run on the GPU.
__global__ void fillHistogramGPU(float *randomNumbers, int *hist)
{
	//Shared memory histogram one copy per block
	__shared__ int sharedHist[NUMBER_OF_BINS];
	
	int id = threadIdx.x +blockDim.x*blockIdx.x; //global thread id been here done that
	//Zero the shared histogram cause i dont wanna do cudaMemSet
	for(int i = threadIdx.x; i<NUMBER_OF_BINS; i += blockDim.x)
	{
		sharedHist[i] = 0;
	}
	__syncthreads(); //We both know my feelings about the titanic at this point but lets talk about some crazy conspiracy theories
	//It was actually the Olympic sister ship that sunk not the titanic! 6/10
	//J.P Morgan sunk it to kill several millionaries he didn't like 9//10 probably true
	//A mummy cursed the ship 3/10 really really funny but no
	
	//Each thread bins its own number using an atomic add in shared memory
	if(id < NUMBER_OF_RANDOM_NUMBERS)
	{
		float stepSize = MAX_RANDOM_NUMBER/(float)NUMBER_OF_BINS;
		int bin = (int)(randomNumbers[id]/stepSize);
		
		//A value of exactly the max would land one past the last bin so clamp it
		if(bin>= NUMBER_OF_BINS) bin = NUMBER_OF_BINS -1;
		
		atomicAdd(&sharedHist[bin], 1);
	}
	__syncthreads();//While looking up conspiracy theories I found this quote and its my new favorite thing
	//"so painfully ridiculous that one can only lament the thousands of trees which lost their lives to provide the paper on which it has been articulated."
	//Just thought you would enjoy a new insult
	
	//Merge this block's shared histogram into the global histogram
	for(int i = threadIdx.x; i < NUMBER_OF_BINS; i+= blockDim.x)
	{
		atomicAdd(&hist[i], sharedHist[i]);
	}
}

int main()
{
	float time;
	timeval start, end;
	
	long int test = NUMBER_OF_RANDOM_NUMBERS;
	if(2147483647 < test)
	{
		printf(PINK "\nThe length of your vector is longer than the largest integer value allowed of 2,147,483,647.\n" RESET);
		printf(PINK "You should check your code.\n Go to antartica and stay there until you fix your vector\n" RESET);
		exit(0);
	}
	
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using and padding with zero vector will be a factor of block size.
	AllocateMemory();

	//Loading up values to be added.
	Innitialize();
	
	gettimeofday(&start, NULL);
	fillHistogramCPU();
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf(PINK "\nTime on CPU = %.15f milliseconds\n" RESET, (time/1000.0));
	
	gettimeofday(&start, NULL);
	//Copy Memory from CPU to GPU		
	cudaMemcpyAsync(RandomNumbersGPU, RandomNumbersCPU, NUMBER_OF_RANDOM_NUMBERS*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	fillHistogramGPU<<<GridSize,BlockSize>>>(RandomNumbersGPU, HistogramGPU);
	cudaErrorCheck(__FILE__, __LINE__);
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(HistogramCPUTemp, HistogramGPU, NUMBER_OF_BINS*sizeof(int), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	gettimeofday(&end, NULL);
	time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf(PINK "\nTime on GPU = %.15f milliseconds\n" RESET, (time/1000.0));
	
	//Check
	for(int i = 0; i < NUMBER_OF_BINS; i++)
	{
		printf(PINK"\n Deference in histogram bins %d is %d." RESET, i, abs(HistogramCPUTemp[i] - HistogramCPU[i]));
	}
	
	//You're done so cleanup your mess.
	CleanUp();	
	
	printf("\n\n");
	return(0);
}
