// Name: Sadie Shudde
// Page-locked memory test
// nvcc M_PageLockedMemory_Sadie_HW15.cu -o temp

/*
 What to do:
 Read about **page-locked (pinned) memory**. Fill in the ???s in this code to understand how to
 set up and test page-locked memory on the host.
*/

/*
 Purpose:
 To learn how page-locked (pinned) memory works and how to use it effectively.
*/

/*
 Explain what you did to fix the code:
 1. Used cudaHostAlloc in allocateMemory() to allocate the page locked host memory
 2. Used cudaFreeHost in cleanUp() to free pinned memory
 3. Used cudaMemcpyAsync in copyPageLockedMemoryUp() to copy pinned memory from CPU to GPU
 4. Used cudaMemcpyAsync in copyPageLockedMemoryDown() to copy from GPU back to pinned memory in CPU
 5. Fixed typo in main 
 6. Face Copyright allogations against Grays Anatomy
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define SIZE 2000000 
#define NUMBER_OF_COPIES 1000
#define PINK "\033[38;2;255;105;180m"
#define RESET "\033[0m"

//Globals
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
cudaEvent_t StartEvent, StopEvent;

//Function prototypes
void cudaErrorCheck(const char *, int);
void setUpCudaDevices();
void allocateMemory();
void cleanUp();
void copyPageableMemoryUp();
void copyPageLockedMemoryUp();
void copyPageableMemoryDown();
void copyPageLockedMemoryDown();

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
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&NumbersOnGPU, SIZE*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	//Allocate pageable Host (CPU) Memory
	PageableNumbersOnCPU = (float*)malloc(SIZE*sizeof(float));
	
	//Allocate page locked Host (CPU) Memory
	cudaHostAlloc(&PageLockedNumbersOnCPU, SIZE*sizeof(float), cudaHostAllocDefault); //Allocation 
	cudaErrorCheck(__FILE__, __LINE__);
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(NumbersOnGPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaFreeHost(PageLockedNumbersOnCPU);//Free memory more like Free Bird not imagine an epic air guitar solo right here
	cudaErrorCheck(__FILE__, __LINE__);
	
	free(PageableNumbersOnCPU); 
	
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
}

void copyPageableMemoryUp() //Paging Dr. Shepard
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(NumbersOnGPU, PageableNumbersOnCPU, SIZE*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageableMemoryDown()//Where is Dr. Shepard?
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpy(PageableNumbersOnCPU, NumbersOnGPU, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageLockedMemoryUp()//Oh thats right he is dead...Sorry Merdith
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpyAsync(NumbersOnGPU, PageLockedNumbersOnCPU, SIZE*sizeof(float), cudaMemcpyHostToDevice);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}

void copyPageLockedMemoryDown()
{
	for(int i = 0; i < NUMBER_OF_COPIES; i++)
	{
		cudaMemcpyAsync(PageLockedNumbersOnCPU, NumbersOnGPU, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
		cudaErrorCheck(__FILE__, __LINE__);
	}
}
/*
You don't have the right to question me. You don't have the right to mention my daughter's name. You stood over my dying husband and you didn't act fast enough. You failed to order a head CT. You let him die because you were a coward. - Meredith Gray
*/

int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(PINK "\n Time on GPU using pageable memory up = %3.1f milliseconds" RESET, timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryUp();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(PINK "\n Time on GPU using page locked memory up = %3.1f milliseconds" RESET, timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageableMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(PINK "\n Time on GPU using pageable memory down = %3.1f milliseconds" RESET, timeEvent);
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	copyPageLockedMemoryDown();
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(PINK "\n Time on GPU using page locked memory down = %3.1f milliseconds" RESET, timeEvent);
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	//Hopefully you have seen Grays Anatomy before or else none of my jokes will make sense sorryyyyyyyyyyy
	
	return(0);
}
