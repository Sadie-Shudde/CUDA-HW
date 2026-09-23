// Name: Sadie Shudde
// Robust Vector Dot product 
// nvcc J_GeneralDotProductWithAtomics_Sadie_HW10.cu -o temp
/*
 What to do:
 This code computes the dot product of vectors of any length using shared memory to
 reduce the number of global memory accesses. However, since blocks can’t synchronize
 with each other, the final reduction must be handled on the CPU.

 To simplify the GPU-side logic, we’ll add some “pregame” setup and use atomic adds.

 1. Make sure the number of threads per block is a power of 2. This avoids messy edge
    cases during the reduction step. If it’s not a power of 2, print an error message
    and exit. (Without this, you'd have to check if the reduction is even or not,
    add the last element to the first, adjust the loop, etc.)

 2. Calculate the correct number of blocks needed to process the entire vector.
    Then check device properties to ensure the grid and block sizes are within hardware limits.
    Just because it works on your fancy GPU doesn’t mean it will work on your client’s older one.
    If the block or grid size exceeds the device’s capabilities, report the issue and exit gracefully.

 3. It’s inefficient to check inside your kernel if a thread is working past the end of the vector
    on every iteration. Instead, figure out how many extra elements are needed to fill out the grid,
    and pad the vector with zeros. Zero-padding doesn’t affect the dot product (0 * anything = 0).
    Use `cudaMemset` to explicitly zero out your device memory — don’t rely on "getting lucky"
    like you might have in previous assignments.

 4. In previous assignments, we had to do the final reduction on the CPU because we couldn't sync blocks.
    Now, use **atomic adds** to sum partial results directly on the GPU and avoid CPU post-processing.
    Then, copy the final result back to the CPU using `cudaMemcpy`.

    Note: Atomic operations on floats are only supported on GPUs with compute capability 3.0 or higher.
    Use device properties to check this before running the kernel.
    While you’re at it, if multiple GPUs are available, select the best one based on compute capability.

 5. Add any additional bells and whistles to make your code more robust and user-proof.
    Think of edge cases or bad input your client might provide and handle it cleanly.
*/

/*
 Purpose:
 To learn how to use atomic adds to avoid jumping out of the kernel for block synchronization.
 This is also your opportunity to make the code "foolproof" — handling edge cases gracefully.

 At this point, you should understand all the CUDA basics.
 From now on, we’ll focus on refining that knowledge and adding advanced features.
*/
/* Explain what you did to fix the code:
1. Added a global variable for padding the vector length
2. Add an if statement and bitwise to check if block size if power of 2 kick them out if they don’t
3. Query the device. Find out how many NVIDIA GPU’s they have and then make sure they have one that works if not kick them out.
4. Now if they have multiple NVIDIA GPUs they must clearly be in love with Jensen so first contact Jensen’s wife, Lisa, and Dr. Wyatt cause someone is trying to steal their man. Then compare and find the best device and set it as the device. 
5. Now validate the grid/block size against the chosen device.
6. Make sure it can do atomic adds needs a compute capability of at least 3
7. Finally figure out the padded vector length to pad the last block with 0’s
8. In allocateMemory() for GPU memory multiply it by length of PaddedN
9. For the C_GPU it only needs to hold one float now for the atomic sum
10. Zero out the padded input arrays so the tail elements are exact
11. Now you can get rid of the even check in dot product and get rid of checking to see if its in your neighbors yard.
12. Add the atomicAdd() - adding atoms not splitting them. I was going to go on a rant about the atomic bomb dropping and make it a whole speech but I realize this will be on github and online so I will refrain… this time 
13. In main() reset the single result float to 0 before the atomic add
14. Copy back just the single final float dont need a perblock loop.
*/


// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 100000 // Length of the vector
#define BLOCK_SIZE 1024 // Threads in a block - Needs to be a power of 2 change - Works up to 1024

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;
int PaddedN; //Pad those N's like football players GO CHEIFS

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

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

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	//Make sure BlockSize.x is a power of 2
	//Its why most familys stop at 4 kids cause the 5th is just a waste of space or so I'm told
	if((BlockSize.x & (BlockSize.x-1)) != 0)
	{
		printf("\n\n Block Size must be a power of 2 idiot! You set it to %d.", BlockSize.x);
		printf("\n Go sail on the Titanic please!\n\n");
		exit(0);
	}
	
	GridSize.x = (N - 1)/BlockSize.x + 1; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
	
	//Check block and grid size against device limits
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cudaErrorCheck(__FILE__,__LINE__);
	
	//Make sure you have a NVIDIA GPU
	if(deviceCount == 0)
	{
		printf("\n\n No CUDA-capable device found. Jensen hater!");
		printf("\n Go back to the great depression!\n\n");
		exit(0);
	}
	
	//Get device properties
	//So you have a NVIDIA GPU Jensen would be proud now lets find the best one
	int bestDevice = 0;
	cudaDeviceProp bestProp;
	cudaGetDeviceProperties(&bestProp, 0);
	cudaErrorCheck(__FILE__,__LINE__);
	
	for(int i = 1; i< deviceCount; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__,__LINE__);
		
		if((prop.major > bestProp.major)|| (prop.major == bestProp.major && prop.minor > bestProp.minor))
		{
			bestDevice = i;
			bestProp = prop;
		}
	}
	
	cudaSetDevice(bestDevice);
	cudaErrorCheck(__FILE__,__LINE__);
	
	//Validate block/grid size against the chosen device's actual limit
	if(BlockSize.x > (unsigned int)bestProp.maxThreadsPerBlock)
	{
		printf("\n\n Block size %d exceeds this device's max threads per block (%d).", BlockSize.x, bestProp.maxThreadsPerBlock);
		printf("\n Bye Bye Bye.\n\n"); // NSYNC a decently good boyband not as good as One Direction
		exit(0);
	}
	
	if(GridSize.x > (unsigned int)bestProp.maxGridSize[0])
	{
		printf("\n\n Block size %d exceeds this device's max threads per block (%d).", GridSize.x, bestProp.maxGridSize[0]);
		printf("\n Bye Bye Bye.\n\n"); // NSYNC a decently good boyband not as good as One Direction
		exit(0);
	}
	//Check to make sure it can do atomic adds
	//You're the bomb.com don't look it up
	if(bestProp.major < 3)
	{
		printf("\n\n This device's compute capability (%d.%d) is to low for float attomic add.", bestProp.major, bestProp.minor);
		printf("\n Bye Bye Bye.\n\n"); // NSYNC a decently good boyband not as good as One Direction
		exit(0);
	}
	
	//Padded vector length
	PaddedN = GridSize.x * BlockSize.x;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU, sizeof(float)); //Only need one float for the answer
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Zero out the padded input so tail elements are 0
	cudaMemset(A_GPU, 0, PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(B_GPU, 0, PaddedN*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will doting.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	int threadIndex = threadIdx.x;
	int vectorIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ float c_sh[BLOCK_SIZE];
	
	c_sh[threadIndex] = (a[vectorIndex] * b[vectorIndex]);
	__syncthreads();
	
	int fold = blockDim.x;
	while(1 < fold)//You can get rid of all the n checks now cause of padding and power block of 2
	{
		/*if(fold%2 != 0) Power of 2 dont need this
		{
			if(threadIndex == 0)
			{
				c_sh[0] = c_sh[0] + c_sh[0 + fold - 1];
			}
			fold = fold - 1;
		}*/
		fold = fold/2;
		if(threadIndex < fold)
		{
			c_sh[threadIndex] = c_sh[threadIndex] + c_sh[threadIndex + fold];
			
		}
		__syncthreads();
	}
	
	//WATCH OUT AN ATOMIC ADD IS ABOUT TO DROP
	if(threadIndex == 0)
	{
		atomicAdd(c, c_sh[0]);
	}
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
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

// Cleaning up memory after we are finished.
void cleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Reset to single result float to 0 before the atmoic adds accumulate into it
	cudaMemset(C_GPU, 0, sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to DotGPU	
	cudaMemcpyAsync(&DotGPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	cleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}


