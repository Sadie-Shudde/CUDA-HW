// Name: Sadie Shudde
// GPU random walk. 
// nvcc P_GPURandomWalk_Sadie_HW18.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Explain what you did to fix the code:
 1. Included the library needed for cuRAND in kernal
 2. Defined NUMBER_OF_WALKS as 20, one walk per GPU thread.
 3. Wrote the kernal where each thread runs its own 10,00 step random walk
 4. Used curand_init with seed+thread ID so each walk gets a different seed
 5. Used curand_uniform to pick each step: below 0.5 is -1 and otherwise +1, for both x and y
 6. Used cudaMallocManaged to allocate the position arrays in unified memory so no CudaMemcpy is needed
 7. Launced 1 block of 20 threads and called the synchronize so the CPU waits before reading the result
 8. Printed all 20 final positions and freed memory with cudaFree

*/

/*
 Note:
Okay dedication night of dancing wiht the stars!!! What an emotional night! Basically everyone picks one person to dedicate their dance to and that person comes and dances with them. Or if that person is no longer with us someone else with dance with them. Okay big moments
1. Jenna danced with Alan instead of Val after liftgate they did really well some fans want her switched to only dance with Alan I personally don't care
2. Connor who is like my favorite besides Ezra(one legged guy) finally go the respect he deserves and got all 8's!
3. Erza's dance was amazing as always no surprise there highly recommend watching
4. Tyler Cameron dedicated his dance to moms everywhere after his mom passed suddenly passed in 2020. He danced with his fiance and they announced they are pregnant! They did a gender reveal after the dance and are having a baby girl.
5. Right after Tylers dance they had a promotional dance for the new Cat in the Hat movie and it was so weird

Quote of the week
"I saw you drinking a smoothie of 8th avenue and you paid me no mind!"
Basicaly Connor did an interview where he said Jennifer Lawerence was his celebrity crush and he just wanted her to say he was funny. She then commented the quote above on his instgram and its now become a huge thing.
Now we need to know if they will actually get to talk will they have a smoothie together!

Everyone needs to text "CONNOR" 10 times to 21523. On Tuesday at 7pm. No one was eliminatedon dedication night and there will now be a double elimination on Horror movie night
So everyone text CONNOR to 21523 10 times!
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h> //Include this for the cuRAND

// Defines
#define NUMBER_OF_WALKS 20 //20 walks per questions

// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
int main(int, char**);
__global__ void randomWalkGPU(int *, int *, int, unsigned long long);

int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}
//KERNAL FUNCTION
__global__ void randomWalkGPU(int *positionX, int *positionY, int steps, unsigned long long seed)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x; //One thread per walk
	
	if(id < NUMBER_OF_WALKS)
	{
		curandState state;
		curand_init(seed+id, 0, 0, &state); //Different seed for each walk
		
		int x = 0;
		int y = 0;
		for(int i = 0; i<steps; i++)
		{
			//curand_uniform gives a float in (0,1]. below 0.5 goes to -1 otherwise +1
			if(curand_uniform(&state) < 0.5f) x -= 1;
			else x += 1;
			
			if(curand_uniform(&state) < 0.5f) y -= 1;
			else y += 1;
		}
		
		positionX[id] = x;
		positionY[id] = y;
	}
}

int main(int argc, char** argv)
{
	srand(time(NULL));
	
	printf(" RAND_MAX for this implementation is = %d \n", RAND_MAX);
	
	int *positionX, *positionY;
	//Unified Memory accessible from both CPU and the GPU
	cudaMallocManaged(&positionX, NUMBER_OF_WALKS*sizeof(int));
	cudaMallocManaged(&positionY, NUMBER_OF_WALKS*sizeof(int));
	
	//One block of 20 threads, one thread per random walk
	randomWalkGPU<<<1, NUMBER_OF_WALKS>>>(positionX, positionY, NumberOfRandomSteps, (unsigned long long)time(NULL));
	cudaDeviceSynchronize(); //Wait for the GPU to finish before the CPU reads the unified memory
	
	for(int i = 0; i < NUMBER_OF_WALKS; i++)
	{
		printf("\n Walk %2d final position = (%d,%d)\n", i+1, positionX[i], positionY[i]);
	}
	
	cudaFree(positionX);
	cudaFree(positionY);
	
	printf("\n");
	return 0;
}

