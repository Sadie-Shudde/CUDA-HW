// Name: Sadie Shudde
// Device query
// nvcc E_DeviceQuery_Sadie_HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

/*
 Purpose:
 To learn how to find out what is on the GPU(s) in your machine and if you even have a GPU.
*/

/*
 Explain what you did to fix the code:
 1. Lines 71-84 include the General Information added about the GPU. Each line has a comment to explain what was added
 2. Lines 93-111 include the Memory Information added about the GPU. Each line has a comment to explain what was added
 3. Lines 114-117 include the Multiprocessor Information added about the GPU. Each line has a comment to explain what was added
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		//Added Information
		printf("Integrated GPU (shares memory with CPU): %s\n", prop.integrated ? "Yes" : "No"); //integrated vs discrete chip
		printf("Can map host memory (zero-copy): %s\n", prop.canMapHostMemory ? "Yes" : "No"); //Can GPU directly access pinned CPU memory		
		printf("Compute Mode: %d\n", prop.computeMode);  // 0=Default,1=Exclusive,2=Prohibited,3=ExclusiveProcess - controls how many processes/threads can use the device at once
		printf("Concurrent kernels supported: %s\n", prop.concurrentKernels ? "Yes" : "No"); // can the GPU run more than one kernel at the same time
		printf("ECC enabled: %s\n", prop.ECCEnabled ? "Yes" : "No"); // error correcting memory
		printf("PCI Bus ID: %d, Device ID: %d, Domain ID: %d\n", prop.pciBusID, prop.pciDeviceID, prop.pciDomainID); // physical location of the card on the PCI bus, useful for multi-GPU systems
		printf("Unified addressing (CPU/GPU share address space): %s\n", prop.unifiedAddressing ? "Yes" : "No");
		printf("Managed memory (cudaMallocManaged) supported: %s\n", prop.managedMemory ? "Yes" : "No"); // supports Unified Memory
		printf("Async engine count (copy engines): %d\n", prop.asyncEngineCount); // number of engines that can do memory copies concurrently with kernel execution
		printf("Stream priorities supported: %s\n", prop.streamPrioritiesSupported ? "Yes" : "No");
		printf("Cooperative kernel launch supported: %s\n", prop.cooperativeLaunch ? "Yes" : "No"); // supports grid-wide synchronization (cooperative groups)
		printf("TCC driver: %s\n", prop.tccDriver ? "Yes" : "No"); // Windows-specific: Tesla Compute Cluster driver mode vs normal display driver
		printf("Is multi-GPU board: %s\n", prop.isMultiGpuBoard ? "Yes" : "No"); // true if this is one GPU of a physical dual-GPU board
		printf("\n");
		
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		
		//Added Information
		printf("Surface Alignment: %ld\n", prop.surfaceAlignment); // required byte alignment for surface memory base addresses
		printf("L2 cache size: %d bytes\n", prop.l2CacheSize); // size of the GPU's L2 cache, shared by all SMs
		printf("Memory clock rate: %d kHz\n", prop.memoryClockRate); // speed of the GDDR/HBM memory itself separate from core clock
		printf("Memory bus width: %d bits\n", prop.memoryBusWidth); // width of the memory interface, used with clock rate to get bandwidth
		printf("Peak theoretical memory bandwidth: %.2f GB/s\n",
			2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6); // x2 because most GPU memory is double data rate (DDR)
		printf("Max Texture 1D size: %d\n", prop.maxTexture1D); // largest allowed 1D texture
		printf("Max Texture 2D size: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]); // largest allowed 2D texture dimensions
		printf("Max Texture 3D size: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]); // largest allowed 3D texture dimensions
		printf("\n");
		
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		
		//Added Information
		printf("Shared mem per multiprocessor total, not per block: %ld\n", prop.sharedMemPerMultiprocessor); // total shared mem available on one SM, can be split among resident blocks
		printf("Registers per multiprocessor total, not per block: %d\n", prop.regsPerMultiprocessor); // total register file size per SM, split among resident threads
		printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor); // max resident threads (across all blocks) that can live on one SM at once - used for occupancy calculations
		printf("\n");
	}	
	return(0);
}

