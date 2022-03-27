//Adam Bayley 20176309 19ahb

//Adam Bayley 20176309 19ahb Machine Problem 3

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <ctime>

#define S 16 //16x16, 256x256,... etc

int flag = 0; //flag for checking if matrices are =

//device matrix mult. calculates row and col of the grid / block and then 
//flattens matrix before inserting values
__global__ void DeviceMatrixMultiplication(int *A, int *B, int *O, int size) {
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < size && col < size) {
		float temp = 0;
		for (int i = 0; i < size; i++) {
			temp = temp + A[row*size + i] * B[i*size + col];
			O[row*size + col] = temp;
		}//close for i
	}//close if
}//close void devicematrix

void HostMatrixMultiplication(int *A, int *B, int *C, int size) { //maybe needs to be a __host__ ??
	int offset1, offset2;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			float temp = 0;
			for (int k = 0; k < size; k++) {
				offset1 = i*size + k;
				offset2 = k*size + j;

				temp = temp + A[offset1] * B[offset2];
			}//close for
			C[i*size + j] = temp;
		}//close for
	}//close for
}//close hostmatrix

int main() {

	//setup timer events
	cudaEvent_t start1, stop1, start2, stop2;
	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);

	//variable to keep track of time
	time_t t;

	float gpu_time1 = 0.0f;
	float gpu_time2 = 0.0f;

	//synchronize
	cudaDeviceSynchronize();



	//seed the random values
	srand((unsigned)time(&t));

	//get the size of the matrix
	size_t hostSize = S*S*sizeof(int);

	//allocate host memory
	int* h_A = (int*)malloc(hostSize);
	int* h_B = (int*)malloc(hostSize);
	int* h_C = (int*)malloc(hostSize);
	int* h_P = (int*)malloc(hostSize);

	//initialize host matrix
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < S; j++) {
			//get the 2 random values and assign
			int rand1 = rand() % 10;
			int rand2 = rand() % 10;
			*(h_A + i * S + j) = rand1;
			*(h_B + i * S + j) = rand2;
		}//close for j
	}//close for i

	 //allocate device memory
	int* d_A;
	int* d_B;
	int* d_C;
	cudaMalloc((void**)&d_A, hostSize);
	cudaMalloc((void**)&d_B, hostSize);
	cudaMalloc((void**)&d_C, hostSize);

	//time transfer of values and copy the memory
	cudaEventRecord(start1, 0);
	cudaMemcpy(d_A, h_A, hostSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, hostSize, cudaMemcpyHostToDevice);
	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);

	//get the recorded time difference and print it out 
	cudaEventElapsedTime(&gpu_time1, start1, stop1);
	printf("Matrices transfer time: %0.2f \n", gpu_time1);

  //get dimensions of threads and then number of blocks based on num threads
	dim3 threadsPerBlock(S, S, 1);
	dim3 numberOfBlocks(ceil(S / threadsPerBlock.x), ceil(S / threadsPerBlock.y), 1);

	DeviceMatrixMultiplication << <numberOfBlocks, threadsPerBlock >> >(d_A, d_B, d_C, S);

	cudaMemcpy(h_C, d_C, hostSize, cudaMemcpyDeviceToHost);

	HostMatrixMultiplication(h_A, h_B, h_P, S);


	for (int x = 0; x < S; x++) {
		for (int y = 0; y < S; y++) {
			if (*(h_P + x * S + y) != *(h_C + x * S + y))
				flag = 1;
		}
	}
	if (flag == 0)
		printf("Test Passed.");
	else
		printf("Test Failed.");

}//close main
