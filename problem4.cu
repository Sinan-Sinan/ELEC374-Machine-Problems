//Adam Bayley 20176309 19ahb Machine Problem 4

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <ctime>

#define TILEWIDTH 4
#define S 16

//0 clue if this is correct but it's what I got from the CUDA documentation.
//need 4 case checks and then a running total of the multiplication stuff (saved to temp)
__global__ void TiledMatrixMultiplication(int *a, int *b, int *c, int size) {

	//create the 2 shared thingies
	__shared__ int tile1[TILEWIDTH][TILEWIDTH];
	__shared__ int tile2[TILEWIDTH][TILEWIDTH];

	//get the row col stuff
	int row = blockIdx.y * TILEWIDTH + threadIdx.y;
	int col = blockIdx.x * TILEWIDTH + threadIdx.x;

	int temp = 0; int location;//temp for sum, location for element #

							   //4 cases. based on the cases, the element will either enter tile 1 or tile2.

	for (int i = 0; i< gridDim.x; i++) {//so long as its within the grid dimension (x/y will work bc square matrix,)

		location = (i * TILEWIDTH + threadIdx.y)* size + col; //calculation for the location for tile 2 (input b)
		if (location >= size*size) //not sure if the tile1/2 check should be >=. might just be >?
			tile2[threadIdx.y][threadIdx.x] = 0;
		else
			tile2[threadIdx.y][threadIdx.x] = b[location];

		//now run through a similar process but with tile1 (input 1 in terms of parameters.)

		 location = row * size + i * TILEWIDTH + threadIdx.x;

		if (location >= size*size)//it means its gone outside of bounds, so set it to default at 0
			tile1[threadIdx.y][threadIdx.x] = 0;
		else
			tile1[threadIdx.y][threadIdx.x] = a[location]; //otherwise it is inside the bounds so we can transfer the value


		

		//running total of the multiplication values.
		for (int j = 0; j < TILEWIDTH; j++)
			temp = temp + tile1[threadIdx.y][j] * tile2[j][threadIdx.x];

		__syncthreads(); //idk if this goes inside or outside the for loop. but it synchronizes the shared memory after
						 //doing the computations of location and whatnot.

	}//close for i< gridDim.x

	if (row < size && col < size) {
		int temp2;
		temp2 = row * size + col;
		c[temp2] = temp;
	}//close if row < size...

}//close tiledmatrixmult



 //I dont think tiled host matrix mult exists? could be wrong tho?,...
void HostMatrixMultiplication(int *A, int *B, int *C, int size) { //maybe needs to be a __host__ ??..
	int offset1, offset2;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			float temp = 0;
			for (int k = 0; k < size; k++) {
				offset1 = i*size + k;
				offset2 = k*size + j;

				temp = temp + A[offset1] * B[offset2];
			}//close for k
			C[i*size + j] = temp;
		}//close for j
	}//close for i
}//close hostmatrix


int main() {
	int correctFlag = 0;

	time_t t; //for seeding purposes

			  //memory size for matrix
	size_t size = S*S*sizeof(int);

	//allocate memory for host
	int *h_A = (int*)malloc(size);
	int *h_B = (int*)malloc(size);
	int *h_C = (int*)malloc(size);
	int *h_P = (int*)malloc(size);

	//create pointers for device related stuff, allocate the memory required
	int *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	//seed the random values
	srand((unsigned)time(&t));

	//send in values into the host 2 input matrices
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < S; j++) {
			int rand1 = rand() % 10;
			int rand2 = rand() % 10;
			*(h_A + i * S + j) = rand1;
			*(h_B + i * S + j) = rand2;
		}
	}


	//copy contents of host input matrices to the device
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	//calculation for the
	dim3 threadsPerBlock(S, S, 1);
	dim3 numberOfBlocks(ceil(S / threadsPerBlock.x), ceil(S / threadsPerBlock.y), 1);

	TiledMatrixMultiplication << < numberOfBlocks, threadsPerBlock >> >(d_A, d_B, d_C, S);//need to double check input size stuff?

			//copy back
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	HostMatrixMultiplication(h_A, h_B, h_P, S);



	//equality check, need to sort out tolerances.
	for (int i = 0; i < S; i++) {
		for (int j = 0; j < S; j++) {
			if (*(h_P + i * S + j) != *(h_C + i * S + j))
				correctFlag = 1;
		}
	}

	if (correctFlag == 0)
		printf("Test passed.\n");
	else
		printf("Test failed.\n");


	//free host
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);

	//free device
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}//close main

