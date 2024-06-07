#include <stdio.h>
const int BLOCK_SIZE =8;

template<typename T>
__global__ void dev_vec_matmul(const T *dev_a, const T *dev_b, T *dev_output, int N, int M){
	int ROW = threadIdx.x;
	T temp_value=0;

	for(size_t i=0; i < M; ++i){
		temp_value += dev_a[ROW*M + i]*dev_b[i];
	}
	dev_output[ROW] = temp_value;
}

template<typename T>
__global__ void dev_matmul(const T *a, const T *b, T *output, int N, int M, int K){
	//a is NxK
	//b is KxM
	int bx = blockIdx.x, by = blockIdx.y;	
	int row = threadIdx.y, col = threadIdx.x;
	//loop to find the sub-matrix of output
	//iterates through sub-matrices of A and B to copy to shared memory
	if(by*BLOCK_SIZE+row < N && bx*BLOCK_SIZE+col < M){
		for(int i=0; i < K; i+=BLOCK_SIZE){
			__shared__ T A[BLOCK_SIZE][BLOCK_SIZE], B[BLOCK_SIZE][BLOCK_SIZE]; 

			//copying to shared memory
			for(int j=0; j < BLOCK_SIZE; ++j){
				for(int k=0; k < BLOCK_SIZE; ++k){
					A[j][k] = a[(by*BLOCK_SIZE+j)*K + i+k]; //i+k -> column, by*BLOCK_SIZE+j -> row 
					B[j][k] = b[(i+j)*M + bx*BLOCK_SIZE+k]; //bx*BLOCK_SIZE+k -> column, i+j -> row 
				}
			}


			//wait for completion
			__syncthreads();

			//multiply and add
			T temp_value=0;
			for(int j=0; j < BLOCK_SIZE; ++j){
				temp_value += A[row][j]*B[j][col];
			}

			__syncthreads();
			//if(bx == 0 && by == 0 && row ==0 && col ==0 && i==0){
				//printf("%d\n", temp_value);
			//}

			output[(by*BLOCK_SIZE+row)*M + bx*BLOCK_SIZE+col] += temp_value;

		}
	}
}

void to_avoid_linker_errors_int(const int* a, const int* b, int *c, int N, int M, int K)
{
	dev_matmul<int><<<8, 8>>>(a, b, c, N, M, K);	
}

void to_avoid_linker_errors_float(const float* a, const float* b, float *c, int N, int M, int K)
{
	dev_matmul<float><<<8, 8>>>(a, b, c, N, M, K);	
}

void to_avoid_linker_errors_double(const double* a, const double* b, double *c, int N, int M, int K)
{
	dev_matmul<double><<<8, 8>>>(a, b, c, N, M, K);	
}
