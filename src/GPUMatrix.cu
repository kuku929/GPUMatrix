#include "kernel.cuh" 
#include "dev_vector.h"
#include "GPUMatrix.h"
#include <vector>
#include <array>
#include <iostream>
#include <cstdio>
extern const int BLOCK_SIZE;

template<class T>
void Matrix<T>::vec_matmul(const std::vector<T> &b, std::vector<T> &output){
	//copying data to gpu 
	dev_vector<T> dev_a(*this);
	dev_vector<T> dev_b(b);
	dev_vector<T> dev_output(output.size());
	
	//launching kernel
	dev_vec_matmul<T><<<1, b.size()>>>(dev_a.data(), dev_b.data(), dev_output.data(), this->nrows, this->ncols);	
	cudaDeviceSynchronize();

	//back to cpu
	cudaMemcpy(output.data(), dev_output.begin(), sizeof(T)*output.size(), cudaMemcpyDeviceToHost);
	return;
}

template<class T>
void Matrix<T>::matmul(const Matrix<T> &b, Matrix<T> &output){

	//timing 
	float gpu_elapsed_time_ms;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//copying data to gpu 
	dev_vector<T> dev_a(*this);
	dev_vector<T> dev_b(b);
	dev_vector<T> dev_output(output.nrows*output.ncols);
	
	//launching kernel
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((output.ncols + dim_block.x - 1)/dim_block.x, (output.nrows+dim_block.y - 1)/dim_block.y);
	dev_matmul<T><<<dim_grid, dim_block>>>(dev_a.data(), dev_b.data(), dev_output.data(), output.nrows, output.ncols, b.nrows);	
	cudaDeviceSynchronize();

	//back to cpu
	auto result = cudaMemcpy(output.data(), dev_output.begin(), sizeof(T)*output.size, cudaMemcpyDeviceToHost);

	//timing 
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	std::cout << "gpu time : " << gpu_elapsed_time_ms <<"ms\n";
	if(result != cudaSuccess){
		std::cout << "error!\n";
	}
	return;
}

template<class T>
void Matrix<T>::cpu_mul(const Matrix<T> &b, Matrix<T> &output){
	float cpu_elapsed_time_ms;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(size_t i=0; i < this->nrows; ++i){
		for(size_t j=0; j < b.ncols; ++j){
			for(size_t k=0; k < this->ncols; ++k){
				output.get(i,j) += this->get(i,k)*b.get(k,j);	
			}
		}
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
	std::cout << "cpu time : " << cpu_elapsed_time_ms <<"ms\n";
	return;
}
void to_avoid_linker_errors()
{
	//creating an object and mulitiplying so that the template function matmul is converted 
	//to an actual function at compile time and the linker can find it
	//while linking the source file, Tis' the only way -_-

	Matrix<int> temp_obj;
	std::vector<int> vecobj;
	temp_obj*vecobj;
	Matrix<int> temp_obj1;
	temp_obj*temp_obj1;
	Matrix<int> output;
	temp_obj.cpu_mul(temp_obj1, output);

	Matrix<float> temp_obj2;
	std::vector<float> vecobj2;
	temp_obj2*vecobj2;
	Matrix<float> temp_obj3;
	temp_obj2*temp_obj3;
	Matrix<float> output1;
	temp_obj2.cpu_mul(temp_obj3, output1);

	Matrix<double> temp_obj4;
	std::vector<double> vecobj4;
	temp_obj4*vecobj4;
	Matrix<double> temp_obj5;
	temp_obj4*temp_obj5;
	Matrix<double> output2;
	temp_obj4.cpu_mul(temp_obj5, output2);
}
