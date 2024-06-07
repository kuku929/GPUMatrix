#pragma once
#include "GPUMatrix.h"
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <iostream>
template <class T>
class dev_vector{
	public:
		explicit dev_vector(): start_(0), end_(0){}
		//constructor
		explicit dev_vector(size_t size){
			allocate(size);
		}

		explicit dev_vector(const std::vector<T> &host_vector){
			allocate(host_vector.size());
			set(host_vector.data(), host_vector.size());
		}

		explicit dev_vector(const Matrix<T> &host_matrix){
			allocate(host_matrix.size);
			set(host_matrix.data(), host_matrix.size);
		}

		explicit dev_vector(const std::vector<std::vector<T>> &host_vector){
			size_t N = host_vector.size();
			size_t M = host_vector[0].size();
			convert_matrix_to_vector(host_vector,N*M);	
		}

		~dev_vector(){
			free();
		}

		const T* begin() const{
			return start_;
		}

		__host__ __device__ T* data() const{
			return start_;
		}


		T* end() const{
			return end_;
		}

		size_t size() const{
			return end_ - start_;
		}

		void set(const T* src, size_t size){
			size_t min = std::min(size, this->size());

			cudaError_t result = cudaMemcpy(start_, src, min*sizeof(T), cudaMemcpyHostToDevice);
			if(result != cudaSuccess){
				throw std::runtime_error("failed to copy to device!");
			}
		}



		void get(T* dest, size_t size){
			size_t min = std::min(size, this->size());
			cudaError_t result = cudaMemcpy(dest, start_, min*sizeof(T), cudaMemcpyDeviceToHost);
			if(result != cudaSuccess){
				throw std::runtime_error("failed to copy to host!");
			}

		}

		void convert_matrix_to_vector(std::vector<std::vector<T>> &host_vector, size_t total_size){
			extend(total_size);
			for(const auto& vec : host_vector){
				push(vec.data(),  vec.size());
			}
		}

		//__device__ const T& operator[](int index) const{
			//return start_[index];
		//}

		//__device__ const T& operator[](size_t index) const{
			//return start_[index];
		//}

		//__device__ T& operator[](int index){
			//return start_[index];
		//}

		//__device__ T& operator[](size_t index){
			//return start_[index];
		//}

	private:
		void allocate(size_t size){
			cudaError_t result = cudaMalloc((void**)&start_, size*sizeof(T));
			if(result != cudaSuccess){
				start_=end_=0;
				throw std::runtime_error("failed to copy to host!");
			}
			end_=start_+size;
		}

		void extend(size_t size){
			cudaError_t result = cudaMalloc((void**)&start_, size*sizeof(T));
			if(result != cudaSuccess){
				start_=end_=0;
				throw std::runtime_error("failed to copy to host!");
			}
			end_=start_;
		}

		void push(const T* src, size_t size){
			cudaError_t result = cudaMemcpy(end_, src, size*sizeof(T), cudaMemcpyHostToDevice);
			if(result != cudaSuccess){
				throw std::runtime_error("failed to copy to device!");
			}
			end_+=size;
		}


		void push_back(T* src, size_t size){
			
		}

		void free(){
			if(start_!=0){
				cudaFree(start_);
				start_=end_=0;
			}
		}

		T* start_;
		T* end_;
};
