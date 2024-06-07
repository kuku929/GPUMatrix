#pragma once
#include <iostream>
#include <vector>
#include <array>
#include <cstdio>
#include <stdexcept>

template <class T>
class Matrix{
	/*
	 * this class supports GPU multiplication
	 * it is implemented as a 1-D vector with some objects like nrows, ncols for ease of use
	 * the GPU code is contained in the matmul() and vec_matmul() functions
	 * for Matrix<T> a,b
	 * cache-tiled multiplication can be used by doing a*b
	 */
	public:
		size_t nrows, ncols;
		size_t size;
		Matrix(): nrows(0), ncols(0), size(0){};
		Matrix(int N, int M): nrows(N), ncols(M), size(N*M){
			data_.resize(N*M);
		}
		
		
		const T& get(int row, int col)const{
			/*
			 * returns (i,j)th element in the matrix
			 * used when value does not need to be changed
			 */
			return data_[row*ncols + col];
		}


		T& get(int row, int col){
			/*
			 * returns (i,j)th element in the matrix
			 * used when value needs to be changed
			 */
			return data_[row*ncols + col];
		}

		T* data(){
			/*
			 * returns pointer to the first element of the vector 
			 * used while copying data from device memory to host
			 */
			return &data_[0];
		}

		const T* data() const{
			/*
			 * returns pointer to the first element of the vector 
			 * used when the matrix is not being updated 
			 */
			return data_.data();
		}

		void show() const{
			/*
			 * prints the matrix
			 */
			for(int i=0; i < nrows; ++i){
				for(int j=0; j < ncols; ++j){
					std::cout << data_[i*ncols + j] << ' ';
				}
				std::cout << '\n';
			}
		}


		void cpu_mul(const Matrix<T> &b, Matrix<T> &output);
		void vec_matmul(const std::vector<T> &b, std::vector<T> &output);
		void matmul(Matrix<T> const &b, Matrix<T> &output);

		Matrix<T> operator*(const Matrix<T> &second){
			/*
			 *GPU accelerated multiplication with another matrix
			 */
			if(this->ncols != second.nrows)
				throw std::invalid_argument("dimensions dont match!");
			Matrix output(this->nrows, second.ncols);
			matmul(second, output);
			return output;
		}

		std::vector<T> operator*(const std::vector<T> &second){
			/*
			 *GPU accelerated multiplication with a vector
			 */
			if(this->ncols != second.size())
				throw std::invalid_argument("dimensions dont match!");
			std::vector<T> output(this->nrows);
			vec_matmul(second, output);
			return output;
		}


	private:
		std::vector<T> data_;
};
