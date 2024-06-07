#include "GPUMatrix.h"
#include <iostream>

int main(){
	Matrix<int> a(1024, 1024);
	Matrix<int> b(1024, 1024);
	
	Matrix<int> output = a*b;
	a.cpu_mul(b, output);
}
