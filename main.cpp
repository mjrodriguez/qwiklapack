#include "lapackwrapper.h"
#include <memory>
#include <stdio.h>
#include <vector>
#include <random>

std::mt19937& re();
std::mt19937& re() {
  static std::mt19937 r;
  return r;
}

int main() {

    // allocating a matrix of size nxn
    const int n = 3;
    std::unique_ptr<double[]> A(new double[n*n]);
    std::uniform_real_distribution<double> UNIF_DIST(0.0, 5.0);

    // Creating a random column major matrix
    for (int j = 0; j < n; ++j){
        for (int i = 0; i < n; ++i){
            A[i+j*n] = UNIF_DIST(re());
        }
    }

    LU ALU(n,n,A.get());
    ALU.PrintLU();

    // Multiplying invA*A = I
    ALU.Solve(n,n,A.get(),false);

    std::string str="invA*A = ";
    std::vector<int> size{n,n};
    Print(str, size, A.get());



    return 0;
}