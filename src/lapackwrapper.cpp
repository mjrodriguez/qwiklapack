#include "lapackwrapper.h"
#include <memory>
#include <cassert>
#include <iostream>

// Compute Bidiagonal SVD
int cdbdsqr(char uplo, int n, int ncvt, int nru, int ncc,
            double *D, double *E, double *VT, int ldvt,
            double *U, int ldu, double *C, int ldc)
{
  int info;
  std::unique_ptr<double[]> work(new double[4*n]);
  dbdsqr_(&uplo, &n, &ncvt, &nru, &ncc, D, E, VT, &ldvt, U, &ldu, C, &ldc,
          work.get(), &info);
  return info;
}
// Compute LU Factorization of A
int cdgetrf(int m, int n, double *A, int lda, int *ipiv){
    int info;
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}

// Solve linear system from LU factorization from cdgetrf
int cdgetrs(char trans, int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb){
    int info;
    dgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    return info;
}

// Invert matrix given the LU factorization from dgetrf_
int cdgetri(int n, double *A, int lda, int *ipiv){
    int info;
    int lwork = -1;
    double optwork;
    dgetri_(&n, A,&lda, ipiv, &optwork, &lwork, &info);

    lwork = optwork;
    std::unique_ptr<double[]> work(new double[lwork]);
    dgetri_(&n, A,&lda, ipiv, work.get(), &lwork, &info);
    return info;
}

// Solve sylvester system
int cdtrsyl(char trana, char tranb, int isgn, int m, int n, double *A, int lda, double *B, int ldb, double *C, int ldc, double scale){
    int info;
    dtrsyl_(&trana, &tranb, &isgn, &m, &n, A, &lda, B, &ldb, C, &ldc, &scale, &info);
    return info;
}

//-----------------------------------------------------------------
// Computing the Schur Decompotision to apply preconditioner
// --- 1. Reduce to upper Hessenberg using cdgehrd
// --- 2. Generate Orthogonal Matrix using cdorghr
// --- 3. Compute Schur decomposition of Hessenberg Matrix using cdorghr
//----------------------------------------------------------------

// Reduce matrix to upper Hessenberg form
int cdgehrd(int n, int ilo, int ihi, double *A, int lda, double *tau){
    int info;
    int lwork = -1; // For workspace query
    double optwork;
    // find optimal work
    dgehrd_(&n, &ilo, &ihi, A, &lda, tau, &optwork, &lwork, &info);

    lwork = int(optwork);
    std::unique_ptr<double[]> work(new double[lwork]);
    dgehrd_(&n, &ilo, &ihi, A, &lda, tau, work.get(), &lwork, &info);

    return info;
}
// Schur decomposition of a Hessenberg matrix
int cdhseqr(char job, char compz, int n, int ilo, int ihi, double *H, int ldh, double *WR, double *WI, double *Z, int ldz){
    int info;
    int lwork = -1;
    double optwork;
    // Finding optimal work
    dhseqr_(&job, &compz, &n, &ilo, &ihi, H, &ldh, WR, WI, Z, &ldz, &optwork, &lwork, &info);
    lwork = int(optwork);
    std::unique_ptr<double[]> work(new double[lwork]);
    dhseqr_(&job, &compz, &n, &ilo, &ihi, H, &ldh, WR, WI, Z, &ldz, work.get(), &lwork, &info);

    return info;
}
// Generate the orthogonal matrix Q implictly created in dgehrd
int cdorghr(int n, int ilo, int ihi, double *A, int lda, double *tau){
    int info;
    int lwork = -1;
    double optWork;
    // find optimal lwork
    dorghr_(&n, &ilo, &ihi, A, &lda, tau, &optWork, &lwork, &info);
    lwork = int(optWork);
    std::unique_ptr<double[]> work(new double[lwork]);
    dorghr_(&n, &ilo, &ihi, A, &lda, tau, work.get(), &lwork, &info);

    return info;
}



//-----------------------------------------------------------------------------------
// C++ Class interfaces for LAPACK FUNCTIONS
//-----------------------------------------------------------------------------------

// LU Class
LU::LU(const int m_, const int n_, const double *data_){
    size.push_back(m_);
    size.push_back(n_);
    data = new double[size[0]*size[1]];
    ipiv = new int[size[0]];
    SetData(data_);
    Factor();
}

LU::~LU(){
    delete[] data;
    delete[] ipiv;
}

void LU::Factor(){
    // computes LU
    if (cdgetrf(size[0],size[1],data,size[0],ipiv) != 0){
        throw std::runtime_error("LAPACK DGETRF failed to factor.");
    }

}

// Data is column major
void LU::SetData(const double *data_){
    for (int j = 0; j < size[1]; ++j){
        for (int i = 0; i < size[0]; ++i){
            data[i + j*size[0]] = data_[i + j*size[0]];
        }
    }
}

// Solve matrix using LU factorization
void LU::Solve(const int mb, const int nb, double *B, bool trans){
    assert(size[1] == mb);
    char T = trans ? 'T' : 'N';
    if ( cdgetrs(T,size[0], nb, data, size[0], ipiv, B, mb) != 0){
        // THROW ERROR
        throw std::runtime_error("LAPACK DGETRS failed to solve system.");
    }
}

// Get Inverse from LU factorization
void LU::GetInverse(const int m, const int n, double *invA){
    assert(size[0] == m && size[1] == n);
    std::copy(data, data + m*n, invA);
    if (cdgetri(size[0], invA,size[0],ipiv) != 0){
        throw std::runtime_error("LAPACK DGETRI failed to invert matrix.");
    }
}

void LU::PrintLU(){
    std::string str = "LU = ";
    Print(str, size, data);
}

// Schur Class
Schur::Schur(const int m_, const int n_, const double *data){
    size.push_back(m_);
    size.push_back(n_);

    Tdata = new double[size[0]*size[1]];
    Qdata = new double[size[0]*size[1]];

    std::copy(data, data+size[0]*size[1], Tdata);
    Factor();
}

Schur::~Schur(){
    delete[] Tdata;
    delete[] Qdata;
}

void Schur::Factor(){
    int n = size[0];
    int ilo = 1;
    int ihi = n;
    std::unique_ptr<double[]> tau(new double[n-1]), wr(new double[n]), wi(new double[n]);

    // transform to Hessenberg form
    cdgehrd(n, ilo, ihi, Tdata, n, tau.get());
    std::copy(Tdata, Tdata + size[0]*size[1], Qdata);

    // Generate the orthogonal similiraty matrix
    cdorghr(n, ilo, ihi, Qdata, n, tau.get());

    // Transform to quasi-triangular (i.e. complete schur decomposition)
    cdhseqr('S','V', n, 1, n, Tdata, n, wr.get(), wi.get(), Qdata, n);
}

double* Schur::Q() const {
    return Qdata;
}

double* Schur::T() const {
    return Tdata;
}

void Schur::PrintQ(){
    std::string str = "Q = ";
    Print(str, size, Qdata);
}
void Schur::PrintT(){
    std::string str = "T = ";
    Print(str, size, Tdata);
}


// Transpose a matrix
void Transpose(std::vector<int> &size, double *A, double *AT){
    for (int j = 0; j < size[1]; ++j){
        for (int i = 0; i < size[0]; ++i){
            AT[i + j*size[0]] = A[j + i*size[1]];
        }
    }
}

void Print(const std::string str, const std::vector<int> &size, const double *A){
    std::cout << str << std::endl;
    for (int i = 0; i < size[0]; ++i){
        printf("[ ");
        for (int j = 0; j < size[1]; ++j){
            printf("%+e ", A[i + j*size[0]]);
        }
        printf("]\n");
    }
}