#ifndef LAPACKWRAPPER_H
#define LAPACKWRAPPER_H

#include <vector>

extern "C" {
    // void dbdsqr_(char *uplo, int *n, int *ncvt, int *nru, int *ncc, double *D, double *E, double *VT, int *ldvt, double *U, int *ldu, double *C, int *ldc, double *work, int *info);
    // void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);
    // void dgetrs_(char *trans, int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, int *info);
    // void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work, int *lwork, int *info);
    // void dgehrd_(int *n, int*ilo, int *ihi, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
    // void dhseqr_(char *job, char *compz, int* n, int* ilo, int *ihi, double *H, int *ldh, double *WR, double *WI, double *Z, int *ldz, double *work, int *lwork, int *info);

    // void dorghr_(int *n, int *ilo, int *ihi, double *A, int *lda, double *tau, double *work, int *lwork, int *info);

    // void dtrsyl_(char *trana, char *tranb, int *isgn, int *m, int *n, double *A, int *lda, double *B, int *ldb, double *C, int *ldc, double *scale, int *info);
    #include <lapack.h>
    #include <cblas.h>
}

//---------------------------------------------------------------------------
// LAPACK FUNCTIONS
//---------------------------------------------------------------------------


// Compute the bidiagonal SVD
int cdbdsqr(char uplo, int n, int ncvt, int nru, int ncc,
            double *D, double *E, double *VT, int ldvt,
            double *U, int ldu, double *C, int ldc);

// Compute LU factorization of A
int cdgetrf(int m, int n, double *A, int lda, int *ipiv);
// Solve linear system from LU factorization from cdgetrf
int cdgetrs(char trans, int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb);
// Invert matrix given the LU factorization from dgetrf_
int cdgetri(int n, double *A, int lda, int *ipiv);

//-----------------------------------------------------------------
// Computing the Schur Decompotision to apply preconditioner
// --- 1. Reduce to upper Hessenberg using cdgehrd
// --- 2. Generate Orthogonal Matrix using cdorghr
// --- 3. Compute Schur decomposition of Hessenberg Matrix using cdorghr
//----------------------------------------------------------------

// Reduce matrix to upper Hessenberg form
int cdgehrd(int n, int ilo, int ihi, double *A, int lda, double *tau);
// Schur decomposition of a Hessenberg matrix
int cdhseqr(char job, char compz, int n, int ilo, int ihi, double *H, int ldh, double *WR, double *WI, double *Z, int ldz);
// Generate the orthogonal matrix Q implictly created in dgehrd
int cdorghr(int n, int ilo, int ihi, double *A, int lda, double *tau);

// Solve Sylvester System
int cdtrsyl(char trana, char tranb, int isgn, int m, int n, double *A, int lda, double *B, int ldb, double *C, int ldc, double scale);

//-----------------------------------------------------------------------------------
// C++ Class interfaces for LAPACK FUNCTIONS
//-----------------------------------------------------------------------------------

void Print(const std::string str, const std::vector<int> &size, const double *A);
void Transpose(std::vector<int> &size, double *A, double *AT);
// LU decomposition
class LU{
private:
    // Data is stored column-major fashion
    double *data;
    int *ipiv;
public:
    std::vector<int> size; // (number of rows, number of cols)
    LU(const int m_, const int n_, const double *data_);
    ~LU();
    void Factor();
    void SetData(const double *data_);
    void Solve(const int mb, const int nb, double *B, bool trans);
    void GetInverse(const int m, const int n, double *invA);
    void PrintLU();
};


// Schur Decomposition 
class Schur{
private:
    double *Qdata, *Tdata;
public:
    std::vector<int> size;
    Schur(const int m_, const int n_, const double *data);
    ~Schur();
    void Factor();
    double* Q() const;
    double* T() const;
    void PrintQ();
    void PrintT();
};

#endif