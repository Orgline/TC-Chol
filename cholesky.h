#include <iostream> 
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#include <cusolverDn.h>
// #include <cumpsgemm/cumpsgemm.hpp>


__global__
void s2h(int m, int n, float *as, int ldas, __half *ah, int ldah);

__global__
void clearTri(char uplo, long int m, long int n, float *a, long int lda);

__global__
void clearTri(char uplo, int m, int n, double *a, int lda);

__global__
void setEyePlus( long int m, long int n, double *a, long int lda);

__global__
void setEyePlus( long int m, long int n, float *a, long int lda);

__global__
void setEye( int m, int n, float *a, int lda);

__global__
void setInitialValue( int m, int n, float *a, int lda, float val);

void startTimer();

float stopTimer();

void generateNormalMatrix(float *dA,long int m,long int n);

void generateNormalMatrix(double *dA,int m,int n);

void generateUniformMatrix(float *dA,int m,int n);

void generateUniformMatrix(double *dA,int m,int n);

float snorm(long int m,long int n,float* dA);

double dnorm(int m, int n, double *dA);

template<typename T>
void printMatrixDeviceBlock(char *filename,int m, int n, T* dA, int lda)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    //printf("Perform printmatrixdevice\n");
    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
        }
    }
    fclose(f);
	//cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    //printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)


void print_env();

void tc_rtrsm(cublasHandle_t handle, int m, int n, float* A, int lda, float* B, int ldb, __half* hwork, int nb);

void tc_syrk(cublasHandle_t handle, int n, int k, float* A, int lda, float* C, int ldc, __half* Ah, int nb);

// void tc_rtrsm(cublasHandle_t handle, cumpsgemm::handle_t cuMpSGEMM_handle, int m, int n, float* A, int lda, float* B, int ldb, __half* hwork, int nb);

// void tc_syrk(cublasHandle_t handle,  cumpsgemm::handle_t cuMpSGEMM_handle, int n, int k, float* A, int lda, float* C, int ldc, __half* Ah, int nb);

const float sone = 1.0;
const float snegone = -1.0;
const float szero = 0.0;