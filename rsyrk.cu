#include "cholesky.h"

static int n, k, nb;


// int parseArguments(int argc,char *argv[])
// {
//     if(argc < 4)
//     {
//         printf("Needs m, n and nb as inputs\n");
//         return -1;
//     }
//     n = atoi(argv[1]);
//     k = atoi(argv[2]);
//     nb = atoi(argv[3]);
//     return 0;
// }

__global__
void clearTriSetToOne(char uplo, int m, int n, float *a, int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		if (uplo == 'l') {
			if (i>j) {
				a[i+j*lda] = 1.0;
			}
        } 
        else
        {
            if (i<j)
                a[i+j*lda] = 1.0;
		}
	}
}

void tc_syrk(cublasHandle_t handle, int n, int k, float* A, int lda, float* C, int ldc, __half* Ah, int nb)
{
    // dim3 grid((n+31)/32, (k+31)/32);
    // dim3 block(32,32);
    // s2h<<<grid, block>>>(n, k, A, lda, Ah, n);

    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                nb, nb, k, &snegone,
                                A, CUDA_R_32F, lda, nb,
                                A, CUDA_R_32F, lda, nb,
                                &sone, C, CUDA_R_32F, ldc, nb+nb*lda,
                                n/nb, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    for(int i = 1;n / nb / i / 2 >= 1; i*=2)
    {
        //printf("offset of ah is %d, size is %d, block is %d\n", i*nb, i*nb, n/nb/i/2);
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                    i*nb, i*nb, k, &snegone,
                                    A+i*nb, CUDA_R_32F, lda, 2*i*nb,
                                    A, CUDA_R_32F, lda, 2*i*nb,
                                    &sone, C+i*nb, CUDA_R_32F, ldc, 2*(i*nb+i*nb*lda),
                                    n/nb/i/2, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }

}

// int main(int argc,char *argv[])
// {
//     if(parseArguments(argc, argv)==-1)
//         return 0;
//     cublasHandle_t cublas_handle;
//     cublasCreate(&cublas_handle);

//     // for(int i = 512; i<32768;i+=512){

//     //     k = 512;
//     //     n = i;

//     //     nb = 1024;

//     float *A;
//     cudaMalloc(&A, sizeof(float)*n*k);

//     float *C;
//     cudaMalloc(&C, sizeof(float)*n*n);

//     dim3 gridc((n+31)/32, (n+31)/32);
//     dim3 blockc(32,32);

//     setInitialValue<<<gridc, blockc>>>(n, n ,C, n, 1.0);

//     generateUniformMatrix(A, n, k);

//     __half *hwork;
//     cudaMalloc(&hwork, sizeof(__half)*n*k);
//     startTimer();
//     tc_syrk(cublas_handle, n, k, A, n, C, n, hwork, nb);
//     float ms = stopTimer();
//     printf("tc_syrk %dx%d takes %f ms, flops is %f\n", n, k,ms, 1.0*n*n*k/ms/1e9);
//     clearTriSetToOne<<<gridc, blockc>>>('r', n, n, C, n);

//     //printMatrixDeviceBlock("C.csv", n, n, C, n);
//     float *tC;
//     cudaMalloc(&tC, sizeof(float)*n*n);
//     setInitialValue<<<gridc, blockc>>>(n, n ,tC, n, 1.0);

//     cublasSsyrk(cublas_handle,
//         CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
//         n, k,
//         &snegone,
//         A, n,
//         &sone,
//         tC, n
//     );


//     //printMatrixDeviceBlock("tC.csv", n, n, tC, n);

//     cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
//             &sone, C, n, &snegone, tC, n,
//             C, n);
//     printf("Forward error is %.6e\n",snorm(n, n, C)/snorm(n, n, tC));
//     cudaFree(A);
//     cudaFree(C);
//     cudaFree(hwork);
//     cudaFree(tC);
//     //}

// }


