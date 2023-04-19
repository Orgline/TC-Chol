#include "cholesky.h"
static int m, n, nb;

// int parseArguments(int argc,char *argv[])
// {
//     if(argc < 4)
//     {
//         printf("Needs m, n and nb as inputs\n");
//         return -1;
//     }
//     m = atoi(argv[1]);
//     n = atoi(argv[2]);
//     nb = atoi(argv[3]);
//     return 0;
// }

void tc_rtrsm(cublasHandle_t handle, int m, int n, float* A, int lda, float* B, int ldb, __half* hwork, int nb)
{
    if(n <= nb)
    {
        //startTimer();
        cublasStrsm(handle,
            CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            m, n, &sone,
            A, lda,
            B, ldb
        );
        return;
    }
    
    tc_rtrsm(handle, m, n/2, A, lda, B, ldb, hwork, nb);

    // __half *Ah = hwork;
    // __half *Bh = hwork+n/2*n/2;

    // dim3 grid((n/2+31)/32, (n/2+31)/32);
    // dim3 block(32,32);
    // s2h<<<grid, block>>>(n/2, n/2, A+n/2, lda, Ah, n/2);

    // dim3 grid1((m+31)/32, (n/2+31)/32);
    // dim3 block1(32,32);
    // s2h<<<grid1, block1>>>(m, n/2, B, ldb, Bh, m);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n/2, n/2,
        &snegone, B, CUDA_R_32F, ldb, A+n/2, CUDA_R_32F, lda,
        &sone, B+n/2*ldb, CUDA_R_32F, ldb, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );
    // cumpsgemm::gemm(
	// 				cuMpSGEMM_handle,
	// 				CUBLAS_OP_N, CUBLAS_OP_T, // cublasOperation_t
	// 				m, n/2, n/2,
	// 				&snegone,
	// 				B, ldb,
	// 				A+n/2, lda,
	// 				&sone,
	// 				B+n/2*ldb, ldb,
	// 				CUMPSGEMM_FP16TC);

    tc_rtrsm(handle, m, n/2, A+n/2*lda+n/2, lda, B+n/2*ldb, ldb, hwork, nb);
}

// int main(int argc,char *argv[])
// {
//     if(parseArguments(argc, argv)==-1)
//         return 0;
//     cublasHandle_t cublas_handle;
//     cublasCreate(&cublas_handle);

//     float *A;
//     cudaMalloc(&A, sizeof(float)*n*n);
//     float *B;
//     cudaMalloc(&B, sizeof(float)*m*n);

//     __half *hwork;
//     cudaMalloc(&hwork, sizeof(__half)*(n/2*n/2+m/2*n));

//     generateUniformMatrix(A, n, n);
//     generateNormalMatrix(B, m, n);
//     // dim3 gridb((m+31)/32, (n+31)/32);
//     // dim3 blockb(32,32);
//     // setInitialValue<<<gridb, blockb>>>(m, n ,B, m, 1.0);

//     dim3 grid((n+31)/32, (n+31)/32);
//     dim3 block(32,32);
//     setInitialValue<<<grid, block>>>(n, n ,A, n, 0.1);
//     clearTri<<<grid, block>>>('u', n, n, A, n);
//     //printMatrixDeviceBlock("A.csv", n, n, A, n);

//     float *work;
//     cudaMalloc(&work, sizeof(float)*m*n);
//     cudaMemcpy(work, B, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);
    

//     startTimer();
//     tc_rtrsm(cublas_handle, m, n, A, n, B, m, hwork, nb);
//     float ms = stopTimer();

//     printf("rtrsm takes %f ms, flops is %f\n", ms, 1.0*m*n*n/ms/1e9);
//     float normB = snorm(m, n, work);
//     float normA = snorm(n, n, A);
//     cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, m,
//             &sone, B, m, A, n,
//             &snegone, work, m
//         );

//     printf("Backward error ||X*L^T-B||/||B|| is %.6e\n", snorm(m,n,work)/normB);

// }