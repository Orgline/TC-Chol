#include "cholesky.h"
static int m, n, nb;

static int parseArguments(int argc,char *argv[])
{
    if(argc < 3)
    {
        printf("Needs n and nb as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nb = atoi(argv[3]);
    return 0;
}

void tc_btrsm(cublasHandle_t handle, int m, int n, float* A, int lda, float* B, int ldb, __half* hwork, int nb)
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
    
    tc_btrsm(handle, m, nb, A, lda, B, ldb, hwork, nb);

    __half *Ah = hwork;
    __half *Bh = hwork+nb*(n-nb);

    dim3 grid((n-nb+31)/32, (nb+31)/32);
    dim3 block(32,32);
    s2h<<<grid, block>>>(n - nb, nb, A+nb, lda, Ah, n-nb);

    dim3 grid1((m+31)/32, (nb+31)/32);
    dim3 block1(32,32);
    s2h<<<grid1, block1>>>(m, nb, B, ldb, Bh, m);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n-nb, nb,
        &snegone, Bh, CUDA_R_16F, m, Ah, CUDA_R_16F, n-nb,
        &sone, B+nb*ldb, CUDA_R_32F, ldb, CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    tc_btrsm(handle, m, n-nb, A+nb*lda+nb, lda, B+nb*ldb, ldb, hwork, nb);
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    float *A;
    cudaMalloc(&A, sizeof(float)*n*n);
    float *B;
    cudaMalloc(&B, sizeof(float)*m*n);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*(m*n));

    generateUniformMatrix(A, n, n);
    generateNormalMatrix(B, m, n);
    // dim3 gridb((m+31)/32, (n+31)/32);
    // dim3 blockb(32,32);
    // setInitialValue<<<gridb, blockb>>>(m, n ,B, m, 1.0);

    dim3 grid((n+31)/32, (n+31)/32);
    dim3 block(32,32);
    setInitialValue<<<grid, block>>>(n, n ,A, n, 0.1);
    clearTri<<<grid, block>>>('u', n, n, A, n);
    //printMatrixDeviceBlock("A.csv", n, n, A, n);

    float *work;
    cudaMalloc(&work, sizeof(float)*m*n);
    cudaMemcpy(work, B, sizeof(float)*m*n, cudaMemcpyDeviceToDevice);
    

    startTimer();
    tc_btrsm(cublas_handle, m, n, A, n, B, m, hwork, nb);
    float ms = stopTimer();

    printf("btrsm takes %f ms, flops is %f\n", ms, 1.0*m*n*n/ms/1e9);
    float normB = snorm(m, n, work);
    float normA = snorm(n, n, A);
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, m,
            &sone, B, m, A, n,
            &snegone, work, m
        );

    printf("Backward error ||X*L^T-B||/||B|| is %.6e\n", snorm(m,n,work)/normB);

}