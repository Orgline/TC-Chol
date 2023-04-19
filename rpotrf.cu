#include "cholesky.h"

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
long int n, nb, trsm_nb, syrk_nb;
int lwork;

int parseArguments(int argc,char *argv[])
{
    if(argc < 3)
    {
        printf("Needs n and nb as inputs\n");
        return -1;
    }
    n = atoi(argv[1]);
    nb = atoi(argv[2]);
    return 0;
}

void tc_rpotrf(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle, int n, float *A, int lda, float *work, __half *hwork, int nb, int *devInfo)
{
    if(n <= nb)
    {
        //printMatrixDeviceBlock("oA22.csv", nb, nb, A, lda);
        cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                        nb, A, lda, work, lwork, devInfo);
        // int info;
        // cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("Info %d\n", info);
        //printMatrixDeviceBlock("L22.csv", nb, nb, A, lda);
        return;
    }


    tc_rpotrf(cublas_handle, cusolver_handle,  n/2, A, lda, work, hwork, nb, devInfo);
    // cublasStrsm(cublas_handle,
    //             CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
    //             CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
    //             n/2, n/2, &sone, A, lda, A+n/2, lda);

    tc_rtrsm(cublas_handle, n/2, n/2, A, lda, A+n/2, lda, hwork, trsm_nb);

    //printMatrixDeviceBlock("A21.csv", n/2, n/2, A+n/2, lda);


    //     cublasSsyrk(cublas_handle,
    //     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
    //     n/2, n/2,
    //     &snegone,
    //     A+n/2, lda,
    //     &sone,
    //     A+n/2+n/2*lda, lda
    // );

    //printMatrixDeviceBlock("A22.csv", n/2, n/2, A+n/2+n/2*lda, lda);
    
    tc_syrk(cublas_handle, n/2, n/2, A+n/2, lda, A+n/2+n/2*lda, lda, hwork, syrk_nb);

    tc_rpotrf(cublas_handle, cusolver_handle, n/2, A+n/2+n/2*lda, lda, work, hwork, nb, devInfo);

    return;
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cusolverDnHandle_t cusolver_handle ;
    cusolverDnCreate(&cusolver_handle);
    // cumpsgemm::handle_t cuMpSGEMM_handle;
    // cumpsgemm::create(cuMpSGEMM_handle);

    float *dA;
    cudaMalloc(&dA,sizeof(float)*long(n*n));
    generateNormalMatrix(dA, n, n);

    trsm_nb = 128;
    syrk_nb = 256;

    if(nb <= trsm_nb)
        trsm_nb = nb;
    if(nb <= syrk_nb)
        syrk_nb = nb;

    float *work;
    cudaMalloc(&work, sizeof(float)*long(n*n));

    startTimer();

    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    n, n, n, 
                    &sone, dA, CUDA_R_32F, n,
                    dA, CUDA_R_32F, n, 
                    &sone, work, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
                    
    float t = stopTimer();
    printf("GEMM %f\n", t);
    dim3 grid0( (n+31)/32, (n+31)/32 );
    dim3 block0( 32, 32 );

    setEyePlus<<<grid0, block0>>>(n, n, work, n);

    //printMatrixDeviceBlock("work.csv", n, n, work, n);

    // setEye<<<grid0, block0>>>(n, n, work, n);

    cudaMemcpy(dA, work, sizeof(float)*long(n)*long(n), cudaMemcpyDeviceToDevice);

    startTimer();
    float Anorm = snorm(n, n ,work);
    float ms = stopTimer();
    printf("norm takes %fms\n", ms);
    printf("Anorm = %f\n", Anorm);
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));

    float *fwork;
    cusolverDnSpotrf_bufferSize(cusolver_handle,
                 CUBLAS_FILL_MODE_LOWER,
                 nb,
                 dA,
                 n,
                 &lwork);
    printf("lwork = %d\n", lwork);
    cudaMalloc(&fwork, sizeof(float)*lwork);

    __half *hwork;
    cudaMalloc(&hwork, sizeof(__half)*long(n*n/4));
    startTimer();
    tc_rpotrf(cublas_handle, cusolver_handle, n, dA, n, fwork, hwork, nb, devInfo);
    ms = stopTimer();

    cudaMemcpy(dA, work, sizeof(float)*long(n)*long(n), cudaMemcpyDeviceToDevice);

    startTimer();
    tc_rpotrf(cublas_handle, cusolver_handle, n, dA, n, fwork, hwork, nb, devInfo);
    ms = stopTimer();
    printf("TC_POTRF size %dx%d takes %fms, rate is %f TFLOPs\n", n, n, ms, 1.0/3.0*n*n*n/1e9/ms);
    
    clearTri<<<grid0, block0>>>('u', n, n, dA, n);


    
    // printMatrixDeviceBlock("test.csv", 32, 32, dA, n);
    // printMatrixDeviceBlock("testwork.csv", 32, 32, work, n);
    

    // generateUniformMatrix(work, n, n);
    // setEyePlus<<<grid0, block0>>>(n, n, work, n);
    
    //printMatrixDeviceBlock("A.csv", n, n, work, n);
    
    
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
         n,n,n, 
         &sone, dA, n, 
         dA, n, 
         &snegone, work, n);

    
    //printMatrixDeviceBlock("R.csv", n, n, work, n);

    printf("Backward error is %6e\n", snorm(n, n, work)/Anorm);

    cudaFree(dA);
    cudaFree(work);
    cudaFree(fwork);
    cudaFree(hwork);

    return 0;
}