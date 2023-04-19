#include "cholesky.h"

int n, nb;
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

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    cublasHandle_t cublas_handle_0;
    cublasCreate(&cublas_handle_0);
    cublasHandle_t cublas_handle_1;
    cublasCreate(&cublas_handle_1);
    cublasHandle_t cublas_handle_2;
    cublasCreate(&cublas_handle_2);
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    cudaStream_t stream0;
    cudaStreamCreate(&stream0);
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);

    cublasSetStream(cublas_handle_0,stream0);
    cublasSetStream(cublas_handle_1,stream1);
    cublasSetStream(cublas_handle_2,stream2);
    cusolverDnSetStream(cusolver_handle, stream0);


    double done = 1.0;
    //double dzero = 0.0;
    double dnegone = -1.0;

    double *dA;
    cudaMalloc(&dA,sizeof(double)*n*n);
    // double *dB;
    // cudaMalloc(&dB,sizeof(double)*n*n);

    dim3 grid0( (n+31)/32, (n+31)/32 );
    dim3 block0( 32, 32 );    
    
    generateNormalMatrix(dA, n, n);

    double *work;
    cudaMalloc(&work, sizeof(double)*n*n);

    cublasGemmEx(cublas_handle_0, CUBLAS_OP_N, CUBLAS_OP_T,
                    n, n, n, 
                    &done, dA, CUDA_R_64F, n,
                    dA, CUDA_R_64F, n, 
                    &done, work, CUDA_R_64F, n,
                    CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaMemcpy(dA, work, sizeof(double)*n*n, cudaMemcpyDeviceToDevice);

    setEyePlus<<<grid0, block0>>>(n, n, dA, n);

    cudaMemcpy(work, dA, sizeof(double)*n*n, cudaMemcpyDeviceToDevice);
    int lwork;
    startTimer();
    cusolverDnDpotrf_bufferSize(cusolver_handle,
                 CUBLAS_FILL_MODE_LOWER,
                 n,
                 dA,
                 n,
                 &lwork);
    printf("lwork = %d\n", lwork);
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));

    double *dwork;
    cudaMalloc(&dwork, sizeof(double)*lwork);

    int lda = n;
    float potf_time = 0;
    float syrk_time = 0;
    float trsm_time = 0;
    float ms = 0;
    
    startTimer();
    for(int i = 0; i <= n - nb; i += nb)
    {
        
        if(i > 0)
        {
            cublasDsyrk(cublas_handle_0,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, nb, nb,
                &dnegone, dA+i+(i-nb)*lda, lda, 
                &done, dA+i+i*lda, lda);
            if(n - i - nb > 0)
            {
                // printf("GEMM size %dx%dx%d\n", n-i-nb, nb, nb);
                cublasDgemm(cublas_handle_0, CUBLAS_OP_N, CUBLAS_OP_T,
                            n-i-nb, nb, nb, 
                            &dnegone, dA+i+nb+(i-nb)*lda, lda, 
                            dA+i+(i-nb)*lda, lda, 
                            &done, dA+i+nb+i*lda, lda);
            
                // printf("SYRK size %dx%d\n", n - i - nb, nb);
            
            
                cublasDsyrk(cublas_handle_2,
                            CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N, n - i - nb, nb,
                            &dnegone, dA+i+nb+(i-nb)*lda, lda, 
                            &done, dA+i+nb+(i+nb)*lda, lda);
            }
            
        }
        //startTimer();
        cusolverDnDpotrf(cusolver_handle,
           CUBLAS_FILL_MODE_LOWER,
           nb,
           dA+i+i*lda,
           n,
           dwork,
           lwork,
           devInfo);
        //ms = stopTimer();
        //potf_time += ms;
        //printf("potrf takes %fms\n", ms);
        if(i == n - nb)
            break;
        //startTimer();
        // printf("TRSM size %dx%d\n", n - i - nb, nb);
        //cudaStreamSynchronize(stream1);
        cublasDtrsm(cublas_handle_0,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                n - i - nb, nb, &done, dA+i+i*lda, lda, dA+i+nb+i*lda, lda);
        //ms = stopTimer();
        //trsm_time += ms;
        //printf("trsm takes %fms\n", ms);
        //cudaDeviceSynchronize();
        cudaStreamSynchronize(stream0);
        //cudaStreamSynchronize(stream2);
        
    }
    float flops;
    flops = 1.0/3.0*n*n*n;
    // printf("DTCPOTRF size %dx%d takes %fms, rate is %f TFLOPs\n", n, n, (potf_time + trsm_time + syrk_time), flops/1e9/(potf_time + trsm_time + syrk_time));
    // printf("POTF takes %fms\n", potf_time);
    // printf("TRSM takes %fms\n", trsm_time);
    // printf("SYRK takes %fms\n", syrk_time);
    ms = stopTimer();
    printf("DTCPOTRF size %dx%d takes %fms, rate is %f TFLOPs\n", n, n, ms, flops/1e9/ms);
    clearTri<<<grid0, block0>>>('u', n, n, dA, n);

    double A_norm = dnorm(n, n, work);
    cublasDgemm(cublas_handle_0, CUBLAS_OP_N, CUBLAS_OP_T,
         n,n,n, 
         &done, dA, n, 
         dA, n, 
         &dnegone, work, n);
    
    printf("A norm is %.3e, res norm is %.3e\n", A_norm, dnorm(n, n ,work)/A_norm);
    
    cudaFree(work);
    cudaFree(dA);
    cudaFree(dwork);
    
    return 0;
}