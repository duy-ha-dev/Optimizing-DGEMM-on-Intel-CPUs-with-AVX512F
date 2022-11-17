#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]

void scale_c_k2(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

/**
 * Register re-use weirdmatmul
 * - Loads (C(i,j)) into register before k loop
 * - Same problem with wierdmatmul1; additional memory access to A and B
 * - However, since we are only calling A(i, k) twice (and similarly B(k, j) twice), perf diff could be negligible
 * compared to creating a temp
*/
void mydgemm_cpu_v2(int M, int N, int K, double alpha, double *A, int LDA, double *B, int LDB, double beta, double *C, int LDC){
    int i,j,k;
    if (beta != 1.0) scale_c_k2(C,M,N,LDC,beta);
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            double tmp=C(i,j);
            for (k=0;k<K;k++){
                tmp += alpha*(A(i,k) - B(k,j))*(A(i,k) - B(k,j));
            }
            C(i,j) = tmp;
        }
    }
}