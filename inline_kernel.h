#include "immintrin.h"
#include <stdint.h>
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define OUT_M_BLOCKING 1152
#define OUT_N_BLOCKING 9216
#define M_BLOCKING 192
#define N_BLOCKING 96
#define K_BLOCKING 384


void scale_c(double *C,int M, int N, int LDC, double scalar){
    int i,j;
    for (i=0;i<M;i++){
        for (j=0;j<N;j++){
            C(i,j)*=scalar;
        }
    }
}

void packing_b_24x8_edge_version2(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_second;
    for (count_second=0;count_sub>1;count_second+=2,count_sub-=2){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
        }
    }
    for (;count_sub>0;count_second++,count_sub-=1){
        tosrc1=src+count_second*leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
        }
    }
}

void packing_a_24x8_edge(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    //dim_first: M, dim_second: K
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    for (count_first=0;count_sub>23;count_first+=24,count_sub-=24){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            _mm512_store_pd(todst+8,_mm512_loadu_pd(tosrc+8));
            _mm512_store_pd(todst+16,_mm512_loadu_pd(tosrc+16));
            tosrc+=leading_dim;
            todst+=24;
        }
    }
    // edge case
    for (;count_sub>7;count_first+=8,count_sub-=8){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=8;
        }
    }
    for (;count_sub>1;count_first+=2,count_sub-=2){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            _mm_store_pd(todst,_mm_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=2;
        }
    }
    for (;count_sub>0;count_first+=1,count_sub-=1){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            *todst=*tosrc;
            tosrc+=leading_dim;
            todst++;
        }
    }
}

#define LOAD_A_COL \
  "vmovaps (%0),%%zmm1;vmovaps 64(%0),%%zmm2;vmovaps 128(%0),%%zmm3;addq $192,%0;"
#define KERNEL_m24n8 \
  LOAD_A_COL \
  "vbroadcastsd (%1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm8; vfmadd231pd %%zmm2,%%zmm4,%%zmm9; vfmadd231pd %%zmm3,%%zmm4,%%zmm10;"\
  "vbroadcastsd 8(%1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm11; vfmadd231pd %%zmm2,%%zmm5,%%zmm12; vfmadd231pd %%zmm3,%%zmm5,%%zmm13;"\
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm14; vfmadd231pd %%zmm2,%%zmm4,%%zmm15; vfmadd231pd %%zmm3,%%zmm4,%%zmm16;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm17; vfmadd231pd %%zmm2,%%zmm5,%%zmm18; vfmadd231pd %%zmm3,%%zmm5,%%zmm19;"\
  "vbroadcastsd (%1,%%r11,2),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm20; vfmadd231pd %%zmm2,%%zmm4,%%zmm21; vfmadd231pd %%zmm3,%%zmm4,%%zmm22;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm23; vfmadd231pd %%zmm2,%%zmm5,%%zmm24; vfmadd231pd %%zmm3,%%zmm5,%%zmm25;"\
  "vbroadcastsd (%%r12),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm26; vfmadd231pd %%zmm2,%%zmm4,%%zmm27; vfmadd231pd %%zmm3,%%zmm4,%%zmm28;"\
  "vbroadcastsd 8(%%r12),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm29; vfmadd231pd %%zmm2,%%zmm5,%%zmm30; vfmadd231pd %%zmm3,%%zmm5,%%zmm31;"\
  "addq $16,%1;addq $16,%%r12;"
#define SAVE_m24n8 \
  "vaddpd (%2),%%zmm8,%%zmm8;vaddpd 64(%2),%%zmm9,%%zmm9;vaddpd 128(%2),%%zmm10,%%zmm10;"\
  "vmovups %%zmm8,(%2);vmovups %%zmm9,64(%2);vmovups %%zmm10,128(%2);"\
  "vaddpd (%2,%3,1),%%zmm11,%%zmm11;vaddpd 64(%2,%3,1),%%zmm12,%%zmm12;vaddpd 128(%2,%3,1),%%zmm13,%%zmm13;"\
  "vmovups %%zmm11,(%2,%3,1);vmovups %%zmm12,64(%2,%3,1);vmovups %%zmm13,128(%2,%3,1);leaq (%2,%3,2),%2;"\
  "vaddpd (%2),%%zmm14,%%zmm14;vaddpd 64(%2),%%zmm15,%%zmm15;vaddpd 128(%2),%%zmm16,%%zmm16;"\
  "vmovups %%zmm14,(%2);vmovups %%zmm15,64(%2);vmovups %%zmm16,128(%2);"\
  "vaddpd (%2,%3,1),%%zmm17,%%zmm17;vaddpd 64(%2,%3,1),%%zmm18,%%zmm18;vaddpd 128(%2,%3,1),%%zmm19,%%zmm19;"\
  "vmovups %%zmm17,(%2,%3,1);vmovups %%zmm18,64(%2,%3,1);vmovups %%zmm19,128(%2,%3,1);leaq (%2,%3,2),%2;"\
  "vaddpd (%2),%%zmm20,%%zmm20;vaddpd 64(%2),%%zmm21,%%zmm21;vaddpd 128(%2),%%zmm22,%%zmm22;"\
  "vmovups %%zmm20,(%2);vmovups %%zmm21,64(%2);vmovups %%zmm22,128(%2);"\
  "vaddpd (%2,%3,1),%%zmm23,%%zmm23;vaddpd 64(%2,%3,1),%%zmm24,%%zmm24;vaddpd 128(%2,%3,1),%%zmm25,%%zmm25;"\
  "vmovups %%zmm23,(%2,%3,1);vmovups %%zmm24,64(%2,%3,1);vmovups %%zmm25,128(%2,%3,1);leaq (%2,%3,2),%2;"\
  "vaddpd (%2),%%zmm26,%%zmm26;vaddpd 64(%2),%%zmm27,%%zmm27;vaddpd 128(%2),%%zmm28,%%zmm28;"\
  "vmovups %%zmm26,(%2);vmovups %%zmm27,64(%2);vmovups %%zmm28,128(%2);"\
  "vaddpd (%2,%3,1),%%zmm29,%%zmm29;vaddpd 64(%2,%3,1),%%zmm30,%%zmm30;vaddpd 128(%2,%3,1),%%zmm31,%%zmm31;"\
  "vmovups %%zmm29,(%2,%3,1);vmovups %%zmm30,64(%2,%3,1);vmovups %%zmm31,128(%2,%3,1);leaq (%2,%3,2),%2;"

void micro_kernel_24x8(double *a_ptr, double *b_ptr, double *c_ptr, int64_t ldc_in_bytes, int64_t K){
    __asm__ __volatile__(
        "movq %4,%%r10;movq %4,%%r11;\n\t"
        "salq $4,%%r11;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %4,%%r13;\n\t"
        "vpxorq %%zmm8,%%zmm8,%%zmm8;vpxorq %%zmm9,%%zmm9,%%zmm9;vpxorq %%zmm10,%%zmm10,%%zmm10;\n\t"
        "vpxorq %%zmm11,%%zmm11,%%zmm11;vpxorq %%zmm12,%%zmm12,%%zmm12;vpxorq %%zmm13,%%zmm13,%%zmm13;\n\t"
        "vpxorq %%zmm14,%%zmm14,%%zmm14;vpxorq %%zmm15,%%zmm15,%%zmm15;vpxorq %%zmm16,%%zmm16,%%zmm16;\n\t"
        "vpxorq %%zmm17,%%zmm17,%%zmm17;vpxorq %%zmm18,%%zmm18,%%zmm18;vpxorq %%zmm19,%%zmm19,%%zmm19;\n\t"
        "vpxorq %%zmm20,%%zmm20,%%zmm20;vpxorq %%zmm21,%%zmm21,%%zmm21;vpxorq %%zmm22,%%zmm22,%%zmm22;\n\t"
        "vpxorq %%zmm23,%%zmm23,%%zmm23;vpxorq %%zmm24,%%zmm24,%%zmm24;vpxorq %%zmm25,%%zmm25,%%zmm25;\n\t"
        "vpxorq %%zmm26,%%zmm26,%%zmm26;vpxorq %%zmm27,%%zmm27,%%zmm27;vpxorq %%zmm28,%%zmm28,%%zmm28;\n\t"
        "vpxorq %%zmm29,%%zmm29,%%zmm29;vpxorq %%zmm30,%%zmm30,%%zmm30;vpxorq %%zmm31,%%zmm31,%%zmm31;\n\t"
        "cmpq $4,%%r13;jb 2f;\n\t"
        "1:\n\t"//main kernel loop
        KERNEL_m24n8 \
        KERNEL_m24n8 \
        KERNEL_m24n8 \
        KERNEL_m24n8 \
        "subq $4,%%r13;cmpq $4,%%r13;jnb 1b;\n\t"
        "cmpq $0,%%r13;je 3f;\n\t"
        "2:\n\t"
        KERNEL_m24n8 \
        "decq %%r13;cmpq $0,%%r13;jnb 2b;\n\t"
        "3:\n\t"
        SAVE_m24n8 \
        "4:\n\t"
        :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(ldc_in_bytes)
        :"m"(K)
        :"r10","r11","r12","r13","cc","memory"
    );
}

void macro_kernel(double *a_buffer,double *b_buffer,int m,int n,int k,double *C, int LDC,int k_inc,double alpha){
    int m_count,n_count,m_count_sub,n_count_sub;
    if (m==0||n==0||k==0) return;
    int64_t M=(int64_t)m,K=(int64_t)k,ldc_in_bytes=(int64_t)LDC*sizeof(double);
    double *a_ptr,*b_ptr=b_buffer,*c_ptr=C;
    double ALPHA=alpha;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);
    for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
        //call the m layer with n=8;
        a_ptr=a_buffer;b_ptr=b_buffer+n_count*k;c_ptr=C+n_count*LDC;
        for (m_count_sub=m,m_count=0;m_count_sub>23;m_count_sub-=24,m_count+=24){
            //call the micro kernel: m24n8;
            a_ptr=a_buffer+m_count*K;
            b_ptr=b_buffer+n_count*k;
            c_ptr=C+n_count*LDC+m_count;
            micro_kernel_24x8(a_ptr,b_ptr,c_ptr,ldc_in_bytes,K);
        }
        for (;m_count_sub>7;m_count_sub-=8,m_count+=8){
            //call the micro kernel: m8n8;
        }
        for (;m_count_sub>1;m_count_sub-=2,m_count+=2){
            //call the micro kernel: m2n8;
        }
        for (;m_count_sub>0;m_count_sub-=1,m_count+=1){
            //call the micro kernel: m1n8;
        }
    }
    for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
        //call the m layer with n=4
        //kernel_n_4_v2(a_buffer,b_buffer+n_count*k_inc,C+n_count*LDC,m,k_inc,LDC,alpha);
    }
    for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
        //call the m layer with n=2
    }
    for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
        //call the m layer with n=1
    }
}


void dgemm_asm(\
    int M, \
    int N, \
    int K, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB, \
    double beta, \
    double *C, \
    int LDC)\
{
    int i,j,k;
    if (beta != 1.0) scale_c(C,M,N,LDC,beta);
    if (alpha == 0.||K==0) return;
    int M4,N8=N&-8,K4;
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_M_BLOCKING*sizeof(double));
    int second_m_count,second_n_count,second_m_inc,second_n_inc;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc;
    for (k_count=0;k_count<K;k_count+=k_inc){
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N;n_count+=n_inc){
            n_inc=(N-n_count>OUT_N_BLOCKING)?OUT_N_BLOCKING:N-n_count;
            packing_b_24x8_edge_version2(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc);
            //print_matrix(b_buffer,k_inc,n_inc);
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc=(M-m_count>OUT_M_BLOCKING)?OUT_M_BLOCKING:M-m_count;
                packing_a_24x8_edge(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
                //print_matrix(a_buffer,m_inc,k_inc);
                for (second_m_count=m_count;second_m_count<m_count+m_inc;second_m_count+=second_m_inc){
                    second_m_inc=(m_count+m_inc-second_m_count>M_BLOCKING)?M_BLOCKING:m_count+m_inc-second_m_count;
                    for (second_n_count=n_count;second_n_count<n_count+n_inc;second_n_count+=second_n_inc){
                        second_n_inc=(n_count+n_inc-second_n_count>N_BLOCKING)?N_BLOCKING:n_count+n_inc-second_n_count;
                        macro_kernel(a_buffer+(second_m_count-m_count)*k_inc,b_buffer+(second_n_count-n_count)*k_inc,second_m_inc,second_n_inc,k_inc,&C(second_m_count,second_n_count),LDC,k_inc,alpha);
                        //printf("m=%d,m_inc=%d,n=%d,n_inc=%d\n",second_m_count,second_m_inc,second_n_count,second_n_inc);
                    }
                }
            }
        }
    }
    free(a_buffer);free(b_buffer);
}