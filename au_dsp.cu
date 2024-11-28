#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024
#define MAX_ERR 1e-6
static const int irlen = 1024;
__global__ void outside_convo(double *out_base, double *sig_base, double *ir_base) {
  int out_idx = threadIdx.x;
  double suma = 0.0;
  for (int i=0; i<irlen; i++) {
    int sig_idx = out_idx-i;
    suma += sig_base[sig_idx]*ir_base[i];
  }
  out_base[out_idx] = suma; 
}

__global__ void outside_fma_convo(double *out_base, double *sig_base, double *ir_base) {
  int out_idx = threadIdx.x;
  double suma = 0.0;
  for (int i=0; i<irlen; i++) {
    int sig_idx = out_idx-i;
    suma = fma(sig_base[sig_idx], ir_base[i], suma);
  }
  out_base[out_idx] = suma;
}

__global__ void polar_convo(double *out_base, double *sig_base, double *ir_base) {
  int out_idx = threadIdx.x*2;
  double mag0 = sig_base[out_idx];
  double mag1 = ir_base[out_idx];
  out_base[out_idx] = __dmul_rn(mag0,mag1);
  double phs0 = sig_base[out_idx+1];
  double phs1 = ir_base[out_idx+1];
  out_base[out_idx+1] =__dadd_rn(phs0, phs1);
}

__global__ void idft(double *out, double *a, double *b) {
      float sum = 0.0;
      for (int j=0; j<N; j ++){
	        sum += (a[j]*sin(3.141597*threadIdx.x*j/N)/(N/2) - b[j]*cos(2*3.14159*threadIdx.x*j/N)/(N/2));
      }
      out[threadIdx.x]=sum + b[0]/N + b[N/2]*cos(2*3.14159*threadIdx.x*N)/N;
      // }
}
__global__ void haar_blur(double *out, double *a) {
  int xx = threadIdx.x;
  out[xx]=a[xx*2]+a[xx*2+1];
}
__global__ void haar_detail(double *out, double *a) {
  int xx = threadIdx.x;
  out[xx]=a[xx*2]-a[xx*2+1];
}

int main(){
    double *a, *b, *out;
    double *d_a, *d_b, *d_out, *wlet_blur_out, *wlet_detail_out; 

    // Allocate host memory
    a   = (double*)malloc(sizeof(double) * N/2);
    b   = (double*)malloc(sizeof(double) * N/2);
    out = (double*)malloc(sizeof(double) * N);
    //    wletout = (float*)malloc(sizeof(float) * N);

   
    // Initialize host arrays
    for(int i = 0; i < N/2; i++){
      a[i] = 0.0f;
      b[i] = 0.0f;
    }
    for (int i = 0; i < N; i++) {
      out[i] = 0.0f;
	 
    }
    a[2] = 2.3;
    a[3] = 100.0;
    a[5] = .7;
    b[100] = 2000.0;
    a[500] = 1000.0;
    b[3] = 2.3;
     

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(double) * N/2);
    cudaMalloc((void**)&d_b, sizeof(double) * N/2);
    cudaMalloc((void**)&d_out, sizeof(double) * N);
    cudaMalloc((void**)&wlet_blur_out, sizeof(double) * N/2);
    cudaMalloc((void**)&wlet_detail_out, sizeof(double) * N/2);    

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(double) * N/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(double) * N/2, cudaMemcpyHostToDevice);

    cudaMemcpy(d_out, out, sizeof(double)*N, cudaMemcpyHostToDevice);
    // Executing kernel 
    idft<<<2,N>>>(d_out, d_a, d_b);

    haar_detail<<<1,N/2>>>(wlet_detail_out, d_out);
    haar_blur<<<1,N/2>>>(wlet_blur_out, d_out);
    
    // Transfer data back to host memory
    //    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, wlet_blur_out, sizeof(double) * N/2, cudaMemcpyDeviceToHost);

    // Verification
    //    for(int i = 0; i < N; i++){
    //   assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    // }
    //    printf("out[0] = %f\n", out[0]);
    printf("BLUR\n");
    for (int j = 0; j<N/2; j++) {
      printf("ok %f\n", out[j]);
    }


    cudaMemcpy(out, wlet_detail_out, sizeof(double) * N/2, cudaMemcpyDeviceToHost);

    printf("DETAIL\n");
    for (int j = 0; j<N/2; j++) {
      printf("ok %f\n", out[j]);
    }
    

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
