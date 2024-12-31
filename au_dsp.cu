
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

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

inline __device__ double sinc(double ix) {
  if (ix==0.0l) {
    return 1.0;
  }
  else {
    double theta = M_PI*ix;
    return sin(theta)/theta;
  }
}
inline __device__ double blackman(double ix, double m) {
  return 0.42l + (-0.5l)*cos(2.0l*M_PI*ix/m) + 0.08l*cos(4.0l*M_PI*ix/m);
}
inline double hanning(double ix, double m) {
  return 0.5l*(1.0l - cos(2.0l*M_PI*ix/m));
}
inline __device__ double sinc_interpol(double *sig_base, double cut_ix, double winsize, double kfq) {
  double ix0_lim = cut_ix - winsize/2.0;
  double ixf_lim = cut_ix + winsize/2.0;
  int ix0 = (int)(std::ceil(ix0_lim));
  int ixf = (int)(std::floor(ixf_lim));
  double suma = 0.0l;
  for (int i=ix0; i<=ixf; i++) {
    double sample = sig_base[ix0];
    double delta = cut_ix-(double)i;
    double sinc_val = sinc(kfq*delta);
    double win_val = blackman((double)i-ix0_lim, winsize);
    suma = fma(sample, sinc_val*win_val, suma);
  }
  return suma;
} 
__global__ void outside_fma_sinc(double *out_base, double *sig_base, double winsize, double kStep, double kfq) {
  int out_ix = threadIdx.x;
  double cut_ix = ((double)out_ix)*kStep;
  double suma = sinc_interpol(sig_base, cut_ix, winsize, kfq);
  out_base[out_ix] = suma;
}


__global__ void cpx_convo_ilv(double *out_base, double *sig0_base, double *sig1_base) {
  int out_idx = threadIdx.x*2;
  double re0 = sig0_base[out_idx];
  double im0 = sig0_base[out_idx+1];
  double re1 = sig1_base[out_idx];
  double im1 = sig1_base[out_idx+1];
  double ac  = __dmul_rn(re0, re1);
  double bd  = __dmul_rn(im0, im1);
  double ad  = __dmul_rn(re0, im1);
  double bc  = __dmul_rn(re1, im0);
  double re2 = __dsub_rn(ac,  bd);
  double im2 = __dadd_rn(ad,  bc);
  out_base[out_idx] = re2;
  out_base[out_idx+1] = im2;
}


__global__ void cpx_convo_dilv(double *out_base0, double* out_base1, double *sig0_base, double *sig1_base) {
  int in_idx = threadIdx.x*2;
  double re0 = sig0_base[in_idx];
  double im0 = sig0_base[in_idx+1];
  double re1 = sig1_base[in_idx];
  double im1 = sig1_base[in_idx+1];
  int out_idx = threadIdx.x;
  double ac  = __dmul_rn(re0, re1);
  double bd  = __dmul_rn(im0, im1);
  double ad  = __dmul_rn(re0, im1);
  double bc  = __dmul_rn(re1, im0);
  double re2 = __dsub_rn(ac,  bd);
  double im2 = __dadd_rn(ad,  bc);
  out_base0[out_idx] = re2;
  out_base1[out_idx] = im2;
}



__global__ void polar_eq_0_ilv(double *out_base, double *sig_base, double *eq_curve) {
  int in_idx = threadIdx.x;
  int out_idx = threadIdx.x*2;
  double mag0 = sig_base[in_idx];
  double kmag = eq_curve[in_idx];
  out_base[out_idx] = __dmul_rn(mag0, kmag);
}

__global__ void polar_eq_1_ilv(double *out_base, double *sig_base, double *eq_curve) {
  int in_idx = threadIdx.x;
  int out_idx = threadIdx.x*2;
  double phs0 = sig_base[in_idx];
  out_base[out_idx+1] = phs0;
}



__global__ void polar_eq_0_dilv(double *out_base, double *sig_base, double *eq_curve) {
  int in_idx = threadIdx.x;
  double mag0 = sig_base[in_idx];
  double kmag = eq_curve[in_idx];
  out_base[in_idx] = __dmul_rn(mag0, kmag);
}


__global__ void xsyn_ilv(double *out_base, double *sig0_mags, double *sig1_phses) {
  int in_idx  = threadIdx.x;
  int out_idx = threadIdx.x*2;
  double mag0 = sig0_mags[in_idx];
  double phs1 = sig1_phses[in_idx];
  out_base[out_idx] = mag0;
  out_base[out_idx+1] = phs1;
}

__global__ void xsyn_0_ilv(double *out_base, double *sig0_mags, double *sig1_phses) {
  int in_idx  = threadIdx.x;
  int out_idx = threadIdx.x*2;
  double mag0 = sig0_mags[in_idx];
  out_base[out_idx] = mag0;
}

__global__ void xsyn_1_ilv(double *out_base, double *sig0_mags, double *sig1_phses) {
  int in_idx  = threadIdx.x;
  int out_idx = threadIdx.x*2;
  double phs1 = sig1_phses[in_idx];
  out_base[out_idx+1] = phs1;
}

__global__ void xsyn_0_dilv(double *out_mags, double *sig0_mags, double *sig1_phses) {
  int in_idx  = threadIdx.x;
  int out_idx = threadIdx.x*2;
  double mag0 = sig0_mags[in_idx];
  out_mags[in_idx] = mag0;
}

__global__ void xsyn_1_dilv(double *out_phses, double *sig0_mags, double *sig1_phses) {
  int in_idx  = threadIdx.x;
  int out_idx = threadIdx.x*2;
  double phs1 = sig1_phses[in_idx];
  out_phses[in_idx] = phs1;
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


__global__ void polar_convo_0_dilv(double *out_base, double *sig_base, double *ir_base) {
  int out_idx = threadIdx.x;
  double mag0 = sig_base[out_idx];
  double mag1 = ir_base[out_idx];
  out_base[out_idx] = __dmul_rn(mag0,mag1);
}

__global__ void polar_convo_1_dilv(double *out_base, double *sig_base, double *ir_base) {
  int out_idx = threadIdx.x;
  double phs0 = sig_base[out_idx];
  double phs1 = ir_base[out_idx];
  double suma =__dadd_rn(phs0, phs1);
  if (suma>2.0l*M_PI) {
    suma = __dsub_rn(suma, 2.0l*M_PI);
  }
  else if (suma < 0) {
    suma = __dadd_rn(suma, 2.0l*M_PI);
  }
  out_base[out_idx] = suma;
}



__global__ void polar_convo_0_ilv(double *out_base, double *sig_base, double *ir_base) {
  int out_idx = threadIdx.x;
  double mag0 = sig_base[out_idx];
  double mag1 = ir_base[out_idx];
  out_base[2*out_idx] = __dmul_rn(mag0,mag1);
}

__global__ void polar_convo_1_ilv(double *out_base, double *sig_base, double *ir_base) {
  int out_idx = threadIdx.x;
  double phs0 = sig_base[out_idx];
  double phs1 = ir_base[out_idx];
  double suma =__dadd_rn(phs0, phs1);
  if (suma>2.0l*M_PI) {
    suma = __dsub_rn(suma, 2.0l*M_PI);
  }
  else if (suma < 0) {
    suma = __dadd_rn(suma, 2.0l*M_PI);
  }
  out_base[2*out_idx+1] = suma;
}




__global__ void idft(double *out, double *a, double *b) {
  //  for (int i=0; i<n; i ++){
      float sum = 0.0;
      //      for (int j=0; j<N/2; j ++){
      for (int j=0; j<N; j ++){
	//        sum += (a[j]*sin(2*3.141597*threadIdx.x*j/N)/(N/2) - b[j]*cos(2*3.14159*threadIdx.x*j/N)/(N/2));
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
static double BUF_0[262144];
int len0 = 0;
static double BUF_1[262144];
int len1 = 0;
static double BUF_2[262144];
int len2 = 0;
static double BUF_3[262144];
int len3 = 0;
extern "C" {
  void set_buf0(int ix, double nu_val) {
    BUF_0[ix] = nu_val;
  }
  double buf0_at(int ix) {
    return BUF_0[ix];
  }
  void set_buf1(int ix, double nu_val) {
    BUF_1[ix] = nu_val;
  }
  double buf1_at(int ix) {
    return BUF_1[ix];
  }
  void set_buf2(int ix, double nu_val) {
    BUF_2[ix] = nu_val;
  }
  double buf2_at(int ix) {
    return BUF_2[ix];
  }
  void set_buf3(int ix, double nu_val) {
    BUF_3[ix] = nu_val;
  }
  double buf3_at(int ix) {
    return BUF_3[ix];
  }
  
  void set_buflen(int buf_ix, int nu_len) {
    if (buf_ix == 0) {
      len0 = nu_len;
    }
    else if (buf_ix == 1) {
      len1 = nu_len;
    }
    else if (buf_ix == 2) {
      len2 = nu_len;
    }
    else if (buf_ix == 3) {
      len3 = nu_len;
    }
  }
  void erase_buf(int buf_ix) {
    if (buf_ix == 0) {
      for (int i=0; i<len0; i++) {
        BUF_0[i] = 0.0l;
      }
    }
    else if (buf_ix == 1) {
      for (int i=0; i<len1; i++) {
        BUF_1[i] = 0.0l;
      }
    }
    else if (buf_ix == 2) {
      for (int i=0; i<len3; i++) {
        BUF_2[i] = 0.0l;
      }
    }
    else if (buf_ix == 3) {
      for (int i=0; i<len3; i++) {
        BUF_3[i] = 0.0l;
      }
    }
  }
}
int main(){
    double *a, *b, *out;
    double *d_a, *d_b, *d_out, *wlet_blur_out, *wlet_detail_out; 

    a   = (double*)malloc(sizeof(double) * N/2);
    b   = (double*)malloc(sizeof(double) * N/2);
    out = (double*)malloc(sizeof(double) * N);
    //    wletout = (float*)malloc(sizeof(float) * N);

   
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
     

    cudaMalloc((void**)&d_a, sizeof(double) * N/2);
    cudaMalloc((void**)&d_b, sizeof(double) * N/2);
    cudaMalloc((void**)&d_out, sizeof(double) * N);
    cudaMalloc((void**)&wlet_blur_out, sizeof(double) * N/2);
    cudaMalloc((void**)&wlet_detail_out, sizeof(double) * N/2);    

    cudaMemcpy(d_a, a, sizeof(double) * N/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(double) * N/2, cudaMemcpyHostToDevice);

    cudaMemcpy(d_out, out, sizeof(double)*N, cudaMemcpyHostToDevice);
    // Executing kernel 
    idft<<<2,N>>>(d_out, d_a, d_b);

    haar_detail<<<1,N/2>>>(wlet_detail_out, d_out);
    haar_blur<<<1,N/2>>>(wlet_blur_out, d_out);
    
    //    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, wlet_blur_out, sizeof(double) * N/2, cudaMemcpyDeviceToHost);

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
    

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    free(a); 
    free(b); 
    free(out);
}
