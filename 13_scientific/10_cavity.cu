#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <time.h>

using namespace std;
using namespace cooperative_groups;

/**
 * copy src array to dst array.
 */
__device__ void copy(float *src, float *dst, const int idx) {
  dst[idx] = src[idx];
}

/**
 * initialize variables u, v, p, and b.
 */
__global__ void initVar(float *u, float *v, float *p, float *b,
                const int nx, const int ny) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < nx*ny){
    u[idx] = 0;
    v[idx] = 0;
    p[idx] = 0;
    b[idx] = 0;
  }
}

/**
 * compute b.
 */
__global__ void computeB(const float *u, const float *v, float *b,
                const int nx, const int ny, const double dx, const double dy,
                const double dt, const double rho) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const bool boundary = (idx < ny)
                    || (ny*(nx-1) <= idx)
                    || (idx % ny == 0)
                    || (idx % ny == ny-1);
  
  if(idx < nx*ny && !boundary){  //except for boundary
    b[idx] = rho * (1/dt * ((u[idx + ny] - u[idx - ny]) / (2*dx) + (v[idx + 1] - v[idx - 1]) / (2*dy))
                    - pow((u[idx + ny] - u[idx - ny]) / (2*dx), 2)
                    - 2 * ((u[idx + 1] - u[idx - 1]) / (2*dy) * (v[idx + ny] - v[idx - ny]) / (2*dx))
                    - pow((v[idx + 1] - v[idx - 1]) / (2*dy), 2)
                    );
  }
}

/**
 * compute p.
 */
__global__ void computeP(const float *u, const float *v, float *p, const float *b, float *pn,
                const int nx, const int ny, const int nit, const double dx, const double dy) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  grid_group grid = this_grid();

  const bool boundary = (idx < ny)
                    || (ny*(nx-1) <= idx)
                    || (idx % ny == 0)
                    || (idx % ny == ny-1);
  
  if(idx < nx*ny){
    for (int it=0; it<nit; it++) {
      //copy previous p to pn
      copy(p, pn, idx);
      grid.sync();

      //except for boundary
      if(!boundary){
        p[idx] = (pow(dy, 2) * (pn[idx + ny] + pn[idx - ny])
                + pow(dx, 2) * (pn[idx + 1] + pn[idx - 1])
                - b[idx] * pow(dx, 2) * pow(dy, 2))
                  / (2 * (pow(dx, 2) + pow(dy, 2)));
      }
      __syncthreads();

      //only for boundary
      if(boundary){
        if(idx < ny)
          p[idx] = p[idx + ny];
        if(ny*(nx-1) <= idx)
          p[idx] = p[idx - ny];
        if(idx % ny == 0)
          p[idx] = p[idx + 1];
        if(idx % ny == ny-1)
          p[idx] = p[idx - 1];
      }
      __syncthreads();
    }
  }
}

/**
 * compute u and v.
 */
__global__ void computeUV(float *u, float *v, const float *p, float *un, float *vn,
                const int nx, const int ny, const double dx, const double dy,
                const double dt, const double rho, const double nu) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  grid_group grid = this_grid();

  const bool boundary = (idx < ny)
                    || (ny*(nx-1) <= idx)
                    || (idx % ny == 0)
                    || (idx % ny == ny-1);
  
  if(idx < nx*ny){
    //copy previous u,v to un,vn
    copy(u, un, idx);
    copy(v, vn, idx);
    grid.sync();

    //except for boundary
    if(!boundary){
      u[idx] = un[idx] - un[idx] * dt/dx * (un[idx] - un[idx - ny])
                        - un[idx] * dt/dy * (un[idx] - un[idx - 1])
                        - dt / (2*rho*dx) * (p[idx + ny] - p[idx - ny])
                        + nu * dt / pow(dx, 2) * (un[idx + ny] - 2 * un[idx] + un[idx - ny])
                        + nu * dt / pow(dy, 2) * (un[idx + 1] - 2 * un[idx] + un[idx - 1]);
      v[idx] = vn[idx] - vn[idx] * dt/dx * (vn[idx] - vn[idx - ny])
                        - vn[idx] * dt/dy * (vn[idx] - vn[idx - 1])
                        - dt / (2*rho*dy) * (p[idx + 1] - p[idx - 1])
                        + nu * dt / pow(dx, 2) * (vn[idx + ny] - 2 * vn[idx] + vn[idx - ny])
                        + nu * dt / pow(dy, 2) * (vn[idx + 1] - 2 * vn[idx] + vn[idx - 1]);
    }

    //only for boundary
    if(boundary){
      u[idx] = 0;
      v[idx] = 0;

      if(idx % ny == ny-1)
        u[idx] = 1;
    }
  }
}



int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  const int ThreadsPerBlock = 512;
  const int Blocks = (nx*ny + ThreadsPerBlock - 1) / ThreadsPerBlock;

  float* u;
  float* v;
  float* p;
  float* b;
  float* un;
  float* vn;
  float* pn;
  cudaMallocManaged(&u, nx*ny*sizeof(float));
  cudaMallocManaged(&v, nx*ny*sizeof(float));
  cudaMallocManaged(&p, nx*ny*sizeof(float));
  cudaMallocManaged(&b, nx*ny*sizeof(float));
  cudaMallocManaged(&un, nx*ny*sizeof(float));
  cudaMallocManaged(&vn, nx*ny*sizeof(float));
  cudaMallocManaged(&pn, nx*ny*sizeof(float));
  
  initVar<<<Blocks, ThreadsPerBlock>>>(u, v, p, b, nx, ny);
  
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  struct timespec start, stop;
  double elapse = 0;
  clock_gettime(CLOCK_REALTIME, &start);

  for (int n=0; n<nt; n++) {
    void* argsComputeB[]{ &u, &v, &b, &nx, &ny, &dx, &dy, &dt, &rho };
    cudaLaunchCooperativeKernel((void*)computeB, Blocks, ThreadsPerBlock, argsComputeB, 0, nullptr);
    cudaDeviceSynchronize();

    void* argsComputeP[]{ &u, &v, &p, &b, &pn, &nx, &ny, &nit, &dx, &dy };
    cudaLaunchCooperativeKernel((void*)computeP, Blocks, ThreadsPerBlock, argsComputeP, 0, nullptr);
    cudaDeviceSynchronize();

    void* argsComputeUV[]{ &u, &v, &p, &un, &vn, &nx, &ny, &dx, &dy, &dt, &rho, &nu };
    cudaLaunchCooperativeKernel((void*)computeUV, Blocks, ThreadsPerBlock, argsComputeUV, 0, nullptr);
    cudaDeviceSynchronize();
    
    // write to dat file
    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j + ny*i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j + ny*i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j + ny*i] << " ";
      pfile << "\n";
    }
  }
  cudaDeviceSynchronize();

  clock_gettime(CLOCK_REALTIME, &stop);
  elapse += stop.tv_sec - start.tv_sec + (stop.tv_nsec - start.tv_nsec)*1e-9;
  printf("elapsed time = %lf\n", elapse);

  ufile.close();
  vfile.close();
  pfile.close();

  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(b);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);
}
