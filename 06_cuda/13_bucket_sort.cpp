#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void count(int N, int* key, int* bucket){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid<N){
    atomicAdd(&bucket[key[tid]], 1);
  }
}

int main() {
  int n = 50;
  int range = 5;
  int* key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int* bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }

//  for (int i=0; i<n; i++) {
//    bucket[key[i]]++;
//  }
  int threads_per_block = 64;
  int blocks = (n + threads_per_block - 1) / threads_per_block;
  count<<<blocks ,threads_per_block>>>(n, key, bucket);
  cudaDeviceSynchronize();

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
