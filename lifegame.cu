#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define W 20
#define H 20
#define N (W * H)

__global__ void vector_lifegame(char* out, char* in) {
  int x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int y0 = blockDim.y * blockIdx.y + threadIdx.y;
  int xp = (x0 + 1) % W;
  int xm = (x0 - 1 + W) % W;
  int yp = (y0 + 1) % H;
  int ym = (y0 - 1 + H) % W;
  int sum = 0;
  sum += in[xm + ym * W];
  sum += in[x0 + ym * W];
  sum += in[xp + ym * W];
  sum += in[xm + y0 * W];
  sum += in[xp + y0 * W];
  sum += in[xm + yp * W];
  sum += in[x0 + yp * W];
  sum += in[xp + yp * W];
  int isAlive = in[x0 + y0 * W];
  out[x0 + y0 * W] = ((isAlive && (sum == 2 || sum == 3)) || (!isAlive && sum == 3)) ? 1 : 0;
}

int main() {
  // Allocate memory
  char* in = (char*)malloc(sizeof(char) * N);
  char* out = (char*)malloc(sizeof(char) * N);

  // Initialize array
  for (int i = 0; i < N; i++){
    in[i] = 0;
  }
  in[2 + 2 * W] = 1;
  in[3 + 3 * W] = 1;
  in[1 + 4 * W] = 1;
  in[2 + 4 * W] = 1;
  in[3 + 4 * W] = 1;
  
  // Allocate device memory
  char* d_in;
  char* d_out;
  cudaMalloc((void**)&d_in, sizeof(char) * N);
  cudaMalloc((void**)&d_out, sizeof(char) * N);

  for (int k = 0; k < 100; k++) {
    // Transfer data from host to device memory
    cudaMemcpy(d_in, in, sizeof(char) * N, cudaMemcpyHostToDevice);

    // Executing kernel 
    //dim3 grid(W, H), block(1, 1); // NG
    dim3 grid(1, 1), block(W, H);
    vector_lifegame<<<grid, block>>>(d_out, d_in);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(char) * N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        printf("%d", out[i * W + j]);
      }
      printf("\n");
    }
    printf("\n");
    // swap data
    char* t = in;
    in = out;
    out = t;
  }

  // Deallocate device memory
  cudaFree(d_in);
  cudaFree(d_out);

  // Deallocate host memory
  free(in);
  free(out);
}
