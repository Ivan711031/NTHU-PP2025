#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <algorithm>

const int INF=1073741823;
const int B = 64; 

int n, m, n_origin; 
int *Dist; 
int *Dist_GPU; 

void input(char *inFileName);
void output(char *outFileName);
void block_FW(int B);

__global__ void FW_Phase1_2(int n, int Round, int *Dist_GPU, int phase) {
    __shared__ int s_pivot[B][B]; 
    __shared__ int s_self[B][B];

    int bx = blockIdx.x; 
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int pivot_idx = Round * B;
    
    int self_x, self_y;
    int pivot_x=pivot_idx;
    int pivot_y=pivot_idx;

    if(phase == 1){ 
        self_x=pivot_idx; 
        self_y=pivot_idx;
    } else{ 
        if (blockIdx.y==0){ 
            self_x=bx * B; 
            self_y= pivot_idx;
        } else{ 
            self_x=pivot_idx;
            self_y=bx * B;
        }
    }

    int a = tx;     
    int b = ty;
    int c = tx + 32; 
    int d = ty + 32;
    
    s_self[b][a]=Dist_GPU[(self_y + b)*n +(self_x + a)];
    s_self[b][c]=Dist_GPU[(self_y + b)*n +(self_x + c)];
    s_self[d][a]=Dist_GPU[(self_y + d)*n +(self_x + a)];
    s_self[d][c]=Dist_GPU[(self_y + d)*n +(self_x + c)];

    s_pivot[b][a]=Dist_GPU[(pivot_y + b)*n+(pivot_x + a)];
    s_pivot[b][c]=Dist_GPU[(pivot_y + b)*n+(pivot_x + c)];
    s_pivot[d][a]=Dist_GPU[(pivot_y + d)*n+(pivot_x + a)];
    s_pivot[d][c]=Dist_GPU[(pivot_y + d)*n+(pivot_x + c)];

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        int d_ik, d_kj;
        
        if (phase == 1) {
            d_ik = s_self[b][k]; d_kj = s_self[k][a];
        } else if (blockIdx.y == 0) { 
            d_ik = s_pivot[b][k]; d_kj = s_self[k][a];
        } else { 
            d_ik = s_self[b][k]; d_kj = s_pivot[k][a];
        }
        s_self[b][a] = min(s_self[b][a], d_ik + d_kj);

        if (phase == 1) {
            d_ik = s_self[b][k]; d_kj = s_self[k][c];
        } else if (blockIdx.y == 0) {
            d_ik = s_pivot[b][k]; d_kj = s_self[k][c];
        } else {
            d_ik = s_self[b][k]; d_kj = s_pivot[k][c];
        }
        s_self[b][c] = min(s_self[b][c], d_ik + d_kj);

        if (phase == 1) {
            d_ik = s_self[d][k]; d_kj = s_self[k][a];
        } else if (blockIdx.y == 0) {
            d_ik = s_pivot[d][k]; d_kj = s_self[k][a];
        } else {
            d_ik = s_self[d][k]; d_kj = s_pivot[k][a];
        }
        s_self[d][a] = min(s_self[d][a], d_ik + d_kj);
        
        if (phase == 1) {
            d_ik = s_self[d][k]; d_kj = s_self[k][c];
        } else if (blockIdx.y == 0) {
            d_ik = s_pivot[d][k]; d_kj = s_self[k][c];
        } else {
            d_ik = s_self[d][k]; d_kj = s_pivot[k][c];
        }
        s_self[d][c] = min(s_self[d][c], d_ik + d_kj);

        __syncthreads(); 
    }

    Dist_GPU[(self_y + b) * n + (self_x + a)] = s_self[b][a];
    Dist_GPU[(self_y + b) * n + (self_x + c)] = s_self[b][c];
    Dist_GPU[(self_y + d) * n + (self_x + a)] = s_self[d][a];
    Dist_GPU[(self_y + d) * n + (self_x + c)] = s_self[d][c];
}

__global__ void FW_Phase3(int n, int Round, int *Dist_GPU) {
    __shared__ int s_i_k[B][B]; 
    __shared__ int s_k_j[B][B]; 

    int reg_results[4][4]; 

    int bx=blockIdx.x;
    int by = blockIdx.y;
    int pivot_idx=Round * B;

    int block_x=bx*B;
    int block_y = by * B;

    int dep_i_k_x = pivot_idx;
    int dep_i_k_y = block_y;

    int dep_k_j_x = block_x;
    int dep_k_j_y = pivot_idx;

    int tx = threadIdx.x; int ty = threadIdx.y;
    
    #pragma unroll
    for(int i=0; i<4; ++i) {
        #pragma unroll
        for(int j=0; j<4; ++j) {
            reg_results[i][j] = Dist_GPU[(block_y + ty*4 + i) * n + (block_x + tx*4 + j)];
        }
    }

    int tid=ty * 16 + tx; 
    #pragma unroll
    for (int k=0;k<16;++k){
        int linear_idx = k * 256 + tid;
        int r = linear_idx >> 6; 
        int c = linear_idx & 63;
        
        s_i_k[r][c] = Dist_GPU[(dep_i_k_y + r) * n + (dep_i_k_x + c)];
        s_k_j[r][c] = Dist_GPU[(dep_k_j_y + r) * n + (dep_k_j_x + c)];
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        int b_vals[4]; 
        b_vals[0] = s_k_j[k][tx * 4 + 0];
        b_vals[1] = s_k_j[k][tx * 4 + 1];
        b_vals[2] = s_k_j[k][tx * 4 + 2];
        b_vals[3] = s_k_j[k][tx * 4 + 3];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int a_val = s_i_k[ty * 4 + i][k];
            reg_results[i][0] = min(reg_results[i][0], a_val + b_vals[0]);
            reg_results[i][1] = min(reg_results[i][1], a_val + b_vals[1]);
            reg_results[i][2] = min(reg_results[i][2], a_val + b_vals[2]);
            reg_results[i][3] = min(reg_results[i][3], a_val + b_vals[3]);
        }
    }

    #pragma unroll
    for(int i=0; i<4; ++i) {
        #pragma unroll
        for(int j=0; j<4; ++j) {
            Dist_GPU[(block_y + ty*4 + i) * n + (block_x + tx*4 + j)] = reg_results[i][j];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) return 1;
    input(argv[1]);
    block_FW(B); 
    output(argv[2]);
    return 0;
}

void input(char *inFileName) {
    FILE *file = fopen(inFileName, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    n_origin = n;
    if (n % B != 0) n = (n / B + 1) * B;
    cudaMallocHost(&Dist, n * n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j && i < n_origin) Dist[i * n + j] = 0;
            else Dist[i * n + j] = INF;
        }
    }
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * n + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName) {
    FILE *file = fopen(outFileName, "w");
    for (int i = 0; i < n_origin; ++i) {
        fwrite(&Dist[i * n], sizeof(int), n_origin, file);
    }
    fclose(file);
    cudaFreeHost(Dist);
}

void block_FW(int B) {
    size_t size = (size_t)n*n*sizeof(int);
    cudaMalloc(&Dist_GPU, size);
    cudaMemcpy(Dist_GPU, Dist, size, cudaMemcpyHostToDevice);
    int round = n / B;

    dim3 threads_standard(32, 32);
    dim3 threads_opt(16, 16);

    for (int r = 0; r < round; ++r) {
        FW_Phase1_2<<<1, threads_standard>>>(n, r, Dist_GPU, 1);
        
        dim3 grid2(round, 2);
        FW_Phase1_2<<<grid2, threads_standard>>>(n, r, Dist_GPU, 2);
        
        dim3 grid3(round, round);
        FW_Phase3<<<grid3, threads_opt>>>(n, r, Dist_GPU);
    }

    cudaMemcpy(Dist, Dist_GPU, size, cudaMemcpyDeviceToHost);
    cudaFree(Dist_GPU);
}