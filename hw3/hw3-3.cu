#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>

const int INF = 1073741823;
const int B = 64; 

int n, m, n_origin; 
int *Dist; 
int *Dist_GPU[2]; 

void input(char *inFileName);
void output(char *outFileName);
void block_FW_MultiGPU(int B);

__global__ void FW_Phase1(int n, int Round, int *Dist_GPU) {
    __shared__ int s_pivot[B][B]; 
    int tx = threadIdx.x; int ty = threadIdx.y;
    int pivot_idx = Round * B;

    int a = tx; int b = ty;
    int c = tx + 32; int d = ty + 32;
    
    s_pivot[b][a] = Dist_GPU[(pivot_idx + b) * n + (pivot_idx + a)];
    s_pivot[b][c] = Dist_GPU[(pivot_idx + b) * n + (pivot_idx + c)];
    s_pivot[d][a] = Dist_GPU[(pivot_idx + d) * n + (pivot_idx + a)];
    s_pivot[d][c] = Dist_GPU[(pivot_idx + d) * n + (pivot_idx + c)];
    
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        int dik1 = s_pivot[b][k]; int dkj1 = s_pivot[k][a];
        int dik2 = s_pivot[b][k]; int dkj2 = s_pivot[k][c];
        int dik3 = s_pivot[d][k]; int dkj3 = s_pivot[k][a];
        int dik4 = s_pivot[d][k]; int dkj4 = s_pivot[k][c];

        s_pivot[b][a] = min(s_pivot[b][a], dik1 + dkj1);
        s_pivot[b][c] = min(s_pivot[b][c], dik2 + dkj2);
        s_pivot[d][a] = min(s_pivot[d][a], dik3 + dkj3);
        s_pivot[d][c] = min(s_pivot[d][c], dik4 + dkj4);
        __syncthreads();
    }

    Dist_GPU[(pivot_idx + b) * n + (pivot_idx + a)] = s_pivot[b][a];
    Dist_GPU[(pivot_idx + b) * n + (pivot_idx + c)] = s_pivot[b][c];
    Dist_GPU[(pivot_idx + d) * n + (pivot_idx + a)] = s_pivot[d][a];
    Dist_GPU[(pivot_idx + d) * n + (pivot_idx + c)] = s_pivot[d][c];
}

__global__ void FW_Phase2_Row(int n, int Round, int *Dist_GPU) {
    __shared__ int s_pivot[B][B]; 
    __shared__ int s_self[B][B];

    int bx = blockIdx.x; 
    if (bx == Round) return; 

    int tx = threadIdx.x; int ty = threadIdx.y;
    int pivot_idx = Round * B;
    
    int self_x = bx * B; 
    int self_y = pivot_idx;

    int a = tx; int b = ty;
    int c = tx + 32; int d = ty + 32;

    s_self[b][a] = Dist_GPU[(self_y + b) * n + (self_x + a)];
    s_self[b][c] = Dist_GPU[(self_y + b) * n + (self_x + c)];
    s_self[d][a] = Dist_GPU[(self_y + d) * n + (self_x + a)];
    s_self[d][c] = Dist_GPU[(self_y + d) * n + (self_x + c)];

    s_pivot[b][a] = Dist_GPU[(pivot_idx + b) * n + (pivot_idx + a)];
    s_pivot[b][c] = Dist_GPU[(pivot_idx + b) * n + (pivot_idx + c)];
    s_pivot[d][a] = Dist_GPU[(pivot_idx + d) * n + (pivot_idx + a)];
    s_pivot[d][c] = Dist_GPU[(pivot_idx + d) * n + (pivot_idx + c)];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        int dik1 = s_pivot[b][k]; int dkj1 = s_self[k][a];
        int dik2 = s_pivot[b][k]; int dkj2 = s_self[k][c];
        int dik3 = s_pivot[d][k]; int dkj3 = s_self[k][a];
        int dik4 = s_pivot[d][k]; int dkj4 = s_self[k][c];

        s_self[b][a] = min(s_self[b][a], dik1 + dkj1);
        s_self[b][c] = min(s_self[b][c], dik2 + dkj2);
        s_self[d][a] = min(s_self[d][a], dik3 + dkj3);
        s_self[d][c] = min(s_self[d][c], dik4 + dkj4);
        __syncthreads();
    }
    
    Dist_GPU[(self_y + b) * n + (self_x + a)] = s_self[b][a];
    Dist_GPU[(self_y + b) * n + (self_x + c)] = s_self[b][c];
    Dist_GPU[(self_y + d) * n + (self_x + a)] = s_self[d][a];
    Dist_GPU[(self_y + d) * n + (self_x + c)] = s_self[d][c];
}

__global__ void FW_Phase2_Col(int n, int Round, int *Dist_GPU, int offsetY) {
    __shared__ int s_pivot[B][B]; 
    __shared__ int s_self[B][B];

    int bx = blockIdx.y; 
    int global_by = bx + offsetY; 
    
    if (global_by == Round) return;

    int tx = threadIdx.x; int ty = threadIdx.y;
    int pivot_idx = Round * B;

    int self_x = pivot_idx;
    int self_y = global_by * B;

    int a = tx; int b = ty;
    int c = tx + 32; int d = ty + 32;

    s_self[b][a] = Dist_GPU[(self_y + b) * n + (self_x + a)];
    s_self[b][c] = Dist_GPU[(self_y + b) * n + (self_x + c)];
    s_self[d][a] = Dist_GPU[(self_y + d) * n + (self_x + a)];
    s_self[d][c] = Dist_GPU[(self_y + d) * n + (self_x + c)];

    s_pivot[b][a] = Dist_GPU[(pivot_idx + b) * n + (pivot_idx + a)];
    s_pivot[b][c] = Dist_GPU[(pivot_idx + b) * n + (pivot_idx + c)];
    s_pivot[d][a] = Dist_GPU[(pivot_idx + d) * n + (pivot_idx + a)];
    s_pivot[d][c] = Dist_GPU[(pivot_idx + d) * n + (pivot_idx + c)];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < B; ++k) {
        int dik1 = s_self[b][k]; int dkj1 = s_pivot[k][a];
        int dik2 = s_self[b][k]; int dkj2 = s_pivot[k][c];
        int dik3 = s_self[d][k]; int dkj3 = s_pivot[k][a];
        int dik4 = s_self[d][k]; int dkj4 = s_pivot[k][c];

        s_self[b][a] = min(s_self[b][a], dik1 + dkj1);
        s_self[b][c] = min(s_self[b][c], dik2 + dkj2);
        s_self[d][a] = min(s_self[d][a], dik3 + dkj3);
        s_self[d][c] = min(s_self[d][c], dik4 + dkj4);
        __syncthreads();
    }
    
    Dist_GPU[(self_y + b) * n + (self_x + a)] = s_self[b][a];
    Dist_GPU[(self_y + b) * n + (self_x + c)] = s_self[b][c];
    Dist_GPU[(self_y + d) * n + (self_x + a)] = s_self[d][a];
    Dist_GPU[(self_y + d) * n + (self_x + c)] = s_self[d][c];
}

__global__ void __launch_bounds__(256) FW_Phase3_MultiGPU(int n, int Round, int *Dist_GPU, int offsetY) {
    __shared__ int s_i_k[B][B]; 
    __shared__ int s_k_j[B][B]; 

    int reg_results[4][4]; 

    int bx = blockIdx.x; 
    int local_by = blockIdx.y; 
    int by = local_by + offsetY; 

    if (bx == Round || by == Round) return;

    int pivot_idx = Round * B;
    int block_x = bx * B;
    int block_y = by * B;

    int dep_i_k_x = pivot_idx; int dep_i_k_y = block_y; 
    int dep_k_j_x = block_x;   int dep_k_j_y = pivot_idx; 

    int tx = threadIdx.x; int ty = threadIdx.y;
    
    #pragma unroll
    for(int i=0; i<4; ++i) {
        #pragma unroll
        for(int j=0; j<4; ++j) {
            reg_results[i][j] = Dist_GPU[(block_y + ty*4 + i) * n + (block_x + tx*4 + j)];
        }
    }

    int tid = ty * 16 + tx; 
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
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
        b_vals[0] = s_k_j[k][tx * 4 + 0]; b_vals[1] = s_k_j[k][tx * 4 + 1];
        b_vals[2] = s_k_j[k][tx * 4 + 2]; b_vals[3] = s_k_j[k][tx * 4 + 3];
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
    block_FW_MultiGPU(64); 
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

void block_FW_MultiGPU(int B) {
    int round = n / B;
    size_t size = (size_t)n * n * sizeof(int);

    #pragma omp parallel num_threads(2)
    {
        int gpu_id = omp_get_thread_num();
        cudaSetDevice(gpu_id);
        
        if (gpu_id == 0) cudaDeviceEnablePeerAccess(1, 0);
        else cudaDeviceEnablePeerAccess(0, 0);

        cudaMalloc(&Dist_GPU[gpu_id], size);
        cudaMemcpy(Dist_GPU[gpu_id], Dist, size, cudaMemcpyHostToDevice);

        int num_gpus = 2;
        int blocks_per_gpu = round / num_gpus; 
        int start_round = gpu_id * blocks_per_gpu;
        int end_round = (gpu_id + 1) * blocks_per_gpu;
        if (gpu_id == num_gpus - 1) end_round = round;
        
        int my_height_blocks = end_round - start_round;

        dim3 threads_std(32, 32);
        dim3 threads_opt(16, 16);

        for (int r = 0; r < round; ++r) {
            int owner_id = (r < blocks_per_gpu) ? 0 : 1;
            
            if (gpu_id == owner_id) {
                FW_Phase1<<<1, threads_std>>>(n, r, Dist_GPU[gpu_id]);
                FW_Phase2_Row<<<round, threads_std>>>(n, r, Dist_GPU[gpu_id]);
            }
            
            #pragma omp barrier 

            if (gpu_id == owner_id) {
                size_t offset = (size_t)r * B * n * sizeof(int);
                size_t copy_size = (size_t)B * n * sizeof(int);
                
                cudaMemcpyPeer(
                    (char*)Dist_GPU[1 - gpu_id] + offset, 
                    1 - gpu_id, 
                    (char*)Dist_GPU[gpu_id] + offset, 
                    gpu_id, 
                    copy_size 
                );
            }
            
            #pragma omp barrier 

            FW_Phase2_Col<<<dim3(1, my_height_blocks), threads_std>>>(n, r, Dist_GPU[gpu_id], start_round);
            
            dim3 grid3(round, my_height_blocks);
            FW_Phase3_MultiGPU<<<grid3, threads_opt>>>(n, r, Dist_GPU[gpu_id], start_round);
        }
        
        int *host_ptr_offset = Dist + start_round * B * n;
        int *dev_ptr_offset = Dist_GPU[gpu_id] + start_round * B * n;
        size_t copy_back_size = (size_t)my_height_blocks * B * n * sizeof(int);
        
        cudaMemcpy(host_ptr_offset, dev_ptr_offset, copy_back_size, cudaMemcpyDeviceToHost);
        
        cudaFree(Dist_GPU[gpu_id]);
    }
}