#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BR 64
#define BC 64
#define MAX_D 64 
#define PAD 4

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

void input(char *input_filename);
void output(char *output_filename);


__global__ void 
__launch_bounds__(BR) 
flash_attention_kernel_occupancy(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ O,
    int N, int d, float scale
) {

    __shared__ float s_K[BC][MAX_D + PAD];
    __shared__ float s_V[BC][MAX_D + PAD];

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int tx = threadIdx.x; 

    int row_idx = by * BR + tx;
    

    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float o_i[MAX_D]; 

    for(int x = 0; x < d; x++) o_i[x] = 0.0f;

    int batch_offset = bx * N * d;
    
    const float4* q_batch_vec = (const float4*)(Q + batch_offset);
    const float4* k_batch_vec = (const float4*)(K + batch_offset);
    const float4* v_batch_vec = (const float4*)(V + batch_offset);


    float reg_q[MAX_D];
    int d_vec = d / 4; 

    if (row_idx < N) {
        for(int x = 0; x < d_vec; x++) {
            float4 loaded_vec = q_batch_vec[row_idx * d_vec + x];
            reg_q[x * 4 + 0] = loaded_vec.x;
            reg_q[x * 4 + 1] = loaded_vec.y;
            reg_q[x * 4 + 2] = loaded_vec.z;
            reg_q[x * 4 + 3] = loaded_vec.w;
        }
    } 


    int tc = (N + BC - 1) / BC;

    for (int j = 0; j < tc; j++) {
        
        int tile_row = tx;
        int global_k_row = j * BC + tile_row;

        float4* s_K_vec_ptr = (float4*)(&s_K[tile_row][0]);
        float4* s_V_vec_ptr = (float4*)(&s_V[tile_row][0]);

        if (tile_row < BC && global_k_row < N) {
            for (int x = 0; x < d_vec; x++) {
                s_K_vec_ptr[x] = k_batch_vec[global_k_row * d_vec + x];
                s_V_vec_ptr[x] = v_batch_vec[global_k_row * d_vec + x];
            }
        } else if (tile_row < BC) {

            float4 zero_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int x = 0; x < d_vec; x++) {
                s_K_vec_ptr[x] = zero_val;
                s_V_vec_ptr[x] = zero_val;
            }
        }
        
        __syncthreads(); 

        if (row_idx < N) {
            float scores[BC];
            float row_max = -FLT_MAX; 

            for (int k_idx = 0; k_idx < BC; k_idx++) {
                float score = 0.0f;
                
                if (j * BC + k_idx < N) {
                    for (int x = 0; x < d; x++) {
                        score += reg_q[x] * s_K[k_idx][x];
                    }
                    score *= scale;
                } else {
                    score = -FLT_MAX;
                }
                
                scores[k_idx] = score;
                row_max = fmaxf(row_max, score);
            }

            float row_sum = 0.0f;
            for (int k_idx = 0; k_idx < BC; k_idx++) {
                scores[k_idx] = expf(scores[k_idx] - row_max);
                row_sum += scores[k_idx];
            }

            float mi_new = fmaxf(m_i, row_max);
            float alpha = expf(m_i - mi_new);       
            float beta = expf(row_max - mi_new);    

            float li_new = alpha * l_i + beta * row_sum;

            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int k_idx = 0; k_idx < BC; k_idx++) {
                     pv += scores[k_idx] * s_V[k_idx][x];
                }
                o_i[x] = (alpha * l_i * o_i[x] + beta * pv) / li_new;
            }

            m_i = mi_new;
            l_i = li_new;
        }
        __syncthreads();
    }

    if (row_idx < N) {
        float4* o_batch_vec = (float4*)(O + batch_offset);
        for (int x = 0; x < d_vec; x++) {
            float4 res;
            res.x = o_i[x * 4 + 0];
            res.y = o_i[x * 4 + 1];
            res.z = o_i[x * 4 + 2];
            res.w = o_i[x * 4 + 3];
            o_batch_vec[row_idx * d_vec + x] = res;
        }
    }
}

int B, N, d;
float *Q, *K, *V, *O;

double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    float *d_Q, *d_K, *d_V, *d_O;
    size_t size = B * N * d * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_Q, size));
    CUDA_CHECK(cudaMalloc((void**)&d_K, size));
    CUDA_CHECK(cudaMalloc((void**)&d_V, size));
    CUDA_CHECK(cudaMalloc((void**)&d_O, size));

    CUDA_CHECK(cudaMemcpy(d_Q, Q, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, K, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, V, size, cudaMemcpyHostToDevice));

    double start, end;
    start = getTimeStamp();

    int tr = (N + BR - 1) / BR;
    dim3 grid(B, tr);
    dim3 block(BR); 

    float scale = 1.0f / sqrtf((float)d);
    
    flash_attention_kernel_occupancy<<<grid, block>>>(d_Q, d_K, d_V, d_O, N, d, scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    CUDA_CHECK(cudaMemcpy(O, d_O, size, cudaMemcpyDeviceToHost));

    output(argv[2]);

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");
    if(!file) { printf("Failed to open input file\n"); exit(1); }
    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");
    if(!file) { printf("Failed to open output file\n"); exit(1); }
    fwrite(O, sizeof(float), B * N * d, file);
    free(Q); free(K); free(V); free(O);
    fclose(file);
}