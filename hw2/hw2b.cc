#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<sched.h>
#include<png.h>
#include<mpi.h>
#include<omp.h>
#include<emmintrin.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


int main(int argc,char** argv){

    assert(argc==9);
    const char* filename=argv[1];
    int iters = strtol(argv[2],0,10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    
    MPI_Init(&argc, &argv);
    int rank, size;
    
    
    int* image = NULL; 
    int* local_image;  
    int local_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double t0 = MPI_Wtime();


    int process_size = height / size;
    int remain = height % size;
    local_size = (rank < remain) ? (process_size + 1) : process_size;
    
    if (local_size > 0) {
        local_image = (int*)malloc(width * local_size * sizeof(int));
        assert(local_image);
    } else {
        local_image = NULL; 
    }
    
    if (rank == 0) {
        image = (int*)malloc(width * height * sizeof(int));
        assert(image);
    }

    double t_local = MPI_Wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i=0;i<local_size; i++) {
        int global_i=i*size+rank; 
        double y0 = global_i * ((upper-lower) / height) + lower;

        int j;
        double temp, x0, x1;
        for (j=0;j<width-1;j+=2) {
            x0=j*((right-left)/width)+left;
            x1 = (j + 1) * ((right - left) / width) + left;
            int repeats1=0,repeats2 = 0;
            int flag1 = 0, flag2 = 0;
            double temp1, temp2;
            __m128d vec_y=_mm_set1_pd(0);
            __m128d vec_x=_mm_set1_pd(0);
            __m128d origin_y=_mm_set_pd(y0, y0);
            __m128d origin_x=_mm_set_pd(x0, x1);
            __m128d vec_temp=_mm_set1_pd(0);
            __m128d vec_2=_mm_set1_pd(2);
            __m128d vec_square = _mm_set1_pd(0);
            
            while (flag1 == 0 || flag2 == 0) {
                vec_temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), origin_x);
                vec_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(vec_2, vec_x), vec_y), origin_y);
                vec_x = vec_temp;
                vec_square = _mm_add_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y));

                if(flag1 == 0){
                    _mm_storeh_pd(&temp1, vec_square);
                    ++repeats1;
                    if(temp1 >= 4 || repeats1 == iters)
                        flag1 = 1;
                }

                if(flag2 == 0){
                    _mm_storel_pd(&temp2, vec_square);
                    ++repeats2;
                    if(temp2 >= 4 || repeats2 == iters)
                        flag2 = 1;
                }
            }
            local_image[i * width + j] = repeats1;
            local_image[i * width + j + 1] = repeats2;
        }

        if(j == width - 1){
            x0 = j * ((right-left) / width) + left;
            int repeats = 0;
            double x = 0, y = 0;
            double length_squared = 0;
            while (repeats<iters && length_squared<4) {
                temp=x*x-y*y+x0;
                y = 2*x*y+y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            local_image[i * width + j] = repeats;
        }
    }

    double t_end = MPI_Wtime();
    printf("Rank %d compute time = %f\n", rank, t_end - t_local);

    if(rank == 0){
        for (int i = 0; i < local_size; i++) {
            int global_i = i * size + rank; 
            if (global_i < height) {
                memcpy(image + (global_i * width), 
                       local_image + (i * width), 
                       width * sizeof(int));
            }
        }
        
        
        for(int r= 1;r<size;r++){ 
            int r_local_size = height / size;
            if (r < height % size) {
                r_local_size++;
            }
            
            if (r_local_size > 0) {
                int* temp_buffer = (int*)malloc(width * r_local_size * sizeof(int));
                assert(temp_buffer);
                
                MPI_Recv(temp_buffer, r_local_size * width, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                for (int i = 0; i < r_local_size; i++) {
                    int global_i = i * size + r; 
                    if (global_i < height) {
                        memcpy(image + (global_i * width), 
                               temp_buffer + (i * width), 
                               width * sizeof(int));
                    }
                }
                free(temp_buffer); 
            }
        }
        
        write_png(filename, iters, width, height, image);
        free(image);
        
    } else {
        if (local_size>0) {
            MPI_Send(local_image, local_size * width, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (local_image!=NULL) {
        free(local_image);
    }
    
    double t_final=MPI_Wtime();
    if (rank == 0) {
        printf("Total time (including communication) = %f sec\n", t_final - t0);
    }

    MPI_Finalize();
    return 0;
}