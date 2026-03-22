#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<sched.h>
#include<png.h>
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


typedef struct{
    int thread_id;
    int* image;
} thread_data;

int iters;
double left, right, lower, upper;
int width, height, num_threads;

void* counting(void* args){
    int id=((thread_data*)args)->thread_id;
    int* image=((thread_data*)args)->image;
    double x0, y0, x1, x, y, length_squared, temp1, temp2, temp;
    int repeats, repeats1, repeats2, j;
    bool flag1=0, flag2=0;
    for(int i=id;i<height;i+=num_threads){
        y0=i*((upper-lower)/height)+lower;
        for(j=0;j<width-1;j+=2){
            x0=j*((right-left)/width)+left;
            x1=(j+1)*((right-left)/width)+left;
            flag1=0;
            flag2=0;
            __m128d vec_y=_mm_set1_pd(0);
            __m128d vec_x=_mm_set1_pd(0);
            __m128d origin_y=_mm_set_pd(y0, y0);
            __m128d origin_x=_mm_set_pd(x0, x1);
            __m128d vec_temp=_mm_set1_pd(0);
            __m128d vec_2=_mm_set1_pd(2);
            __m128d vec_square=_mm_set1_pd(0);
            repeats1=0;
            repeats2=0;
            //length_squared=0;
            while (flag1==0||flag2==0) {
                // temp = x * x - y * y + x0;
                vec_temp=_mm_add_pd(_mm_sub_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y)), origin_x);
                //y = 2 * x * y + y0;
                vec_y=_mm_add_pd(_mm_mul_pd(_mm_mul_pd(vec_2, vec_x), vec_y), origin_y);
                //x = temp;
                vec_x=vec_temp;
                //length_squared = x * x + y * y;
                vec_square=_mm_add_pd(_mm_mul_pd(vec_x, vec_x), _mm_mul_pd(vec_y, vec_y));

                if(flag1==0){
                    _mm_storeh_pd(&temp1, vec_square);
                    ++repeats1;
                    if(temp1>=4||repeats1==iters)
                        flag1=1;
                }
                if(flag2==0){
                    _mm_storel_pd(&temp2, vec_square);
                    ++repeats2;
                    if(temp2>=4||repeats2==iters)
                        flag2=1;

                }
            }
            image[i * width + j] = repeats1;
            image[i * width + j + 1]=repeats2;
        }
        if(j==width-1){
            x0=j*((right-left)/width)+left;
            repeats = 0;
            x = 0;
            y = 0;
            length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[i * width + j] = repeats;
        }
    }
    return NULL;

}
int main(int argc, char** argv){
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    thread_data* args = (thread_data*)malloc(num_threads * sizeof(thread_data));
    for (int i = 0; i < num_threads; ++i) {
        args[i].thread_id = i;
        args[i].image = image;
        int rc = pthread_create(&threads[i], NULL, counting, (void*)&args[i]);
        assert(rc == 0);
    }
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    write_png(filename, iters, width, height, image);
    free(image);
    free(threads);
    free(args);
    return 0;
}