#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include <cstring>
#include <boost/sort/spreadsort/spreadsort.hpp>

void merge(float *data, float *tmp, int len, float *buf, int len_p, bool small) {
    if (small) {
        float *p1=data;
        float *p2=buf;
        float *p3=tmp;

        for (int j=0;j<len;j++){
            if(p2>=buf+len_p||(p1<data+len&& *p1< *p2))
                *p3++ =*p1++;
            else
                *p3++ =*p2++;
        }
    } else {

        float *p1=data+len-1;
        float *p2=buf+len_p-1;
        float *p3=tmp+len-1;

        for (int j=0;j<len;j++){
            if (p2<buf||(p1>=data&&*p1>*p2))
                *p3-- =*p1--;
            else
                *p3-- =*p2--;
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc!=4){
        if(rank==0){
            printf("must provide exactly 3 arguments!\n");
        }
        MPI_Finalize();
        return 1;
    }

    double tik,tok;
    int array_size=atoi(argv[1]);
    int process_size=array_size/size;
    int remain=array_size%size;
    int local_size=(rank<remain)?(process_size+1):process_size;
    const char *const input_filename = argv[2],
               *const output_filename = argv[3];

    MPI_File input_file, output_file;
    float *data = new float[process_size+1];

    //divide the data into "size" parts, and the first "remain" parts have "process_size"<--(array_size/size)+1 elements, while the others have "process_size"<--(array_size/size) elements.
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);

    if(rank<remain)
        MPI_File_read_at(input_file, sizeof(float) * rank*(process_size+1), data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    else{
        int cool=(process_size+1)*remain;
        MPI_File_read_at(input_file, sizeof(float) * (cool+(rank-remain)*process_size), data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&input_file);

    //sort part
    //tik=MPI_Wtime();
    boost::sort::spreadsort::spreadsort(data,data+local_size);//目前最快
    //std::sort(data, data+local_size);    
    //tok=MPI_Wtime();
    //printf("the sorting time of Rank%d is %f\n",rank, tok-tik);

    float *partner_data = new float[process_size + 1];
    float *merged_data = new float[process_size*2 + 1];
    int partner;
    for (int i= 0;i<=size;++i){
        if (i%2==0) {
            partner=(rank%2==0)?rank+1:rank-1;
        }
        else {
            partner=(rank%2!= 0)?rank+1:rank-1;
        }
        int partner_local_size=(partner<remain)?(process_size+1):process_size;

        if (partner<0||partner>=size){
            continue;
        }

        if(rank<partner){
            MPI_Sendrecv(data+local_size-1,1,MPI_FLOAT,partner,0,partner_data,1,MPI_FLOAT,partner,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            if(partner_data[0]>=data[local_size-1])
                continue;
        }
        else{
            MPI_Sendrecv(data,1,MPI_FLOAT,partner,0,partner_data,1,MPI_FLOAT,partner,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            if(partner_data[0]<=data[0])
                continue;
        }

        if (rank<partner) {
            MPI_Sendrecv( data, local_size, MPI_FLOAT, partner, 0, partner_data, partner_local_size, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //MPI_Send(data, local_size, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
            //MPI_Recv(partner_data, partner_local_size, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            merge(data, merged_data, local_size, partner_data, partner_local_size, 1);
            std::swap(data, merged_data);
            }
        else {
            MPI_Sendrecv( data , local_size , MPI_FLOAT , partner , 0 , partner_data , partner_local_size , MPI_FLOAT , partner , 0 , MPI_COMM_WORLD , MPI_STATUS_IGNORE);
            //MPI_Recv(partner_data, partner_local_size, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //MPI_Send(data, local_size, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);

            merge(data, merged_data, local_size, partner_data, partner_local_size,0);
            std::swap(data, merged_data);
        }
    }

    delete[] partner_data;

    //sort part
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if(rank<remain)
        MPI_File_write_at(output_file, sizeof(float) * rank*(process_size+1), data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    else{
        int cool=(process_size+1)*remain;
        MPI_File_write_at(output_file, sizeof(float) * (cool+(rank-remain)*process_size), data, local_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}

