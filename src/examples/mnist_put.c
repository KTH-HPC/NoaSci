#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <byteswap.h>

#include "noa.h"

#define CHUNK_SIZE 100

int dataset_write(const char* container_name, int n, int features, int classes, float* data, float* labels) {
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  size_t hostname_len = 256;
  char hostname[hostname_len];
  assert( gethostname(hostname, hostname_len) == 0);

  char mero_filename[265];
  snprintf(mero_filename, 265, "./%s", hostname);
  noa_init(mero_filename, 4096, world_rank, 0);
printf("connected!\n");

  const char* metadata_path = "./metadata";
  const char* data_path = "./data";
  //const char* container_name = "testcontainer";
  container* bucket;

  int rc;
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);

      printf("metadata_path:   %s\n", bucket->name);
      printf("key_value_store: %s\n", bucket->key_value_store);
      printf("object_store:    %s\n", bucket->object_store);
  

  // long dims[]       = { 2, 4, 8 };
  // long chunk_dims[] = { 2, 2, 4 }; // 1x2x2
  long dims_data[] = { n, features };
  long dims_labels[] = { n, classes };
  long chunk_dims_data[] = { CHUNK_SIZE, features };
  long chunk_dims_labels[] = { CHUNK_SIZE, classes };

  NoaMetadata* metadata_data =
      noa_create_metadata(bucket, "data", FLOAT, BINARY, MERO, //POSIX,
                          sizeof(dims_data) / sizeof(long), dims_data, chunk_dims_data);
  NoaMetadata* metadata_labels =
      noa_create_metadata(bucket, "labels", FLOAT, BINARY, MERO, //POSIX,
                          sizeof(dims_labels) / sizeof(long), dims_labels, chunk_dims_labels);

  printf("dimensionality: %ld\n", metadata_data->n_dims);
  printf("dims: %ld %ld\n", metadata_data->dims[0], metadata_data->dims[1]);
  printf("num_chunks: %d\n", metadata_data->num_chunks);
  printf("dims: %ld %ld\n", metadata_data->dims[0], metadata_data->dims[1]);
  printf("n_chunk_dims: %ld\n", metadata_data->n_chunk_dims);
  printf("chunk_dims: %ld %ld\n", metadata_data->chunk_dims[0],
         metadata_data->chunk_dims[1]);

  const char* header = NULL; //"testHeader";

  int chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
  printf("number of chunks: %d\n", chunks);
  for (int chunk = 0; chunk < chunks; chunk++) {
    printf("Putting chunk: %d\n", chunk);
    rc = noa_put_chunk_by_id(bucket, metadata_data, chunk, data + chunk*CHUNK_SIZE*features, 0, header); assert(rc == 0);
    rc = noa_put_chunk_by_id(bucket, metadata_labels, chunk, labels + chunk*CHUNK_SIZE*classes, 0, header); assert(rc == 0);
    printf("data: %f\n", data);
  }
  rc = noa_put_metadata(bucket, metadata_data); assert(rc == 0);
  rc = noa_put_metadata(bucket, metadata_labels); assert(rc == 0);
  rc = noa_free_metadata(bucket, metadata_data);
  rc = noa_free_metadata(bucket, metadata_labels);
  rc = noa_container_close(bucket);
  MPI_Barrier(MPI_COMM_WORLD);

  // delete
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  metadata_data = noa_get_metadata(bucket, "data");
  metadata_labels = noa_get_metadata(bucket, "labels");
printf("datatype: %d\n", metadata_data->datatype);
printf("id: %s\n", metadata_data->id);
  float *verify_data = malloc(sizeof(float) * CHUNK_SIZE * features);
  float *verify_labels = malloc(sizeof(float) * CHUNK_SIZE * features);
  char *verify_header = NULL;

  for (int chunk = 0; chunk < chunks; chunk++) {
    rc = noa_get_chunk(bucket, metadata_data, (void**)&verify_data, &verify_header, chunk);
    rc = noa_get_chunk(bucket, metadata_labels, (void**)&verify_labels, &verify_header, chunk);
  }

  rc = noa_delete(bucket, metadata_data);
  rc = noa_delete(bucket, metadata_labels);

  MPI_Finalize();
  return 0;
#if 0
  data_orig = malloc(sizeof(double) * total_size);

  for (int mpi_rank = 0; mpi_rank < 4; ++mpi_rank) {
    bucket->mpi_rank = mpi_rank;
    for (int k = 0; k < chunk_dims[2]; k++) {
      for (int j = 0; j < chunk_dims[1]; j++) {
        for (int i = 0; i < chunk_dims[0]; i++) {
          // data[k * chunk_dims[1] * chunk_dims[0] + j * chunk_dims[0] + i] =
          // (double)(k * chunk_dims[1] * chunk_dims[0] + j * chunk_dims[0] + i);
          data_orig[k * chunk_dims[1] * chunk_dims[0] + j * chunk_dims[0] + i] =
              bucket->mpi_rank;
        }
      }
    }

    rc = noa_put_chunk(bucket, metadata, data_orig, 0, header); assert(rc == 0);
    rc = noa_put_metadata(bucket, metadata); assert(rc == 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  rc = noa_free_metadata(bucket, metadata);
  assert(rc == 0);

  metadata = noa_get_metadata(bucket, "testObject");
  char* get_header;
  rc = noa_get_chunk(bucket, metadata, (void**)&data_read, &get_header);
  printf("get header: %s\n", get_header);

  // rc = noa_free_metadata(bucket, metadata);
  // assert(rc == 0);
  //    rc = noa_delete(bucket, metadata);
  noa_metadata__free_unpacked(metadata, NULL);
  rc = noa_container_close(bucket);
  assert(rc == 0);

  for (int rank = 0; rank < world_size; rank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == bucket->mpi_rank) {
      for (size_t k = 0; k < chunk_dims[2]; k++) {
        for (size_t j = 0; j < chunk_dims[1]; j++) {
          for (size_t i = 0; i < chunk_dims[0]; i++) {
            double original = data_orig[k * chunk_dims[1] * chunk_dims[0] + j * chunk_dims[0] + i];
            double retrieved = data_read[k * chunk_dims[1] * chunk_dims[0] + j * chunk_dims[0] + i];
            if (fabs(original - retrieved) > 0.005) fprintf(stderr, "error: %f %f\n", original, retrieved);
            //printf("%f ", data[k * chunk_dims[1] * chunk_dims[0] +
            //                   j * chunk_dims[0] + i]);
          }
          //printf("\n");
        }
        //printf("%d next layer...\n", bucket->mpi_rank);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  free(data_orig);
  free(data_read);
  free(get_header);

  MPI_Finalize();
  return 0;
#endif
}

int main(int argc, char* argv[]) {
  int fd;
  struct stat stat_buf;
  void* data;
  void* labels;
  float* fdata;
  float* flabels;
  int* pi;
  char* pc;
  float* pf;
  int n, features, classes;
  int x;

  MPI_Init(&argc, &argv);

  if (argc != 4) {
    printf("Usage: [container name] [image file] [label file]\n");
    return 1;
  }

  fd = open(argv[2], O_RDONLY);
  fstat(fd, &stat_buf);
  data = mmap(NULL, stat_buf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);

  fd = open(argv[3], O_RDONLY);
  fstat(fd, &stat_buf);
  labels = mmap(NULL, stat_buf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);

  pi = data;
  assert(__bswap_32(*pi) == 2051);
  n = __bswap_32(*(pi+1));
  features = 1;
  features *= __bswap_32(*(pi+2));
  features *= __bswap_32(*(pi+3));

  pi = labels;
  assert(__bswap_32(*pi) == 2049);
  assert(n == __bswap_32(*(pi+1)));
  classes = 10;

  printf("%d\n", n);
  printf("%d\n", features);
  printf("%d\n", classes);

  fdata = (float*)malloc(((n+CHUNK_SIZE-1)/CHUNK_SIZE) * CHUNK_SIZE * features * sizeof(float));
  flabels = (float*)malloc(((n+CHUNK_SIZE-1)/CHUNK_SIZE) * CHUNK_SIZE * classes * sizeof(float));

  pc = data + 16;
  pf = fdata;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < features; ++j) {
      *pf = ((float)*pc) / 255;
      ++pf;
      ++pc;
    }
  }

  pc = labels + 8;
  pf = flabels;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < classes; ++j) {
      *pf = (*pc == j) ? 1 : 0;
      ++pf;
    }
    ++pc;
  }

  //printf("%f\n", *(fdata + 600*CHUNK_SIZE*features));

  dataset_write(argv[1], n, features, classes, fdata, flabels);

  return 0;
}
