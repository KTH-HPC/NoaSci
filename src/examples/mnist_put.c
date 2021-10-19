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

void compare_array(float *array_original, float *array_retrieved, size_t size)
{
  for (size_t k = 0; k < size; k++) {
    float original = array_original[k];
    float retrieved = array_retrieved[k];
    if (fabs(original - retrieved) > 0.005) fprintf(stderr, "error: %f %f\n", original, retrieved);
  }
}


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

  const char* metadata_path = "./metadata";
  const char* data_path = "./data";
  //const char* container_name = "testcontainer";
  container* bucket;

  int rc;
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);

      printf("metadata_path:   %s\n", bucket->name);
      printf("key_value_store: %s\n", bucket->key_value_store);
      printf("object_store:    %s\n", bucket->object_store);

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
  }

  rc = noa_put_metadata(bucket, metadata_data); assert(rc == 0);
  rc = noa_put_metadata(bucket, metadata_labels); assert(rc == 0);

  rc = noa_free_metadata(bucket, metadata_data);
  rc = noa_free_metadata(bucket, metadata_labels);

  rc = noa_container_close(bucket);
  MPI_Barrier(MPI_COMM_WORLD);

  // get and verify
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path); assert(rc == 0);
  metadata_data = noa_get_metadata(bucket, "data");
  metadata_labels = noa_get_metadata(bucket, "labels");

  printf("datatype: %d\n", metadata_data->datatype);
  printf("id: %s\n", metadata_data->id);

  float *verify_data = malloc(sizeof(float) * CHUNK_SIZE * features);
  float *verify_labels = malloc(sizeof(float) * CHUNK_SIZE * classes);
  // header is always NULL for binary since unimplemented
  char *verify_header = NULL;

  for (int chunk = 0; chunk < chunks; chunk++) {
    printf("Verifying chunk: %d\n", chunk);
    rc = noa_get_chunk(bucket, metadata_data, (void**)&verify_data, &verify_header, chunk);
    compare_array(verify_data, data + chunk*CHUNK_SIZE*features, CHUNK_SIZE * features);
    free(verify_data);

    rc = noa_get_chunk(bucket, metadata_labels, (void**)&verify_labels, &verify_header, chunk);
    compare_array(verify_labels, labels + chunk*CHUNK_SIZE*classes, CHUNK_SIZE * classes);
    free(verify_labels);
  }

  // delete
  printf("Deleting chunks\n");
  rc = noa_delete(bucket, metadata_data);
  rc = noa_delete(bucket, metadata_labels);
  rc = noa_container_close(bucket);

  MPI_Finalize();
  return 0;
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
