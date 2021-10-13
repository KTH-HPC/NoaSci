#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "noa.h"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  size_t hostname_len = 256;
  char hostname[hostname_len];
  assert( gethostname(hostname, hostname_len) == 0);

  char mero_filename[265];
  snprintf(mero_filename, 265, "./sagerc_%s", hostname);
  noa_init(mero_filename, 4096, world_rank, 0);

  const char* metadata_path = "./metadata";
  const char* data_path = "./data";
  const char* container_name = "testcontainer";
  container* bucket;

  int rc;
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);

  long dims[] = {
      16,
      4096,
      4096,
  };

  long chunk_dims[] = {
      16,
      2048,
      2048,
  };  // 1x2x2

  NoaMetadata* metadata =
      noa_create_metadata(bucket, "testObject", DOUBLE, VTK,
#ifdef USE_MERO
                          MERO,
#else
                          POSIX,
#endif
                          sizeof(dims) / sizeof(long), dims, chunk_dims);
  printf("dimensionality: %ld\n", metadata->n_dims);
  printf("dims: %ld %ld\n", metadata->dims[0], metadata->dims[1]);
  printf("num_chunks: %d\n", metadata->num_chunks);
  printf("dims: %ld %ld\n", metadata->dims[0], metadata->dims[1]);
  printf("n_chunk_dims: %ld\n", metadata->n_chunk_dims);
  printf("chunk_dims: %ld %ld\n", metadata->chunk_dims[0],
         metadata->chunk_dims[1]);

  size_t total_size = chunk_dims[0];
  for (size_t i = 1; i < sizeof(dims) / sizeof(long); i++) {
    total_size *= chunk_dims[i];
  }

  double *data = malloc(sizeof(double) * total_size);
  if (data == NULL) printf("fail to allocate %ld !\n", sizeof(double) * total_size);
  double counter = 0.0;
  for (size_t k = 0; k < total_size; k++) {
    data[k] = counter++;
  }

  const char* header = "testHeader";
  rc = noa_put_chunk(bucket, metadata, data, 0, header);

  rc = noa_put_metadata(bucket, metadata);
  assert(rc == 0);
  MPI_Barrier(MPI_COMM_WORLD);

  free(data);
  rc = noa_free_metadata(bucket, metadata);
  assert(rc == 0);

  metadata = noa_get_metadata(bucket, "testObject");
  char* get_header;
  rc = noa_get_chunk(bucket, metadata, (void**)&data, &get_header, bucket->mpi_rank);
  if (get_header != NULL) printf("get header: %s\n", get_header);

  // rc = noa_free_metadata(bucket, metadata);
  // assert(rc == 0);
  //    rc = noa_delete(bucket, metadata);
  noa_metadata__free_unpacked(metadata, NULL);
  rc = noa_container_close(bucket);
  assert(rc == 0);

  for (int rank = 0; rank < world_size; rank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == bucket->mpi_rank) {
      for (size_t k = 0; k < total_size; k++) {
        double original = data[k];
        double retrieved = data[k];
        if (fabs(original - retrieved) > 0.005) fprintf(stderr, "error: %f %f\n", original, retrieved);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  free(data);
  if (get_header != NULL) free(get_header);

  MPI_Finalize();
  return 0;
}
