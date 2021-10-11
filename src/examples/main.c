#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "noa.h"

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const char* metadata_path = "./metadata";
  const char* data_path = "./data";
  const char* container_name = "testcontainer";
  container* bucket;

  int rc;
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);

  //    printf("metadata_path:   %s\n", bucket->name);
  //    printf("key_value_store: %s\n", bucket->key_value_store);
  //    printf("object_store:    %s\n", bucket->object_store);
  //

  // long dims[]       = { 2, 4, 8 };
  // long chunk_dims[] = { 2, 2, 4 }; // 1x2x2
  long dims[] = {
      1,
      2,
      4,
  };
  long chunk_dims[] = {
      1,
      1,
      2,
  };  // 1x2x2
  NoaMetadata* metadata =
      noa_create_metadata(bucket, "testObject", DOUBLE, HDF5, POSIX,
                          sizeof(dims) / sizeof(long), dims, chunk_dims);
  printf("dimensionality: %ld\n", metadata->n_dims);
  printf("dims: %ld %ld\n", metadata->dims[0], metadata->dims[1]);
  printf("num_chunks: %d\n", metadata->num_chunks);
  printf("dims: %ld %ld\n", metadata->dims[0], metadata->dims[1]);
  printf("n_chunk_dims: %ld\n", metadata->n_chunk_dims);
  printf("chunk_dims: %ld %ld\n", metadata->chunk_dims[0],
         metadata->chunk_dims[1]);

  double* data;
  size_t total_size = chunk_dims[0];
  for (size_t i = 1; i < sizeof(dims) / sizeof(long); i++) {
    total_size *= chunk_dims[i];
  }

  data = malloc(sizeof(double) * total_size);
  for (int k = 0; k < chunk_dims[2]; k++) {
    for (int j = 0; j < chunk_dims[1]; j++) {
      for (int i = 0; i < chunk_dims[0]; i++) {
        // data[k * chunk_dims[1] * chunk_dims[0] + j * chunk_dims[0] + i] =
        // (double)(k * chunk_dims[1] * chunk_dims[0] + j * chunk_dims[0] + i);
        data[k * chunk_dims[1] * chunk_dims[0] + j * chunk_dims[0] + i] =
            bucket->mpi_rank;
      }
    }
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
  rc = noa_get_chunk(bucket, metadata, (void**)&data, &get_header);
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
            printf("%f ", data[k * chunk_dims[1] * chunk_dims[0] +
                               j * chunk_dims[0] + i]);
          }
          printf("\n");
        }
        printf("%d next layer...\n", bucket->mpi_rank);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  free(data);
  free(get_header);

  MPI_Finalize();
  return 0;
}
