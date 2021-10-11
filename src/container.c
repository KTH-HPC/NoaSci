#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "noa.h"

int noa_container_open(container** bucket, const char* container_name,
                       const char* metadata_path, const char* data_path) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0) {
    struct stat sb;
    if (!(stat(metadata_path, &sb) == 0 && S_ISDIR(sb.st_mode))) {
      fprintf(stderr, "Metadata data path %s does not exist.\n", metadata_path);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!(stat(data_path, &sb) == 0 && S_ISDIR(sb.st_mode))) {
      fprintf(stderr, "Data path %s does not exist.\n", data_path);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // create bucket
  *bucket = malloc(sizeof(container));
  (*bucket)->name = malloc(sizeof(char) * strlen(container_name) + 1);
  (*bucket)->key_value_store = malloc(sizeof(char) * strlen(metadata_path) + 1);
  (*bucket)->object_store = malloc(sizeof(char) * strlen(data_path) + 1);
  assert(*bucket != NULL && (*bucket)->name != NULL &&
         (*bucket)->key_value_store != NULL && (*bucket)->object_store != NULL);

  // copy data
  strncpy((*bucket)->name, container_name, strlen(container_name) + 1);
  strncpy((*bucket)->key_value_store, metadata_path, strlen(metadata_path) + 1);
  strncpy((*bucket)->object_store, data_path, strlen(data_path) + 1);
  (*bucket)->mpi_rank = mpi_rank;

  return 0;
}

int noa_container_close(container* bucket) {
  // free data
  free(bucket->name);
  free(bucket->key_value_store);
  free(bucket->object_store);
  free(bucket);

  // the end
  return 0;
}
