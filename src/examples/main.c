#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "noa.h"

const char* metadata_path = "./metadata";
const char* data_path = "./data";
const char* container_name = "testcontainer";
int world_size, world_rank;

void compare_array(double *array_original, double *array_retrieved, size_t size)
{
  for (size_t k = 0; k < size; k++) {
    double original = array_original[k];
    double retrieved = array_retrieved[k];
    if (fabs(original - retrieved) > 0.005) fprintf(stderr, "error: %f %f\n", original, retrieved);
  }
}

void test_binary(double *array, int dimensionality, long *dims, long *chunk_dims, BACKEND backend)
{
  char object_name[1024];
  if (backend == MERO) {
    if (world_rank == 0) fprintf(stderr, "Test: binary MERO...\n");
    snprintf(object_name, 1024, "testObject-binary-%s", "MERO");
  }
  else {
    if (world_rank == 0) fprintf(stderr, "Test: binary POSIX...\n");
    snprintf(object_name, 1024, "testObject-binary-%s", "POSIX");
  }

  int rc;
  if (world_rank == 0) fprintf(stderr, "Opening container %s at %s and %s...\n", container_name, metadata_path, data_path);

  container* bucket;
  double t0 = MPI_Wtime();
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  double t1 = MPI_Wtime();
  double time_container_open = t1 - t0;
  double max_time_container_open;
  MPI_Reduce(&time_container_open, &max_time_container_open, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (world_rank == 0) fprintf(stderr, "creating object %s...\n", object_name);
  double t2 = MPI_Wtime();
  NoaMetadata* metadata =
      noa_create_metadata(bucket, object_name, DOUBLE, BINARY,
                          backend,
                          dimensionality, dims, chunk_dims);
  double t3 = MPI_Wtime();
  double time_create_metadata = t3 - t2;
  double max_time_create_metadata;
  MPI_Reduce(&time_create_metadata, &max_time_create_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // header should be ignored in the binary format
  const char* header = NULL;
  double t4 = MPI_Wtime();
  rc = noa_put_chunk(bucket, metadata, array, 0, header);
  double t5 = MPI_Wtime();
  double time_put_chunk = t5 - t4;
  double max_time_put_chunk;
  MPI_Reduce(&time_put_chunk, &max_time_put_chunk, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t6 = MPI_Wtime();
  rc = noa_put_metadata(bucket, metadata);
  assert(rc == 0);
  double t7 = MPI_Wtime();
  double time_put_metadata = t7 - t6;
  double max_time_put_metadata;
  MPI_Reduce(&time_put_metadata, &max_time_put_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t8 = MPI_Wtime();
  rc = noa_free_metadata(bucket, metadata);
  assert(rc == 0);
  double t9 = MPI_Wtime();
  double time_free_metadata = t9 - t8;
  double max_time_free_metadata;
  MPI_Reduce(&time_free_metadata, &max_time_free_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t10 = MPI_Wtime();
  rc = noa_container_close(bucket);
  double t11 = MPI_Wtime();
  double time_container_close = t11 - t10;
  double max_time_container_close;
  MPI_Reduce(&time_container_close, &max_time_container_close, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    fprintf(stderr, "Test Put Binary HDF5 on %d backend:\n", backend);
    fprintf(stderr, "Open container (s)  : %f\n", max_time_container_open);
    fprintf(stderr, "Create Metadata (s) : %f\n", max_time_create_metadata);
    fprintf(stderr, "Put chunk (s)       : %f\n", max_time_put_chunk);
    fprintf(stderr, "Put metadata (s)    : %f\n", max_time_put_metadata);
    fprintf(stderr, "Free metadata (s)   : %f\n", max_time_free_metadata);
    fprintf(stderr, "Close container (s) : %f\n", max_time_container_close);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  size_t total_size = 1;
  for (size_t i = 0; i < dimensionality; i++) total_size *= chunk_dims[i];

  // get data back and verify
  double *verify_data = NULL;
  char *verify_header = NULL;

  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  double t12 = MPI_Wtime();
  metadata = noa_get_metadata(bucket, object_name);
  double t13 = MPI_Wtime();
  double time_get_metadata = t13 - t12;
  double max_time_get_metadata;
  MPI_Reduce(&time_get_metadata, &max_time_get_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // check metadata
  if (metadata->n_dims == dimensionality)
    fprintf(stderr, "Dimensionality recovered is correct...\n");
  else
    fprintf(stderr, "Error: Dimensionality recovered is incorrect!!!!!!\n");

  for (size_t i = 0; i < dimensionality; i++) {
    if (metadata->dims[i] != dims[i])
      fprintf(stderr, "Error: Dim[%ld] should be %ld but recovered as %ld!!!!\n", i, dims[i], metadata->dims[i]);
  }

  for (size_t i = 0; i < dimensionality; i++) {
    if (metadata->chunk_dims[i] != chunk_dims[i])
      fprintf(stderr, "Error: chunk_dims[%ld] should be %ld but recovered as %ld!!!!\n", i, chunk_dims[i], metadata->chunk_dims[i]);
  }

  // get actual data
  double t14 = MPI_Wtime();
  rc = noa_get_chunk(bucket, metadata, (void**)&verify_data, &verify_header, bucket->mpi_rank);
  double t15 = MPI_Wtime();
  double time_get_chunk = t15 - t14;
  double max_time_get_chunk;
  MPI_Reduce(&time_get_chunk, &max_time_get_chunk, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // header should be ignored
  compare_array(array, verify_data, total_size);

  // delete object
  if (world_rank == 0) fprintf(stderr, "deleting object %s...\n", object_name);
  double t16 = MPI_Wtime();
  rc = noa_delete(bucket, metadata);
  double t17 = MPI_Wtime();
  double time_delete = t17 - t16;
  double max_time_delete;
  MPI_Reduce(&time_delete, &max_time_delete, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (world_rank == 0) fprintf(stderr, "cleaning up %s...\n", object_name);
  // metadata is already freed during delete
  assert(rc == 0);
  rc = noa_container_close(bucket);

  if (world_rank == 0) {
    fprintf(stderr, "Test Get Binary HDF5 on %d backend:\n", backend);
    fprintf(stderr, "Get Metadata (s)    : %f\n", max_time_get_metadata);
    fprintf(stderr, "Get chunk (s)       : %f\n", max_time_get_chunk);
    fprintf(stderr, "Delete Object (s)   : %f\n", max_time_delete);

  }
  MPI_Barrier(MPI_COMM_WORLD);

  free(verify_data);
}


void test_hdf5(double *array, int dimensionality, long *dims, long *chunk_dims, BACKEND backend)
{
  char object_name[1024];
  if (backend == MERO) {
    if (world_rank == 0) fprintf(stderr, "Test: HDF5 MERO...\n");
    snprintf(object_name, 1024, "testObject-HDF5-%s", "MERO");
  }
  else {
    if (world_rank == 0) fprintf(stderr, "Test: HDF5 POSIX...\n");
    snprintf(object_name, 1024, "testObject-HDF5-%s", "POSIX");
  }

  int rc;
  if (world_rank == 0) fprintf(stderr, "Opening container %s at %s and %s...\n", container_name, metadata_path, data_path);

  container* bucket;
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);

  if (world_rank == 0) fprintf(stderr, "creating object %s...\n", object_name);
  NoaMetadata* metadata =
      noa_create_metadata(bucket, object_name, DOUBLE, HDF5,
                          backend,
                          dimensionality, dims, chunk_dims);

  const char* header = "hello world header";
  rc = noa_put_chunk(bucket, metadata, array, 0, header);

  rc = noa_put_metadata(bucket, metadata);
  assert(rc == 0);
  rc = noa_free_metadata(bucket, metadata);
  assert(rc == 0);
  rc = noa_container_close(bucket);
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  size_t total_size = 1;
  for (size_t i = 0; i < dimensionality; i++) total_size *= chunk_dims[i];

  // get data back and verify
  double *verify_data = NULL;
  char *verify_header = NULL;

  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  metadata = noa_get_metadata(bucket, object_name);

  // check metadata
  if (metadata->n_dims == dimensionality)
    fprintf(stderr, "Dimensionality recovered is correct...\n");
  else
    fprintf(stderr, "Error: Dimensionality recovered is incorrect!!!!!!\n");

  for (size_t i = 0; i < dimensionality; i++) {
    if (metadata->dims[i] != dims[i])
      fprintf(stderr, "Error: Dim[%ld] should be %ld but recovered as %ld!!!!\n", i, dims[i], metadata->dims[i]);
  }

  for (size_t i = 0; i < dimensionality; i++) {
    if (metadata->chunk_dims[i] != chunk_dims[i])
      fprintf(stderr, "Error: chunk_dims[%ld] should be %ld but recovered as %ld!!!!\n", i, chunk_dims[i], metadata->chunk_dims[i]);
  }

  // get actual data
  rc = noa_get_chunk(bucket, metadata, (void**)&verify_data, &verify_header, bucket->mpi_rank);

  // check header
  if (strcmp(header, verify_header) == 0)
    fprintf(stderr, "Header recovered correctly!\n");
  else
    fprintf(stderr, "Header recovered incorrectly!!!!\n");

  compare_array(array, verify_data, total_size);

  // delete object
  if (world_rank == 0) fprintf(stderr, "deleting object %s...\n", object_name);
  rc = noa_delete(bucket, metadata);

  if (world_rank == 0) fprintf(stderr, "cleaning up %s...\n", object_name);
  assert(rc == 0);
  rc = noa_container_close(bucket);
  MPI_Barrier(MPI_COMM_WORLD);

  free(verify_data);
  free(verify_header);
}

void test_hdf5_single(double *array, int dimensionality, long *dims, long *chunk_dims, BACKEND backend)
{
  int rc = 0;

  // test individual chunks
  char object_name[1024];
  if (backend == MERO) {
    if (world_rank == 0) fprintf(stderr, "Test: single process HDF5 MERO...\n");
    snprintf(object_name, 1024, "testObject-HDF5-%s", "MERO");
  }
  else {
    if (world_rank == 0) fprintf(stderr, "Test: single process HDF5 POSIX...\n");
    snprintf(object_name, 1024, "testObject-HDF5-%s", "POSIX");
  }

  if (world_rank == 0) fprintf(stderr, "Opening container %s at %s and %s...\n", container_name, metadata_path, data_path);
  container *bucket;
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  assert(rc == 0);

  size_t total_chunk_size = chunk_dims[0];
  for (size_t i = 1; i < dimensionality; i++) {
    total_chunk_size *= chunk_dims[i];
  }

  if (world_rank == 0) fprintf(stderr, "creating object %s...\n", object_name);
  NoaMetadata* metadata =
      noa_create_metadata(bucket, object_name, DOUBLE, BINARY,
                          backend,
                          dimensionality, dims, chunk_dims);

  const char *header = NULL;
  if (world_rank == 0) {
    for (int chunk_id = 0; chunk_id < metadata->num_chunks; chunk_id++) {
      fprintf(stderr, "Putting chunk %d...\n", chunk_id);
      rc = noa_put_chunk_by_id(bucket, metadata, chunk_id, array, 0, header);
    }
  }

  rc = noa_put_metadata(bucket, metadata);
  assert(rc == 0);
  rc = noa_container_close(bucket);
  assert(rc == 0);
  MPI_Barrier(MPI_COMM_WORLD);
  sleep(1);

  double *verify_data = NULL;

  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  metadata = noa_get_metadata(bucket, object_name);

  // check metadata
  if (metadata->n_dims == dimensionality)
    fprintf(stderr, "Dimensionality recovered is correct...\n");
  else
    fprintf(stderr, "Error: Dimensionality recovered is incorrect!!!!!!\n");

  // get actual data
  if (world_rank == 0) {
    for (int chunk_id = 0; chunk_id < metadata->num_chunks; chunk_id++) {
      fprintf(stderr, "Verifying chunk %d...\n", chunk_id);
      rc = noa_get_chunk(bucket, metadata, (void**)&verify_data, NULL, bucket->mpi_rank);
      assert(rc == 0);
      compare_array(array, verify_data, total_chunk_size);
      free(verify_data);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // delete object
  if (world_rank == 0) fprintf(stderr, "deleting object %s...\n", object_name);
  rc = noa_delete(bucket, metadata);

  if (world_rank == 0) fprintf(stderr, "cleaning up %s...\n", object_name);
  assert(rc == 0);
  rc = noa_container_close(bucket);
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  size_t hostname_len = 256;
  char hostname[hostname_len];
  assert( gethostname(hostname, hostname_len) == 0);

  char mero_filename[265];
  snprintf(mero_filename, 265, "./%s", hostname);
printf("rank: %d\n", world_rank %4);
  noa_init(mero_filename, 4096, world_rank%4, 0);

  long dims[] = {
      16,
      512,
      256,
  };

  long chunk_dims[] = {
      16,
      256,
      128,
  };  // 1x2x2

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

  // test HDF5 format
  test_hdf5(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, POSIX);
  test_hdf5(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, MERO);

  // test binary format
  test_binary(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, POSIX);
  test_binary(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, MERO);

  // test single proc and multiple chunks
  test_hdf5_single(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, POSIX);
  test_hdf5_single(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, MERO);

  free (data);
  noa_finalize();
  MPI_Finalize();
  return 0;
}
