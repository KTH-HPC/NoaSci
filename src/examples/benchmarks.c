#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include "noa.h"

#define HEADER "hello world!"

const char* metadata_path = "./metadata";
const char* data_path = "./data";
const char* container_name = "testcontainer";
int world_size, world_rank;

void compare_array(double *array_original, double *array_retrieved, size_t size)
{
  for (size_t k = 0; k < size; k++) {
    double original = array_original[k];
    double retrieved = array_retrieved[k];
    if (fabs(original - retrieved) > 0.005) {
      fprintf(stderr, "error: %f %f\n", original, retrieved);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
}

void test_put(double *array, int dimensionality, long *dims, long *chunk_dims, FORMAT format, BACKEND backend)
{
  int rc;
  char object_name[1024];

  if (world_rank == 0) fprintf(stderr, "Test PUT: %s %s...\n", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");
  snprintf(object_name, 1024, "testObject-%s-%s", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");

  container* bucket;
  double t0 = MPI_Wtime();
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  double t1 = MPI_Wtime();
  double time_container_open = t1 - t0;
  double max_time_container_open;
  MPI_Reduce(&time_container_open, &max_time_container_open, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t2 = MPI_Wtime();
  NoaMetadata* metadata =
      noa_create_metadata(bucket, object_name, DOUBLE, format,
                          backend,
                          dimensionality, dims, chunk_dims);
  double t3 = MPI_Wtime();
  double time_create_metadata = t3 - t2;
  double max_time_create_metadata;
  MPI_Reduce(&time_create_metadata, &max_time_create_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // header should be ignored in the binary format
  char *header = NULL;
  if (format == HDF5) {
    header = malloc(sizeof(char) * 256);
    snprintf(header, 256, "%s", HEADER);
  }

  double t4 = MPI_Wtime();
  rc = noa_put_chunk(bucket, metadata, array, 0, header);
  assert(rc == 0);
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
    fprintf(stderr, "Open container (s)  : %f\n", max_time_container_open);
    fprintf(stderr, "Create Metadata (s) : %f\n", max_time_create_metadata);
    fprintf(stderr, "Put chunk (s)       : %f\n", max_time_put_chunk);
    fprintf(stderr, "Put metadata (s)    : %f\n", max_time_put_metadata);
    fprintf(stderr, "Free metadata (s)   : %f\n", max_time_free_metadata);
    fprintf(stderr, "Close container (s) : %f\n\n", max_time_container_close);
  }

  if (header != NULL) free(header);
  MPI_Barrier(MPI_COMM_WORLD);
}

void test_get(double *array, int dimensionality, long *dims, long *chunk_dims, FORMAT format, BACKEND backend)
{
  int rc = 0;
  char object_name[1024];

  if (world_rank == 0) fprintf(stderr, "Test GET: %s %s...\n", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");
  snprintf(object_name, 1024, "testObject-%s-%s", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");

  size_t total_size = 1;
  for (size_t i = 0; i < dimensionality; i++) total_size *= chunk_dims[i];

  // get data back and verify
  double *verify_data = NULL;
  char *verify_header = NULL;
  container *bucket;
  NoaMetadata *metadata = NULL;

  double t0 = MPI_Wtime();
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  double t1 = MPI_Wtime();
  double time_container_open = t1 - t0;
  double max_time_container_open;
  MPI_Reduce(&time_container_open, &max_time_container_open, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t2 = MPI_Wtime();
  metadata = noa_get_metadata(bucket, object_name);
  double t3 = MPI_Wtime();
  double time_get_metadata = t3 - t2;
  double max_time_get_metadata;
  MPI_Reduce(&time_get_metadata, &max_time_get_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // check metadata
  if (metadata->n_dims != dimensionality)
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
  double t4 = MPI_Wtime();
  rc = noa_get_chunk(bucket, metadata, (void**)&verify_data, &verify_header, bucket->mpi_rank);
  assert(rc == 0);
  double t5 = MPI_Wtime();
  double time_get_chunk = t5 - t4;
  double max_time_get_chunk;
  MPI_Reduce(&time_get_chunk, &max_time_get_chunk, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // check header if not usinb binary format
  if (format == HDF5)
    if (strcmp(HEADER, verify_header) != 0)
      fprintf(stderr, "Header recovered incorrectly!!!!\n");

  // check array
  compare_array(array, verify_data, total_size);

  double t6 = MPI_Wtime();
  rc = noa_free_metadata(bucket, metadata);
  assert(rc == 0);
  double t7 = MPI_Wtime();
  double time_free_metadata = t7 - t6;
  double max_time_free_metadata;
  MPI_Reduce(&time_free_metadata, &max_time_free_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t8 = MPI_Wtime();
  rc = noa_container_close(bucket);
  assert(rc == 0);
  double t9 = MPI_Wtime();
  double time_container_close = t9 - t8;
  double max_time_container_close;
  MPI_Reduce(&time_container_close, &max_time_container_close, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    fprintf(stderr, "Open container (s)  : %f\n", max_time_container_open);
    fprintf(stderr, "Get Metadata (s)    : %f\n", max_time_get_metadata);
    fprintf(stderr, "Get chunk (s)       : %f\n", max_time_get_chunk);
    fprintf(stderr, "Free Metadata (s)   : %f\n", max_time_free_metadata);
    fprintf(stderr, "Close container (s) : %f\n\n", max_time_container_close);
  }

  free(verify_data);
  if (verify_header != NULL) free(verify_header);
  MPI_Barrier(MPI_COMM_WORLD);
}

void test_delete(FORMAT format, BACKEND backend)
{
  int rc = 0;
  char object_name[1024];
  container *bucket;
  NoaMetadata *metadata = NULL;

  if (world_rank == 0) fprintf(stderr, "Test DELETE: %s %s...\n", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");
  snprintf(object_name, 1024, "testObject-%s-%s", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");

  double t0 = MPI_Wtime();
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  double t1 = MPI_Wtime();
  double time_container_open = t1 - t0;
  double max_time_container_open;
  MPI_Reduce(&time_container_open, &max_time_container_open, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t2 = MPI_Wtime();
  metadata = noa_get_metadata(bucket, object_name);
  double t3 = MPI_Wtime();
  double time_get_metadata = t3 - t2;
  double max_time_get_metadata;
  MPI_Reduce(&time_get_metadata, &max_time_get_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // delete object
  double t4 = MPI_Wtime();
  rc = noa_delete(bucket, metadata);
  assert(rc == 0);
  double t5 = MPI_Wtime();
  double time_delete = t5 - t4;
  double max_time_delete;
  MPI_Reduce(&time_delete, &max_time_delete, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t6 = MPI_Wtime();
  rc = noa_container_close(bucket);
  assert(rc == 0);
  double t7 = MPI_Wtime();
  double time_container_close = t7 - t6;
  double max_time_container_close;
  MPI_Reduce(&time_container_close, &max_time_container_close, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    fprintf(stderr, "Open container (s)  : %f\n", max_time_container_open);
    fprintf(stderr, "Get Metadata (s)    : %f\n", max_time_get_metadata);
    fprintf(stderr, "Delete chunk (s)    : %f\n", max_time_delete);
    fprintf(stderr, "Close container (s) : %f\n\n", max_time_container_close);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void test_put_single(double *array, int dimensionality, long *dims, long *chunk_dims, FORMAT format, BACKEND backend)
{
  int rc = 0;
  char object_name[1024];
  container *bucket;
  NoaMetadata *metadata = NULL;

  if (world_rank == 0) fprintf(stderr, "Test SINGLE PUT: %s %s...\n", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");
  snprintf(object_name, 1024, "testObject-%s-%s", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");

  double t0 = MPI_Wtime();
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  assert(rc == 0);
  double t1 = MPI_Wtime();
  double time_container_open = t1 - t0;
  double max_time_container_open;
  MPI_Reduce(&time_container_open, &max_time_container_open, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t2 = MPI_Wtime();
  metadata =
      noa_create_metadata(bucket, object_name, DOUBLE, format,
                          backend,
                          dimensionality, dims, chunk_dims);
  double t3 = MPI_Wtime();
  double time_create_metadata = t3 - t2;
  double max_time_create_metadata;
  MPI_Reduce(&time_create_metadata, &max_time_create_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // header should be ignored in the binary format
  char *header = NULL;
  if (format == HDF5) {
    header = malloc(sizeof(char) * 256);
    snprintf(header, 256, "%s", HEADER);
  }

  double t4 = 0.0, t5 = 0.0;
  if (world_rank == 0) {
    t4 = MPI_Wtime();
    for (int chunk_id = 0; chunk_id < metadata->num_chunks; chunk_id++) {
      rc = noa_put_chunk_by_id(bucket, metadata, chunk_id, array, 0, header);
    }
    t5 = MPI_Wtime();
  }

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
  assert(rc == 0);
  double t11 = MPI_Wtime();
  double time_container_close = t11 - t10;
  double max_time_container_close;
  MPI_Reduce(&time_container_close, &max_time_container_close, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    fprintf(stderr, "Open container (s)  : %f\n", max_time_container_open);
    fprintf(stderr, "Create Metadata (s) : %f\n", max_time_create_metadata);
    fprintf(stderr, "Put chunk (s)       : %f\n", t5 - t4);
    fprintf(stderr, "Put metadata (s)    : %f\n", max_time_put_metadata);
    fprintf(stderr, "Free metadata (s)   : %f\n", max_time_free_metadata);
    fprintf(stderr, "Close container (s) : %f\n\n", max_time_container_close);
  }

  if (header != NULL) free(header);
  MPI_Barrier(MPI_COMM_WORLD);
}

void test_get_single(double *array, int dimensionality, long *dims, long *chunk_dims, FORMAT format, BACKEND backend)
{
  int rc = 0;
  char object_name[1024];
  container *bucket = NULL;
  NoaMetadata *metadata = NULL;
  double *verify_data = NULL;
  char *verify_header = NULL;

  if (world_rank == 0) fprintf(stderr, "Test SINGLE GET: %s %s...\n", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");
  snprintf(object_name, 1024, "testObject-%s-%s", format == HDF5 ? "HDF5" : "BINARY", backend == 0 ? "POSIX" : "MERO");

  double t0 = MPI_Wtime();
  rc = noa_container_open(&bucket, container_name, metadata_path, data_path);
  assert(rc == 0);
  double t1 = MPI_Wtime();
  double time_container_open = t1 - t0;
  double max_time_container_open;
  MPI_Reduce(&time_container_open, &max_time_container_open, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t2 = MPI_Wtime();
  metadata = noa_get_metadata(bucket, object_name);
  double t3 = MPI_Wtime();
  double time_get_metadata = t3 - t2;
  double max_time_get_metadata;
  MPI_Reduce(&time_get_metadata, &max_time_get_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // check metadata
  if (metadata->n_dims != dimensionality)
    fprintf(stderr, "Error: Dimensionality recovered is incorrect!!!!!!\n");

  for (size_t i = 0; i < dimensionality; i++) {
    if (metadata->dims[i] != dims[i])
      fprintf(stderr, "Error: Dim[%ld] should be %ld but recovered as %ld!!!!\n", i, dims[i], metadata->dims[i]);
  }

  size_t total_chunk_size = 1;
  for (size_t i = 0; i < dimensionality; i++) {
    if (metadata->chunk_dims[i] != chunk_dims[i])
      fprintf(stderr, "Error: chunk_dims[%ld] should be %ld but recovered as %ld!!!!\n", i, chunk_dims[i], metadata->chunk_dims[i]);
    total_chunk_size *= metadata->chunk_dims[i];
  }

  // get actual data
  double t4 = 0.0, t5 = 0.0;
  if (world_rank == 0) {
    for (int chunk_id = 0; chunk_id < metadata->num_chunks; chunk_id++) {
      t4 = MPI_Wtime();
      rc = noa_get_chunk(bucket, metadata, (void**)&verify_data, &verify_header, chunk_id);
      assert(rc == 0);
      t5 = MPI_Wtime();
      compare_array(array, verify_data, total_chunk_size);
      free(verify_data);

      // check header if not usinb binary format
      if (format == HDF5)
        if (strcmp(HEADER, verify_header) != 0)
          fprintf(stderr, "Header recovered incorrectly!!!!\n");
      if (verify_header != NULL) { free(verify_header); verify_header = NULL; }
    }
  }

  double t6 = MPI_Wtime();
  rc = noa_free_metadata(bucket, metadata);
  double t7 = MPI_Wtime();
  double time_free_metadata = t7 - t6;
  double max_time_free_metadata;
  MPI_Reduce(&time_free_metadata, &max_time_free_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  double t8 = MPI_Wtime();
  rc = noa_container_close(bucket);
  double t9 = MPI_Wtime();
  double time_container_close = t9 - t8;
  double max_time_container_close;
  MPI_Reduce(&time_container_close, &max_time_container_close, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) {
    fprintf(stderr, "Open container (s)  : %f\n", max_time_container_open);
    fprintf(stderr, "Get Metadata (s)    : %f\n", max_time_get_metadata);
    fprintf(stderr, "Get chunk (s)       : %f\n", t5 - t4);
    fprintf(stderr, "Free Metadata (s)   : %f\n", max_time_free_metadata);
    fprintf(stderr, "Close container (s) : %f\n\n", max_time_container_close);
  }
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
  noa_init(mero_filename, 4096, world_rank % 4, 0);

  long dims[] = {
      1,
      16384,
      16384,
  };

  long chunk_dims[] = {
      1,
      8192,
      8192,
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

  if (world_rank == 0) fprintf(stderr, "INFO: Chunk size: %d x %f MiB\n", world_size, (total_size * sizeof(double) / 1024.0 / 1024.0));

//  // test POSIX HDF5 format
//  test_put(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, HDF5, POSIX);
//  test_get(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, HDF5, POSIX);
//  test_delete(HDF5, POSIX);

  // test MERO HDF5 format
  test_put(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, HDF5, MERO);
  test_get(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, HDF5, MERO);
  test_delete(HDF5, MERO);

//  // test POSIX BINARY format
//  test_put(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, BINARY, POSIX);
//  test_get(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, BINARY, POSIX);
//  test_delete(BINARY, POSIX);

  // test MERO BINARY format
  test_put(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, BINARY, MERO);
  test_get(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, BINARY, MERO);
  test_delete(BINARY, MERO);

//  // test POSIX BINARY format
//  test_put_single(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, BINARY, POSIX);
//  test_get_single(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, BINARY, POSIX);
//  test_delete(BINARY, POSIX);

  // test MERO BINARY format
  test_put_single(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, BINARY, MERO);
  test_get_single(data, sizeof(dims) / sizeof(*dims), dims, chunk_dims, BINARY, MERO);
  test_delete(BINARY, MERO);

  free (data);
  noa_finalize();
  MPI_Finalize();
  return 0;
}
