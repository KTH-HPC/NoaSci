#include "private/binary_backend.h"

#include <errno.h>
#include <fcntl.h>
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// aux tool to write a file
static int write_binary_file(const char *filename, const void *data,
                             const size_t size, const char *header) {
  int fd, rc;

  /* open file write and sync to disk */
  if ((fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644)) == -1) {
    fprintf(stderr, "Unable to open %s for writing: %s\n", filename,
            strerror(errno));
    return -1;
  }

  if (header != NULL) {
    rc = write(fd, header, strlen(header));
    if (rc == -1) {
      fprintf(stderr, "Writing to %s failed: %s\n", filename, strerror(errno));
      return -1;
    }
  }

  rc = write(fd, data, size);
  if (rc == -1) {
    fprintf(stderr, "Writing to %s failed: %s\n", filename, strerror(errno));
    return -1;
  }

  rc = fsync(fd);
  if (rc < 0) {
    fprintf(stderr, "syncing to %s failed: %s\n", filename, strerror(errno));
    return -1;
  }
  close(fd);

  return 0;
}

// aux to to read a file
static int read_binary_file(const char *filename, void **data, size_t *size) {
  struct stat stbuf;
  int fd, ret;

  /* open file and read size info from inode */
  fd = open(filename, O_RDONLY);
  if (fd == -1) {
    fprintf(stderr, "Unable to open %s for reading: %s\n", filename,
            strerror(errno));
    return -1;
  }

  if ((fstat(fd, &stbuf) != 0) || (!S_ISREG(stbuf.st_mode))) {
    fprintf(stderr, "cannot get information of %s: %s\n", filename,
            strerror(errno));
    return -1;
  }

  *size = stbuf.st_size;
  *data = (char *)malloc(*size);
  ret = read(fd, *data, *size);
  if (ret == -1) {
    fprintf(stderr, "Reading from %s failed: %s\n", filename, strerror(errno));
    return -1;
  }

  close(fd);
  return 0;
}

int put_object_chunk_binary(const container *bucket,
                            const NoaMetadata *object_metadata,
                            const char *suffix, const void *data,
                            const size_t offset, const char *header) {
  // create storage path
  // (data storage)/(uuid)-(chunk id).h5\0
  size_t total_file_size = 1;
  size_t chunk_path_len =
      strlen(bucket->object_store) + strlen(object_metadata->id) +
      snprintf(NULL, 0, "%d.%s", bucket->mpi_rank, suffix) + 3;
  char *chunk_path = malloc(sizeof(char) * chunk_path_len);
  snprintf(chunk_path, chunk_path_len, "%s/%s-%d.%s", bucket->object_store,
           object_metadata->id, bucket->mpi_rank, suffix);

  switch (object_metadata->datatype) {
    case INT:
      total_file_size = sizeof(int);
      break;
    case DOUBLE:
      total_file_size = sizeof(double);
      break;
    case FLOAT:
      total_file_size = sizeof(float);
      break;
    default:
      fprintf(stderr, "Error: Unknown datatype!\n");
      break;
  }

  for (size_t i = 0; i < object_metadata->n_dims; i++)
    total_file_size *= object_metadata->chunk_dims[i];

  write_binary_file(chunk_path, &data[offset], total_file_size, header);
  return 0;
}

int get_object_chunk_binary(const container *bucket,
                            const NoaMetadata *object_metadata,
                            const char *suffix, void **data, char **header, int chunk_id) {
  fprintf(stderr, "Warning: Binary header read is not supported yet.\n");

  // create storage path
  // (data storage)/(uuid)-(chunk id).h5\0
  size_t total_file_size = 1;
  size_t chunk_path_len =
      strlen(bucket->object_store) + strlen(object_metadata->id) +
      snprintf(NULL, 0, "%d.%s", chunk_id, suffix) + 3;
  char *chunk_path = malloc(sizeof(char) * chunk_path_len);
  snprintf(chunk_path, chunk_path_len, "%s/%s-%d.%s", bucket->object_store,
           object_metadata->id, chunk_id, suffix);

  switch (object_metadata->datatype) {
    case INT:
      total_file_size = sizeof(int);
      break;
    case DOUBLE:
      total_file_size = sizeof(double);
      break;
    case FLOAT:
      total_file_size = sizeof(float);
      break;
    default:
      fprintf(stderr, "Error: Unknown datatype!\n");
      break;
  }

  for (size_t i = 0; i < object_metadata->n_dims; i++)
    total_file_size *= object_metadata->chunk_dims[i];
  size_t system_file_size;

  read_binary_file(chunk_path, data, &system_file_size);
  if (object_metadata->backend_format != VTK)
    assert(system_file_size == total_file_size);
  return 0;
}
