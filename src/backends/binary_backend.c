#include "private/binary_backend.h"
#include "storage/noa_motr.h"

#include <errno.h>
#include <fcntl.h>
#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <mpi.h>

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
  size_t chunk_path_len = 0;
  char *chunk_path = NULL;
#ifdef USE_MERO
  uint64_t high_id;
  int rc = 0;
  size_t buffer_with_header_size = 0;
  void *buffer_with_header = NULL;
  MPI_Request high_id_req;
#endif

  // compute total size with data type
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

  switch (object_metadata->backend) {
    case BINARY:
    case VTK:
      // compute path length and allocate memory
      chunk_path_len =
          strlen(bucket->object_store) + strlen(object_metadata->id) +
          snprintf(NULL, 0, "%d.%s", bucket->mpi_rank, suffix) + 3;
      chunk_path = malloc(sizeof(char) * chunk_path_len);

      // generate path
      snprintf(chunk_path, chunk_path_len, "%s/%s-%d.%s", bucket->object_store,
               object_metadata->id, bucket->mpi_rank, suffix);

      // write binary file and free path buffer
      write_binary_file(chunk_path, &data[offset], total_file_size, header);
      free(chunk_path);
      break;
    case MERO:
#ifdef USE_MERO
      if (header != NULL) {
        // if header is provided, we need to adjust the size
        buffer_with_header_size = sizeof(char)*strlen(header) + sizeof(void)*total_file_size;
      }
      else {
        // if header is not provided, we keep what we have and reuse the buffer
        buffer_with_header_size = total_file_size;
        buffer_with_header = data;
      }

      // rank zero creates metadata object and generate high ID
      if (bucket->mpi_rank == 0) {
        rc = motr_create_object_metadata(object_metadata->id, &high_id, buffer_with_header_size, sizeof(void));
        if (rc) {
          fprintf(stderr, "PUT: Failed to create metadata!\n");
          return rc;
        }
      }

      // broadcast high ID
      MPI_Ibcast(&high_id, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD, &high_id_req);

      if (header != NULL) {
        // if we have a header, we need to merge the two buffers
        // TODO in nao_motr to write to the same object with offset
        buffer_with_header = malloc(buffer_with_header_size);
        memcpy(buffer_with_header, header, sizeof(char)*strlen(header));
        memcpy(buffer_with_header + sizeof(char)*strlen(header), data, total_file_size * sizeof(void));
      }

      MPI_Wait(&high_id_req, MPI_STATUS_IGNORE);
      rc = motr_create_object(high_id, bucket->mpi_rank);
      if (rc) {
        motr_delete_object(high_id, bucket->mpi_rank);
        fprintf(stderr, "PUT: Failed to create data object!\n");
        return rc;
      }

      // write object to mero
      rc = motr_write_object(high_id, bucket->mpi_rank, (char *)buffer_with_header, buffer_with_header_size);
      if (rc) {
        motr_delete_object(high_id, bucket->mpi_rank);
        fprintf(stderr, "PUT: Failed to write data!\n");
        return rc;
      }

      if (buffer_with_header != NULL) free(buffer_with_header);
#else
      fprintf(stderr, "Error: Meror not supported!\n");
#endif
      break;
    default:
      fprintf(stderr, "Error: Unknown data backend!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return 0;
}

int get_object_chunk_binary(const container *bucket,
                            const NoaMetadata *object_metadata,
                            const char *suffix, void **data, char **header, int chunk_id) {
  fprintf(stderr, "Warning: Binary header read is not supported yet.\n");
  *header = NULL;

  // create storage path
  // (data storage)/(uuid)-(chunk id).h5\0
  size_t total_file_size = 1;
  size_t chunk_path_len;
  char *chunk_path = NULL;
#ifdef USE_MERO
  size_t num_and_size[2];
  uint64_t high_id;
  MPI_Request high_id_req, num_and_size_req;
  int rc = 0;
#endif

  // compute object size according to data type
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

  size_t system_file_size; // <- for verification

  switch (object_metadata->backend) {
    case BINARY:
    case VTK:
      chunk_path_len =
          strlen(bucket->object_store) + strlen(object_metadata->id) +
          snprintf(NULL, 0, "%d.%s", chunk_id, suffix) + 3;
      chunk_path = malloc(sizeof(char) * chunk_path_len);
      snprintf(chunk_path, chunk_path_len, "%s/%s-%d.%s", bucket->object_store,
           object_metadata->id, chunk_id, suffix);

      read_binary_file(chunk_path, data, &system_file_size);
      if (object_metadata->backend_format != VTK)
        assert(system_file_size == total_file_size);
      free(chunk_path);
      break;
    case MERO:
#ifdef USE_MERO
      if (bucket->mpi_rank == 0) {
        rc = motr_get_object_metadata(object_metadata->id, &high_id, &num_and_size[0], &num_and_size[1]);
        if (rc) { fprintf(stderr, "GET: Failed to get metadata!\n"); return rc; }
      }
      MPI_Ibcast(num_and_size, 2, MPI_UINT64_T, 0, MPI_COMM_WORLD, &num_and_size_req);
      MPI_Ibcast(&high_id, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD, &high_id_req);

      MPI_Wait(&num_and_size_req, MPI_STATUS_IGNORE);
      size_t total_buffer_size = num_and_size[0] * num_and_size[1];
      *data = malloc(total_buffer_size);
      if (*data == NULL) { fprintf(stderr, "GET: Memory alloc failed!\n"); return -1; }

      MPI_Wait(&high_id_req, MPI_STATUS_IGNORE);
      rc = motr_read_object(high_id, chunk_id, *data, total_buffer_size);
      if (rc) { free(*data); fprintf(stderr, "GET: Fail to get object data!\n"); }
#else
      fprintf(stderr, "Error: Mero not supported!\n");
#endif
      break;
    default:
      fprintf(stderr, "Error: Unknown backend!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return 0;
}
