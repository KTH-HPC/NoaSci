#include <errno.h>
#include <fcntl.h>
#include <hdf5.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <uuid/uuid.h>

#include "noa.h"
#include "private/hdf5_backend.h"
#include "object.pb-c.h"

// aux tool to write a file
static int write_binary_file(const char* filename, void* data, size_t size) {
  int fd, rc;

  /* open file write and sync to disk */
  if ((fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644)) == -1) {
    fprintf(stderr, "Unable to open %s for writing: %s\n", filename,
            strerror(errno));
    return -1;
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
static int read_binary_file(const char* filename, void** data, size_t* size) {
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
  *data = (char*)malloc(*size);
  ret = read(fd, *data, *size);
  if (ret == -1) {
    fprintf(stderr, "Reading from %s failed: %s\n", filename, strerror(errno));
    return -1;
  }

  close(fd);
  return 0;
}

// create metadata protobuf that holds info needed to do I/O operations
NoaMetadata* noa_create_metadata(const container* bucket,
                                 const char* object_name, const TYPE datatype,
                                 const FORMAT backend_format,
                                 const BACKEND backend,  // POSIX or Mero
                                 const int dimensionality, const long* dims,
                                 const long* chunk_dims)

{
  if (bucket->mpi_rank == 0) {
    // check if object exists
    size_t metadata_file_path_len = strlen(bucket->key_value_store) +
                                    strlen(bucket->name) +
                                    strlen(object_name) + 4 + 3;
    char* metadata_file_path =
        (char*)malloc(sizeof(char) * metadata_file_path_len);
    snprintf(metadata_file_path, metadata_file_path_len, "%s/%s-%s.bin",
             bucket->key_value_store, bucket->name, object_name);
    if( access(metadata_file_path, F_OK ) != -1) {
      fprintf(stderr, "Error: object with name %s already exist and is immutable\n", metadata_file_path);
      free(metadata_file_path);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    free(metadata_file_path);
  }

  // check if chunking aligns with proc
  int world_size;
  int num_proc_needed = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  for (int i = 0; i < dimensionality; i++)
    num_proc_needed *= dims[i] / chunk_dims[i];
  if (num_proc_needed != world_size)
    if (bucket->mpi_rank == 0)
      fprintf(stderr,
              "Warning: chunking scheme requires %d processes, the total number "
              "of MPI processes is %d.\n",
              num_proc_needed, world_size);

  // create protobuf that stores metadata
  NoaMetadata* object_metadata = (NoaMetadata*)malloc(sizeof(NoaMetadata));
  noa_metadata__init(object_metadata);

  object_metadata->backend = backend;
  object_metadata->datatype = datatype;
  object_metadata->backend_format = backend_format;

  // Global object dimensionality and shapes
  object_metadata->n_dims = dimensionality;
  object_metadata->dims = (long*)malloc(sizeof(long) * dimensionality);

  // number of chunks, chunk sizes and shapes
  object_metadata->num_chunks = 1;
  object_metadata->n_chunk_dims = dimensionality;
  object_metadata->chunk_dims = (long*)malloc(sizeof(long) * dimensionality);
  for (int i = 0; i < dimensionality; i++) {
    object_metadata->chunk_dims[i] = chunk_dims[i];
    object_metadata->dims[i] = dims[i];
    object_metadata->num_chunks *= dims[i] / chunk_dims[i];
  }

  // prepare space for UUID
  object_metadata->id = (char*)malloc(sizeof(char) * 37);

  // rank 0 creates UUID and broadcast
  if (bucket->mpi_rank == 0) {
    // only rank zero needs object name to put metadata
    object_metadata->name =
        (char*)malloc(sizeof(char) * strlen(object_name) + 1);
    snprintf(object_metadata->name, strlen(object_name) + 1, "%s", object_name);

    // generate UUID
    uuid_t uuid;
    uuid_generate(uuid);
    uuid_unparse(uuid, object_metadata->id);
    uuid_clear(uuid);
  }

  // all processes share the same UUID
  MPI_Bcast(object_metadata->id, 37, MPI_CHAR, 0, MPI_COMM_WORLD);

  return object_metadata;
}

int noa_free_metadata(const container* bucket, NoaMetadata* object_metadata) {
  // only rank 0 has the object name
  if (bucket->mpi_rank == 0) free(object_metadata->name);

  // free repeated fields and protobuf object itself
  free(object_metadata->dims);
  free(object_metadata->chunk_dims);
  free(object_metadata->id);
  free(object_metadata);

  return 0;
}

int noa_put_metadata(const container* bucket,
                     const NoaMetadata* object_metadata) {
  int rc;
  int dir_fd;

  // only rank zero does the put operation
  if (bucket->mpi_rank == 0) {
    // create links between chunks
    switch (object_metadata->backend_format) {
      case BINARY:
      case VTK:
        break;
      case HDF5:
        create_hdf5_vds(bucket, object_metadata);
        break;
      default:
        fprintf(stderr, "Error: Unknown backend\n");
        exit(1);
    }

    // prepare to sealized protobu
    unsigned serialized_metadata_size =
        noa_metadata__get_packed_size(object_metadata);
    void* seralized_metadata_buffer = malloc(serialized_metadata_size);

    // first write data with a tmp path, then use rename to get atomaticity
    size_t tmp_metadata_file_path_len =
        strlen(bucket->key_value_store) + 36 + 4 + 2;
    char* tmp_metadata_file_path =
        (char*)malloc(sizeof(char) * tmp_metadata_file_path_len);
    snprintf(tmp_metadata_file_path, tmp_metadata_file_path_len, "%s/%s.bin",
             bucket->key_value_store, object_metadata->id);

    // prepare final path
    size_t metadata_file_path_len = strlen(bucket->key_value_store) +
                                    strlen(bucket->name) +
                                    strlen(object_metadata->name) + 4 + 3;
    char* metadata_file_path =
        (char*)malloc(sizeof(char) * metadata_file_path_len);
    snprintf(metadata_file_path, metadata_file_path_len, "%s/%s-%s.bin",
             bucket->key_value_store, bucket->name, object_metadata->name);

    // open metadata directory (key-vaue store)
    dir_fd = open(bucket->key_value_store, O_RDONLY);
    if (dir_fd < 0) {
      fprintf(stderr, "Unable to open %s: %s\n", bucket->key_value_store,
              strerror(errno));
      return -1;
    }

    // seralize data and write
    noa_metadata__pack(object_metadata, seralized_metadata_buffer);
    rc = write_binary_file(tmp_metadata_file_path, seralized_metadata_buffer,
                           serialized_metadata_size);

    /* replace old metadata by update name */
    rc = rename(tmp_metadata_file_path, metadata_file_path);
    if (rc < 0) {
      fprintf(stderr, "Replacing file %s fail: %s\n", metadata_file_path,
              strerror(errno));
      return -1;
    }

    /* fsync directory */
    rc = fsync(dir_fd);
    if (rc < 0) {
      fprintf(stderr, "Unable to sync %s: %s\n", bucket->key_value_store,
              strerror(errno));
      return -1;
    }

    rc = close(dir_fd);
    if (rc < 0) {
      fprintf(stderr, "Closing directory %s fail: %s\n",
              bucket->key_value_store, strerror(errno));
      return -1;
    }

    free(seralized_metadata_buffer);
    free(metadata_file_path);
    free(tmp_metadata_file_path);
  }

  return 0;
}

NoaMetadata* noa_get_metadata(const container* bucket,
                              const char* object_name) {
  int rc;

  // prepare metadata protobuf file path
  size_t metadata_file_path_len = strlen(bucket->key_value_store) +
                                  strlen(bucket->name) + strlen(object_name) +
                                  4 + 3;
  char* metadata_file_path =
      (char*)malloc(sizeof(char) * metadata_file_path_len);
  snprintf(metadata_file_path, metadata_file_path_len, "%s/%s-%s.bin",
           bucket->key_value_store, bucket->name, object_name);

  // get a buffer of data
  void* seralized_metadata_buffer;
  size_t serialized_metadata_size;

  rc = read_binary_file(metadata_file_path, &seralized_metadata_buffer,
                        &serialized_metadata_size);
  assert (rc == 0);
  NoaMetadata* object_metadata = noa_metadata__unpack(
      NULL, serialized_metadata_size, seralized_metadata_buffer);

  free(metadata_file_path);
  free(seralized_metadata_buffer);
  return object_metadata;
}
