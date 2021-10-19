#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "noa.h"
#include "private/binary_backend.h"
#include "private/hdf5_backend.h"
#include "storage/noa_motr.h"

int noa_put_chunk_by_id(const container* bucket, const NoaMetadata* object_metadata,
                        const int chunk_id, const void* data, const size_t offset, const char* header) {
  int rc = 0;

  switch (object_metadata->backend_format) {
    case BINARY:
      rc = put_object_chunk_binary_by_id(bucket, object_metadata, chunk_id, "bin", data, offset,
                                         header);
      break;
    case HDF5:
      fprintf(stderr, "Error: put_object_chunk_hdf5_by_id() unimplemented!\n");
      break;
    case VTK:
      // reuse the binary backend for now
      rc = put_object_chunk_binary_by_id(bucket, object_metadata, chunk_id, "vtk", data, offset,
                                         header);
      break;
    default:
      fprintf(stderr, "Error: Unknown backend data format!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return rc;
}

int noa_put_chunk(const container* bucket, const NoaMetadata* object_metadata,
                  const void* data, const size_t offset, const char* header) {
  int rc = 0;

  switch (object_metadata->backend_format) {
    case BINARY:
      rc = put_object_chunk_binary(bucket, object_metadata, "bin", data, offset,
                                   header);
      break;
    case HDF5:
      rc = put_object_chunk_hdf5(bucket, object_metadata, data, offset, header);
      break;
    case VTK:
      // reuse the binary backend for now
      rc = put_object_chunk_binary(bucket, object_metadata, "vtk", data, offset,
                                   header);
      break;
    default:
      fprintf(stderr, "Error: Unknown backend data format!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return rc;
}

int noa_get_chunk(const container* bucket, const NoaMetadata* object_metadata,
                  void** data, char** header, int chunk_id) {
  int rc = 0;
  if (chunk_id < 0) {
    chunk_id = bucket->mpi_rank;
    fprintf(stderr, "Note: Setting chunk ID to MPI rank %d since it is -ve.\n", chunk_id);
  }

  switch (object_metadata->backend_format) {
    case BINARY:
      rc =
          get_object_chunk_binary(bucket, object_metadata, "bin", data, header, chunk_id);
      break;
    case HDF5:
      rc = get_object_chunk_hdf5(bucket, object_metadata, data, header, chunk_id);
      break;
    case VTK:
      // reuse the binary backend for now
      rc =
          get_object_chunk_binary(bucket, object_metadata, "vtk", data, header, chunk_id);
      break;
    default:
      fprintf(stderr, "Error: Unknown backend data format!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return rc;
}

// delete all objects and related metadata
int noa_delete(const container* bucket, NoaMetadata* object_metadata) {
  int rc = 0;
  int dir_fd;

  // important, to make sure everyone has comitted their chunk
  MPI_Barrier(MPI_COMM_WORLD);

  // delete metadata
  if (bucket->mpi_rank == 0) {
    /* current copy of metadata */
    size_t metadata_file_path_len = strlen(bucket->key_value_store) +
                                    strlen(bucket->name) +
                                    strlen(object_metadata->name) + 7;
    char* metadata_file_path =
        (char*)malloc(sizeof(char) * metadata_file_path_len);

    // create tmp file path
    size_t dirty_metadata_file_path_len = metadata_file_path_len + 6;
    char* dirty_metadata_file_path =
        (char*)malloc(sizeof(char) * dirty_metadata_file_path_len);

    snprintf(metadata_file_path, metadata_file_path_len, "%s/%s-%s.bin",
             bucket->key_value_store, bucket->name, object_metadata->name);
    snprintf(dirty_metadata_file_path, dirty_metadata_file_path_len, "%s-dirty",
             metadata_file_path);

    /* open directory */
    dir_fd = open(bucket->key_value_store, O_RDONLY);
    if (dir_fd < 0) {
      fprintf(stderr, "Unable to open %s: %s\n", bucket->key_value_store,
              strerror(errno));
      return -1;
    }

    /* mark old metadata dirty */
    rc = rename(metadata_file_path, dirty_metadata_file_path);
    if (rc < 0) {
      fprintf(stderr, "Replacing file %s fail: %s\n", metadata_file_path,
              strerror(errno));
      return -1;
    }

    /* fsync directory */
    rc = syncfs(dir_fd);
    if (rc < 0) {
      fprintf(stderr, "Unable to sync %s: %s\n", bucket->key_value_store,
              strerror(errno));
      return -1;
    }

    rc = close(dir_fd);
    if (rc < 0) {
      fprintf(stderr, "Closing file %s fail: %s\n", bucket->key_value_store,
              strerror(errno));
      return -1;
    }

    if (object_metadata->backend == MERO) {
      uint64_t high_id;
      size_t num, size;
      rc = motr_get_object_metadata(object_metadata->id, &high_id, &num, &size);
      for (int chunk_id = 0; chunk_id < object_metadata->num_chunks;
           chunk_id++) {
        rc = motr_delete_object(high_id, chunk_id);
        assert(rc == 0);
      }
      rc = motr_delete_object(high_id, _METADATA_CHUNK_ID);
      assert(rc == 0);
    }
    else if (object_metadata->backend == POSIX) {
      // delete actualy data
      size_t chunk_path_len;
      char* chunk_path;
      switch (object_metadata->backend_format) {
        case HDF5:
          // delete data
          chunk_path_len =
              strlen(bucket->object_store) + strlen(object_metadata->id) +
              snprintf(NULL, 0, "-%d.h5", object_metadata->num_chunks) + 3;
          chunk_path = malloc(sizeof(char) * chunk_path_len);
          for (int chunk_id = 0; chunk_id < object_metadata->num_chunks;
               chunk_id++) {
            snprintf(chunk_path, chunk_path_len, "%s/%s-%d.h5",
                     bucket->object_store, object_metadata->id, chunk_id);
            rc = remove(chunk_path);
            if (rc < 0) {
              fprintf(stderr, "Error: Unable to delete chunk %d: %s\n", chunk_id,
                      chunk_path);
              MPI_Abort(MPI_COMM_WORLD, 1);
            }
          }

          // delete vds
          snprintf(chunk_path, chunk_path_len, "%s/%s.h5", bucket->object_store,
                   object_metadata->id);
          rc = remove(chunk_path);
          if (rc < 0) {
            fprintf(stderr, "Error: Unable to delete VDS: %s\n", chunk_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
          }

          free(chunk_path);
          break;
        case BINARY:
          // delete data
          chunk_path_len =
              strlen(bucket->object_store) + strlen(object_metadata->id) +
              snprintf(NULL, 0, "-%d.bin", object_metadata->num_chunks) + 3;
          chunk_path = malloc(sizeof(char) * chunk_path_len);
          for (int chunk_id = 0; chunk_id < object_metadata->num_chunks;
               chunk_id++) {
            snprintf(chunk_path, chunk_path_len, "%s/%s-%d.bin",
                     bucket->object_store, object_metadata->id, chunk_id);
            rc = remove(chunk_path);
            if (rc < 0) {
              fprintf(stderr, "Error: Unable to delete chunk %d: %s\n", chunk_id,
                      chunk_path);
              MPI_Abort(MPI_COMM_WORLD, 1);
            }
          }
          free(chunk_path);
          break;
        case VTK:
          // delete data
          chunk_path_len =
              strlen(bucket->object_store) + strlen(object_metadata->id) +
              snprintf(NULL, 0, "-%d.vtk", object_metadata->num_chunks) + 3;
          chunk_path = malloc(sizeof(char) * chunk_path_len);
          for (int chunk_id = 0; chunk_id < object_metadata->num_chunks;
               chunk_id++) {
            snprintf(chunk_path, chunk_path_len, "%s/%s-%d.vtk",
                     bucket->object_store, object_metadata->id, chunk_id);
            rc = remove(chunk_path);
            if (rc < 0) {
              fprintf(stderr, "Error: Unable to delete chunk %d: %s\n", chunk_id,
                      chunk_path);
              MPI_Abort(MPI_COMM_WORLD, 1);
            }
          }
          free(chunk_path);
          break;
        default:
          fprintf(stderr, "Error: Unknown backend format!\n");
          MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    else {
      fprintf(stderr, "Error: Unknown storage backend!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // finally, remove the marked metadata
    rc = remove(dirty_metadata_file_path);
    if (rc < 0) {
      fprintf(stderr, "Unable to delete metadata file %s: %s\n",
              dirty_metadata_file_path, strerror(errno));
      return -1;
    }

    free(dirty_metadata_file_path);
    free(metadata_file_path);
  }

  // free metadata
  rc = noa_free_metadata(bucket, object_metadata);

  return rc;
}
