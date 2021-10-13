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

static void create_hdf5_vds(const container* bucket,
                            const NoaMetadata* object_metadata) {
  hid_t vds_file, vds_dataspace, src_dataspace, vds, dcpl;
  herr_t status;
  hsize_t start[object_metadata->n_dims], count[object_metadata->n_dims],
      block[object_metadata->n_dims];
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  /* initialize chunk pointer */
  for (int i = 0; i < object_metadata->n_dims; i++) {
    start[i] = object_metadata->chunk_dims[i];
    count[i] = 1;
    block[i] = object_metadata->chunk_dims[i];
  }

  // get chunk path buffer to generate links later
  // (data storage)/(uuid)-(chunk id).h5\0
  size_t chunk_path_len = strlen(bucket->object_store) +
                          strlen(object_metadata->id) +
                          snprintf(NULL, 0, "%d", world_size) + 6;
  char* chunk_path = malloc(sizeof(char) * chunk_path_len);

  /* create VDS to link chunks, reuse chunk path buffer for VDS path */
  snprintf(chunk_path, chunk_path_len, "%s/%s.h5", bucket->object_store,
           object_metadata->id);
  vds_file = H5Fcreate(chunk_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  vds_dataspace = H5Screate_simple(object_metadata->n_dims,
                                   (hsize_t*)object_metadata->dims, NULL);
  src_dataspace = H5Screate_simple(object_metadata->n_dims, block, NULL);
  dcpl = H5Pcreate(H5P_DATASET_CREATE);

  /* create fill value for un-defined data */
  switch (object_metadata->datatype) {
    case FLOAT: {
      float fill_value = -65535.0;
      status = H5Pset_fill_value(dcpl, H5T_NATIVE_FLOAT, &fill_value);
      assert(status >= 0);
    } break;
    case DOUBLE: {
      double fill_value = -65535.0;
      status = H5Pset_fill_value(dcpl, H5T_NATIVE_DOUBLE, &fill_value);
      assert(status >= 0);
    } break;
    case INT: {
      int fill_value = -65535;
      status = H5Pset_fill_value(dcpl, H5T_NATIVE_INT, &fill_value);
      assert(status >= 0);
    } break;
    default:
      fprintf(stderr, "invalid datatype\n");
  }

  /* group cycle size for different dimensions */
  int group[object_metadata->n_dims];
  for (int i = 0; i < object_metadata->n_dims; i++) {
    if (i == 0)
      group[i] = (int)((double)object_metadata->dims[i] / (double)block[i]);
    else
      group[i] = (int)((double)object_metadata->dims[i] / (double)block[i] *
                       group[i - 1]);
  }

  /* compute starting point of chunk and link to virtual dataspace */
  for (int i = 0; i < object_metadata->num_chunks; i++) {
    for (int j = 0; j < object_metadata->n_dims; j++) {
      int factor = (int)((double)group[j] /
                         ((double)object_metadata->dims[j] / (double)block[j]));
      int rotate = (i + factor) % factor;
      if (rotate == 0) {
        start[j] = (start[j] + block[j] + object_metadata->dims[j]) %
                   object_metadata->dims[j];
      }
    }
    snprintf(chunk_path, chunk_path_len, "%s-%d.h5", object_metadata->id, i);
    status = H5Sselect_hyperslab(vds_dataspace, H5S_SELECT_SET, start, NULL,
                                 count, block);
    assert(status >= 0);
    status =
        H5Pset_virtual(dcpl, vds_dataspace, chunk_path, "data", src_dataspace);
    assert(status >= 0);
  }

  /* link to virtual dataset */
  switch (object_metadata->datatype) {
    case FLOAT: {
      vds = H5Dcreate2(vds_file, "vds", H5T_NATIVE_FLOAT, vds_dataspace,
                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
      break;
    }
    case DOUBLE: {
      vds = H5Dcreate2(vds_file, "vds", H5T_NATIVE_DOUBLE, vds_dataspace,
                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
      break;
    }
    case INT: {
      vds = H5Dcreate2(vds_file, "vds", H5T_NATIVE_INT, vds_dataspace,
                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
      break;
    }
    default:
      fprintf(stderr, "invalid datatype\n");
  }

  /* cleanup vds */
  H5Dclose(vds);
  H5Sclose(src_dataspace);
  H5Sclose(vds_dataspace);
  H5Fclose(vds_file);
  free(chunk_path);
}

// create metadata protobuf that holds info needed to do I/O operations
NoaMetadata* noa_create_metadata(const container* bucket,
                                 const char* object_name, const TYPE datatype,
                                 const FORMAT backend_format,
                                 const BACKEND backend,  // POSIX or Mero
                                 const int dimensionality, const long* dims,
                                 const long* chunk_dims)

{
  // check if chunking aligns with proc
  int world_size;
  int num_proc_needed = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  for (int i = 0; i < dimensionality; i++)
    num_proc_needed *= dims[i] / chunk_dims[i];
  if (num_proc_needed != world_size)
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

    // prepare to sealized protobuf
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
