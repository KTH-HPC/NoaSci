#include "private/hdf5_backend.h"

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int put_object_chunk_hdf5(const container *bucket,
                          const NoaMetadata *object_metadata, const void *data,
                          const size_t offset, const char *header) {
  // create storage path
  // (data storage)/(uuid)-(chunk id).h5\0
  size_t chunk_path_len = strlen(bucket->object_store) +
                          strlen(object_metadata->id) +
                          snprintf(NULL, 0, "%d", bucket->mpi_rank) + 6;
  char *chunk_path = malloc(sizeof(char) * chunk_path_len);
  snprintf(chunk_path, chunk_path_len, "%s/%s-%d.h5", bucket->object_store,
           object_metadata->id, bucket->mpi_rank);

  hsize_t dims[object_metadata->n_dims];
  hid_t file, dataspace, dataset;
  herr_t status;

  // create file/object on backend
  switch (object_metadata->backend) {
    case POSIX:
      file = H5Fcreate(chunk_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      break;
    case MERO:
#ifdef USE_MERO
      hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
      status = H5Pset_fapl_core(fapl, /* memory increment size: 4M */ 1 << 22,
                                /*backing_store*/ false);
      assert(status >= 0 && "H5Pset_fapl_core failed");
      file = H5Fcreate(chunk_path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
#else
      fprintf(stderr, "Error: Mero not supported!\n");
#endif
      break;
    default:
      fprintf(stderr, "Error: Unknown backend!\n");
      exit(1);
  }

  // write data
  for (size_t i = 0; i < object_metadata->n_dims; i++)
    dims[i] = object_metadata->chunk_dims[i];
  dataspace = H5Screate_simple(object_metadata->n_dims, dims, NULL);

  switch (object_metadata->datatype) {
    case DOUBLE: {
      double *d = (double *)data;
      dataset = H5Dcreate2(file, "/data", H5T_NATIVE_DOUBLE, dataspace,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, &d[offset]);
    } break;
    case INT: {
      int *d = (int *)data;
      dataset = H5Dcreate2(file, "/data", H5T_NATIVE_INT, dataspace,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                        &d[offset]);
    } break;
    case FLOAT: {
      float *d = (float *)data;
      dataset = H5Dcreate2(file, "/data", H5T_NATIVE_FLOAT, dataspace,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, &d[offset]);
    } break;
    default:
      fprintf(stderr, "Datatype undefined\n");
      /* handle cleanup */
      exit(1);
      break;
  }

  // write header as attribute if provided
  if (header != NULL) {
    hid_t header_attribute_type = H5Tcopy(H5T_C_S1);
    assert(header_attribute_type >= 0);

    hid_t header_attribute_space = H5Screate(H5S_SCALAR);
    assert(header_attribute_space >= 0);

    // status = H5Tset_size(header_attribute_type, H5T_VARIABLE);
    status = H5Tset_size(header_attribute_type, strlen(header) + 1);
    assert(status == 0);

    status = H5Tset_cset(header_attribute_type, H5T_CSET_ASCII);
    assert(status == 0);

    hid_t header_attribute =
        H5Acreate(dataset, "header", header_attribute_type,
                  header_attribute_space, H5P_DEFAULT, H5P_DEFAULT);
    assert(header_attribute >= 0);
    status = H5Awrite(header_attribute, header_attribute_type, header);
    status = H5Aclose(header_attribute);

    status = H5Sclose(header_attribute_space);
    status = H5Tclose(header_attribute_type);
  }

  // clean up attributes
  status = H5Dclose(dataset);
  status = H5Sclose(dataspace);

  // close object on backend
  if (object_metadata->backend == MERO) {
#ifdef USE_MERO
    H5Pclose(fapl);
    H5Fflush(file, H5F_SCOPE_GLOBAL);
    int world_rank;
    int rc;
    uint64_t high_id;

    ssize_t imgSize = H5Fget_file_image(file, NULL, 0);
    void *buffer = malloc(imgSize);
    H5Fget_file_image(file, buffer, imgSize);
    //_object_binary_write(uuid_filename, buffer, imgSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) {
      rc = aoi_create_object_metadata(uuid_filename, &high_id, imgSize,
                                      sizeof(*buffer));
      if (rc) {
        fprintf(stderr, "PUT: Failed to create metadata!\n");
        return rc;
      }
    }
    MPI_Bcast(&high_id, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    rc = aoi_create_object(high_id, part_id);
    if (rc) {
      aoi_delete_object(high_id, part_id);
      fprintf(stderr, "PUT: Failed to create data object!\n");
      return rc;
    }

    rc = aoi_write_object(high_id, part_id, (char *)buffer, imgSize);
    if (rc) {
      aoi_delete_object(high_id, part_id);
      fprintf(stderr, "PUT: Failed to write data!\n");
      return rc;
    }

    free(buffer);
#else
    fprintf(stderr, "Error: Mero not supported!\n");
#endif
  }

  status = H5Fclose(file);
  free(chunk_path);
  return 0;
}

int get_object_chunk_hdf5(const container *bucket,
                          const NoaMetadata *object_metadata, void **data,
                          char **header) {
  hid_t status, file, dataset;

  size_t chunk_path_len = strlen(bucket->object_store) +
                          strlen(object_metadata->id) +
                          snprintf(NULL, 0, "%d", bucket->mpi_rank) + 7;
  char *chunk_path = malloc(sizeof(char) * chunk_path_len);
  snprintf(chunk_path, chunk_path_len, "%s/%s-%d.h5", bucket->object_store,
           object_metadata->id, bucket->mpi_rank);

  size_t total_size = 1;
#pragma omp parallel for
  for (int i = 0; i < object_metadata->n_dims; i++) {
    total_size = total_size * object_metadata->chunk_dims[i];
  }

  switch (object_metadata->backend) {
    case POSIX:
      file = H5Fopen(chunk_path, H5F_ACC_RDONLY, H5P_DEFAULT);
      break;
    case MERO:
#ifdef USE_MERO
#else
      fprintf(stderr, "Error: Mero is not supported!\n");
#endif
      break;
    default:
      fprintf(stderr, "Error: unknown backend\n");
  }

  dataset = H5Dopen2(file, "/data", H5P_DEFAULT);
  switch (object_metadata->datatype) {
    case DOUBLE:
      *data = malloc(sizeof(double) * total_size);
      status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                       H5P_DEFAULT, *data);
      break;
    case FLOAT:
      *data = malloc(sizeof(float) * total_size);
      status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       *data);
      break;
    case INT:
      *data = malloc(sizeof(int) * total_size);
      status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       *data);
      break;
    default:
      fprintf(stderr, "Datatype undefined\n");
      /* handle cleanup */
      return -1;
  }

  htri_t header_exist = H5Aexists(dataset, "header");
  if (header_exist >= 0) {
    hid_t header_attribute = H5Aopen(dataset, "header", H5P_DEFAULT);
    hsize_t header_size = H5Aget_storage_size(header_attribute);
    hid_t header_attribute_type = H5Aget_type(header_attribute);
    *header = (char *)malloc(header_size);
    status =
        H5Aread(header_attribute, header_attribute_type, (void *)(*header));
  }

  H5Dclose(dataset);
  H5Fclose(file);
  free(chunk_path);
  return 0;
}
