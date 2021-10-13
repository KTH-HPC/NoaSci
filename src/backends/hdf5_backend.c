#include "private/hdf5_backend.h"

#ifdef USE_MERO
#include "aoi_functions.h"
#include <hdf5_hl.h>
#endif

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

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
  hid_t file, dataspace, dataset, fapl;
  herr_t status;

  // create file/object on backend
  switch (object_metadata->backend) {
    case MERO:
#ifdef USE_MERO
      fapl = H5Pcreate(H5P_FILE_ACCESS);
      status = H5Pset_fapl_core(fapl, /* memory increment size: 4M */ 1 << 22,
                                /*backing_store*/ false);
      assert(status >= 0 && "H5Pset_fapl_core failed");
      file = H5Fcreate(chunk_path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
      break;
#else
      fprintf(stderr, "Error: Mero backend not supported!\n");
      // continue to POSIX backend instead
#endif
    case POSIX:
      file = H5Fcreate(chunk_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
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
    uint64_t high_id = 0;
    int rc = 0;
    void *buffer = NULL;
    MPI_Request mpi_req;
    MPI_Status mpi_status;

    size_t imgSize = H5Fget_file_image(file, NULL, 0);
    if (bucket->mpi_rank == 0) {
      rc = aoi_create_object_metadata(object_metadata->id, &high_id, imgSize,
                                      sizeof(*buffer));
      if (rc) {
        fprintf(stderr, "PUT: Failed to create metadata!\n");
        return rc;
      }
    }
    MPI_Ibcast(&high_id, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD, &mpi_req);

    buffer = malloc(imgSize);
    H5Fget_file_image(file, buffer, imgSize);
    MPI_Wait(&mpi_req, &mpi_status);

    rc = aoi_create_object(high_id, bucket->mpi_rank);
    if (rc) {
      aoi_delete_object(high_id, bucket->mpi_rank);
      fprintf(stderr, "PUT: Failed to create data object!\n");
      return rc;
    }

    rc = aoi_write_object(high_id, bucket->mpi_rank, (char *)buffer, imgSize);
    if (rc) {
      aoi_delete_object(high_id, bucket->mpi_rank);
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
                          char **header, int chunk_id) {
  hid_t status, file, dataset;
  int rc = 0;
  uint64_t high_id, num, size;

  size_t chunk_path_len = strlen(bucket->object_store) +
                          strlen(object_metadata->id) +
                          snprintf(NULL, 0, "%d", chunk_id) + 7;
  char *chunk_path = malloc(sizeof(char) * chunk_path_len);
  snprintf(chunk_path, chunk_path_len, "%s/%s-%d.h5", bucket->object_store,
           object_metadata->id, chunk_id);

  size_t total_size = 1;
  for (int i = 0; i < object_metadata->n_dims; i++) {
    total_size = total_size * object_metadata->chunk_dims[i];
  }

  switch (object_metadata->backend) {
    case POSIX:
      file = H5Fopen(chunk_path, H5F_ACC_RDONLY, H5P_DEFAULT);
      break;
#ifdef USE_MERO
    case MERO:
        if (bucket->chunk_id == 0) {
                size_t total_length = 0;
                rc = aoi_get_object_metadata(object_metadata->id, &high_id, &num, &size);
                if (rc) { fprintf(stderr, "GET: Failed to get metadata!\n"); return rc; }
        }
        MPI_Bcast(&high_id, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

        void *buffer = malloc(size * num);
        if (buffer == NULL) { fprintf(stderr, "GET: Memory alloc failed!\n"); return -1; }

        rc = aoi_read_object(high_id, bucket->mpi_rank, buffer, size * num);
        if (rc) { free(buffer); fprintf(stderr, "GET: Fail to get object data!\n"); }

        file = H5LTopen_file_image(buffer, size * num, H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_OPEN_RW/* | H5LT_FILE_IMAGE_DONT_RELEASE*/);
      break;
#endif
    default:
      fprintf(stderr, "Error: unknown backend\n");
  }

  dataset = H5Dopen2(file, "/data", H5P_DEFAULT);
  switch (object_metadata->datatype) {
    case DOUBLE:
      *data = (double*)malloc(sizeof(double) * total_size);
      status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                       H5P_DEFAULT, (double*)*data);
      break;
    case FLOAT:
      *data = (float*)malloc(sizeof(float) * total_size);
      status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       (float*)*data);
      break;
    case INT:
      *data = (int*)malloc(sizeof(int) * total_size);
      status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       (int*)*data);
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
    *header = (char*)malloc(header_size);
    status =
        H5Aread(header_attribute, header_attribute_type, (void *)(*header));
  }

  H5Dclose(dataset);
  H5Fclose(file);
  free(chunk_path);
  return rc;
}
