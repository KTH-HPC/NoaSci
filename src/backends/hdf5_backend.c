#include "private/hdf5_backend.h"

#ifdef USE_MERO
#include "noa_motr.h"
#include <hdf5_hl.h>
#endif

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void create_hdf5_vds(const container* bucket,
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
      assert(vds >= 0);
      break;
    }
    case DOUBLE: {
      vds = H5Dcreate2(vds_file, "vds", H5T_NATIVE_DOUBLE, vds_dataspace,
                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
      assert(vds >= 0);
      break;
    }
    case INT: {
      vds = H5Dcreate2(vds_file, "vds", H5T_NATIVE_INT, vds_dataspace,
                       H5P_DEFAULT, dcpl, H5P_DEFAULT);
      assert(vds >= 0);
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
#ifdef USE_MERO
  hid_t fapl;
#endif

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
    assert(status >= 0);
    status = H5Aclose(header_attribute);
    assert(status >= 0);

    status = H5Sclose(header_attribute_space);
    assert(status >= 0);
    status = H5Tclose(header_attribute_type);
    assert(status >= 0);
  }

  // clean up attributes
  status = H5Dclose(dataset);
  assert(status >= 0);
  status = H5Sclose(dataspace);
  assert(status >= 0);

  // close object on backend
  if (object_metadata->backend == MERO) {
#ifdef USE_MERO
    H5Pclose(fapl);
    H5Fflush(file, H5F_SCOPE_GLOBAL);
    uint64_t high_id = 0;
    int rc = 0;
    void *buffer = NULL;
    MPI_Request mpi_req;

    size_t imgSize = H5Fget_file_image(file, NULL, 0);
    // TODO USE Allreduce to get the max in case the attribute makes a difference?
    if (bucket->mpi_rank == 0) {
      rc = motr_create_object_metadata(object_metadata->id, &high_id, imgSize,
                                      sizeof(*buffer));
      if (rc) {
        fprintf(stderr, "PUT: Failed to create metadata!\n");
        return rc;
      }
    }
    MPI_Ibcast(&high_id, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD, &mpi_req);

    buffer = malloc(imgSize);
    H5Fget_file_image(file, buffer, imgSize);
    MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);

    rc = motr_create_object(high_id, bucket->mpi_rank);
    if (rc) {
      motr_delete_object(high_id, bucket->mpi_rank);
      fprintf(stderr, "PUT: Failed to create data object!\n");
      return rc;
    }

    rc = motr_write_object(high_id, bucket->mpi_rank, (char *)buffer, imgSize);
    if (rc) {
      motr_delete_object(high_id, bucket->mpi_rank);
      fprintf(stderr, "PUT: Failed to write data!\n");
      return rc;
    }

    free(buffer);
#else
    fprintf(stderr, "Error: Mero not supported!\n");
#endif
  }

  status = H5Fclose(file);
  assert(status >= 0);
  free(chunk_path);
  return 0;
}

int get_object_chunk_hdf5(const container *bucket,
                          const NoaMetadata *object_metadata, void **data,
                          char **header, int chunk_id) {
  hid_t status, file, dataset;
  int rc = 0;
#ifdef USE_MERO
  uint64_t high_id;
  uint64_t num_and_size[2];
  MPI_Request high_id_req, num_and_size_req;
#endif

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
        if (bucket->mpi_rank == 0) {
                rc = motr_get_object_metadata(object_metadata->id, &high_id, &num_and_size[0], &num_and_size[1]);
                if (rc) { fprintf(stderr, "GET: Failed to get metadata!\n"); return rc; }
        }
        MPI_Ibcast(num_and_size, 2, MPI_UINT64_T, 0, MPI_COMM_WORLD, &num_and_size_req);
        MPI_Ibcast(&high_id, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD, &high_id_req);

        MPI_Wait(&num_and_size_req, MPI_STATUS_IGNORE);
        size_t total_buffer_size = num_and_size[0] * num_and_size[1];
        void *buffer = malloc(total_size);
        if (buffer == NULL) { fprintf(stderr, "GET: Memory alloc failed!\n"); return -1; }

        MPI_Wait(&high_id_req, MPI_STATUS_IGNORE);
        rc = motr_read_object(high_id, chunk_id, buffer, total_buffer_size);
        if (rc) { free(buffer); fprintf(stderr, "GET: Fail to get object data!\n"); }

        file = H5LTopen_file_image(buffer, total_buffer_size, H5LT_FILE_IMAGE_DONT_COPY | H5LT_FILE_IMAGE_OPEN_RW/* | H5LT_FILE_IMAGE_DONT_RELEASE*/);
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
      assert(status >= 0);
      break;
    case FLOAT:
      *data = (float*)malloc(sizeof(float) * total_size);
      status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       (float*)*data);
      assert(status >= 0);
      break;
    case INT:
      *data = (int*)malloc(sizeof(int) * total_size);
      status = H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       (int*)*data);
      assert(status >= 0);
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
    assert(status >= 0);
  }

  H5Dclose(dataset);
  H5Fclose(file);
  free(chunk_path);
  return rc;
}
