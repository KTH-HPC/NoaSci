#ifndef __NOA_H__
#define __NOA_H__

// Public APIs

#ifdef __cplusplus
extern "C" {
#endif

#include "object.pb-c.h"

typedef enum {
  INT = 0,
  FLOAT = 1,
  DOUBLE = 2,
} TYPE;

typedef enum {
  BINARY = 0,
  HDF5 = 1,
  VTK = 2,
} FORMAT;

typedef enum {
  POSIX = 0,
  MERO = 1,
} BACKEND;

// Init NOA
int noa_init(const char *mero_config_filename, size_t block_size, int socket, int tier);
int noa_finalize();

// Containers
typedef struct Container {
  char *name;             // label or metadata file name prefix
  char *key_value_store;  // POSIX path to metadata folder
  char *object_store;     // POSIX path to data folder
  int mpi_rank;           // My rank
#ifdef USE_MERO
  uint64_t storage_tier;  // Mero Tier
#endif
} container;

int noa_container_open(container **bucket, const char *container_name,
                       const char *metadata_path, const char *data_path);
int noa_container_close(container *bucket);

// Object operations for one chunk per process
int noa_put_chunk(const container *bucket, const NoaMetadata *object_metadata,
                  const void *data, const size_t offset, const char *header);
int noa_get_chunk(const container *bucket, const NoaMetadata *object_metadata,
                  void **data, char **header);

int noa_delete(const container *bucket, NoaMetadata *object_metadata);

// Metadata Operations
NoaMetadata *noa_create_metadata(const container *bucket,
                                 const char *object_name, const TYPE datatype,
                                 const FORMAT backend_format,
                                 const BACKEND backend,  // POSIX or Mero
                                 const int dimensionality, const long *dims,
                                 const long *chunk_dims);

int noa_free_metadata(const container *bucket, NoaMetadata *object_metadata);
int noa_put_metadata(const container *bucket,
                     const NoaMetadata *object_metadata);
NoaMetadata *noa_get_metadata(const container *bucket, const char *object_name);

#ifdef __cplusplus
}
#endif

#endif
