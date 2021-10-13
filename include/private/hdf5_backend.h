#ifndef __HDF5_BACKEND_H__
#define __HDF5_BACKEND_H__

#include "noa.h"

void create_hdf5_vds(const container* bucket,
                     const NoaMetadata* object_metadata);

int put_object_chunk_hdf5(const container *bucket,
                          const NoaMetadata *object_metadata, const void *data,
                          const size_t offset, const char *header);
int get_object_chunk_hdf5(const container *bucket,
                          const NoaMetadata *object_metadata, void **data,
                          char **header, int chunk_id);

#endif
