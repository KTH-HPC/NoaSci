#ifndef __BINARY_BACKEND_H__
#define __BINARY_BACKEND_H__

#include "noa.h"

int put_object_chunk_binary(const container *bucket,
                            const NoaMetadata *object_metadata,
                            const char *suffix, const void *data,
                            const size_t offset, const char *header);
int get_object_chunk_binary(const container *bucket,
                            const NoaMetadata *object_metadata,
                            const char *suffix, void **data, char **header);

#endif
