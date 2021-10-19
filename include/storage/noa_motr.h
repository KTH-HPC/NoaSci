#ifndef __NOA_MOTR_H__
#define __NOA_MOTR_H__

#include <stddef.h>
#include <stdint.h>

#ifndef MAX_BLOCK_CNT_PER_OP
#define MAX_BLOCK_CNT_PER_OP 200
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define _METADATA_CHUNK_ID 0xffffffffffffffff
#define _DATA_CHUNK_ID 16

typedef unsigned long uint64_t;

int motr_create_object_metadata(const char *object_name, uint64_t *high_id, const size_t num, const size_t size);

int motr_get_object_metadata(const char *object_name, uint64_t *high_id, size_t *num, size_t *size);

int motr_put(const char *object_name, const char *data, const size_t num, const size_t size);

int motr_get(const char *object_name, char **data, size_t *num, size_t *size);

int motr_delete(const char *object_name);

int motr_init(char *laddr, char *ha_addr, char *prof_id, char *proc_fid, size_t block_size, unsigned int tier);

void motr_fini(void);

uint64_t c0appz_m0bs(uint64_t idhi, uint64_t idlo, uint64_t obj_sz);

int motr_create_object(uint64_t high_id, uint64_t low_id, size_t bsz);

int motr_delete_object(uint64_t high_id, uint64_t low_id);

int motr_read_object(uint64_t high_id, uint64_t low_id, char *buffer, size_t length, size_t clovis_block_size);

int motr_write_object(uint64_t high_id, uint64_t low_id, char *buffer, size_t length, size_t clovis_block_size);

int motr_exist_object(uint64_t high_id, uint64_t low_id);

#ifdef __cplusplus
}
#endif

#endif
