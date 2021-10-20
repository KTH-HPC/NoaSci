#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>

#include "motr/client.h"
//#include "motr/idx.h"
#include "motr/client_internal.h"
#include "conf/confc.h"       /* m0_confc_open_sync */
#include "conf/dir.h"         /* m0_conf_dir_len */
#include "conf/helpers.h"     /* m0_confc_root_open */
#include "conf/diter.h"       /* m0_conf_diter_next_sync */
#include "conf/obj_ops.h"     /* M0_CONF_DIREND */
#include "spiel/spiel.h"      /* m0_spiel_process_lib_load */
#include "reqh/reqh.h"        /* m0_reqh */
#include "lib/types.h"        /* uint32_t */

#define MAX_M0_BUFSZ (128*1024*1024) /* max bs for object store I/O  */
#define METADATA_BLOCK_SIZE 4096

#include "noa_motr.h"

#include <openssl/sha.h>
#include "json.h"

#include "motr/lib/trace.h"

static struct m0_client *clovis_instance = NULL;
static struct m0_container clovis_container;
static struct m0_realm clovis_uber_realm;
static struct m0_config clovis_conf;
static struct m0_idx_dix_config dix_conf;

static unsigned int tier_selection = 0;

extern struct m0_addb_ctx m0_addb_ctx;

static struct m0_fid TIER1 = (struct m0_fid) {  // Tier 2 is SSD Pool
	.f_container = 0x6f00000000000001,
	.f_key = 0x3f8
};

static struct m0_fid TIER2 = (struct m0_fid) {  // Tier 2 is SSD Pool
	.f_container = 0x6f00000000000001,
	.f_key = 0x3f8
};

static struct m0_fid TIER3 = (struct m0_fid) {  // Tier 3 is HDD Pool
	.f_container = 0x6f00000000000001,
	.f_key = 0x426
};

static struct m0_fid *const TIERS[] = {
	NULL, // Tier 0 and 1 are 
	&TIER1, // the same as the default one
	&TIER2,
	&TIER3
};

static const size_t N_TIER = sizeof(TIERS) / sizeof(TIERS[0]);

uint64_t hex_to_uint64(char const *str)
{
	uint64_t accumulator = 0;
	for (size_t i = 0 ; isxdigit((unsigned char)str[i]) ; ++i) {
		char c = str[i];
		accumulator *= 16;
		if (isdigit(c)) { /* '0' .. '9'*/
			accumulator += c - '0';
		}
		else if (isupper(c)) {/* 'A' .. 'F'*/
			accumulator += c - 'A' + 10;
		}
		else {/* 'a' .. 'f'*/
			accumulator += c - 'a' + 10;
		}
	}

	// patch overflow
	accumulator = accumulator & 0xffffffff;

	return accumulator;
}

static void sha256(char *digest, const char *str, size_t length)
{
	unsigned char hash[SHA256_DIGEST_LENGTH];
	
	SHA256_CTX sha256;
	SHA256_Init(&sha256);
	SHA256_Update(&sha256, str, length);
	SHA256_Final(hash, &sha256);
	
	for(int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
		sprintf(digest + (i * 2), "%02x", hash[i]);
	}
	
	digest[SHA256_DIGEST_LENGTH * 2] = 0;
}

static uint64_t roundup_power2(uint64_t x)
{
	uint64_t power = 1;

	while (power < x)
		power *= 2;

	return power;
}

int motr_init(char *laddr, char *ha_addr, char *prof_id, char *proc_fid,
              size_t block_size, unsigned int tier)
{
	int rc;

	dix_conf = (struct m0_idx_dix_config) {
		.kc_create_meta = false
	};

	clovis_conf = (struct m0_config) {
		.mc_local_addr = laddr,
		.mc_ha_addr = ha_addr,
		.mc_profile = prof_id,
		.mc_process_fid = proc_fid,
		.mc_is_oostore = true,
		.mc_is_read_verify = false,
		.mc_max_rpc_msg_size = M0_RPC_DEF_MAX_RPC_MSG_SIZE,
		.mc_tm_recv_queue_min_len = M0_NET_TM_RECV_QUEUE_DEF_LEN,
		.mc_idx_service_id = M0_IDX_DIX,
		.mc_idx_service_conf = &dix_conf
	};

	if (tier >= N_TIER) {
		return EPERM;
	}
	//tier_selection = tier;
	// TODO not hardcode
	m0_fid_sscanf("0x6f00000000000001:0x237", TIERS[1]);
	m0_fid_sscanf("0x6f00000000000001:0x244", TIERS[2]);
	m0_fid_sscanf("0x6f00000000000001:0x251", TIERS[3]);

	rc = m0_client_init(&clovis_instance, &clovis_conf, true);

	if (rc)
		return rc;

	m0_container_init(&clovis_container, NULL, &M0_UBER_REALM,
				 clovis_instance);

	rc = clovis_container.co_realm.re_entity.en_sm.sm_rc;

	if (rc != 0)
		goto init_error1;

	clovis_uber_realm = clovis_container.co_realm;
	M0_LOG(M0_DEBUG, "re_instance=%p", clovis_uber_realm.re_instance);
	return 0;

init_error1:
	m0_client_fini(clovis_instance, true);
	return rc;
}

void motr_fini()
{
	m0_client_fini(clovis_instance, true);
}

static int
open_entity(struct m0_entity *entity)
{
	int             rc = 0;
	struct m0_op *ops[1] = { NULL };

	m0_entity_open(entity, &ops[0]);
	m0_op_launch(ops, 1);
	m0_op_wait(ops[0],
		M0_BITS(M0_OS_FAILED, M0_OS_STABLE),
		M0_TIME_NEVER);
	// this return code is not the state machine return code
	rc = m0_rc(ops[0]);
	m0_op_fini(ops[0]);
	m0_op_free(ops[0]);
	return rc;
}

static int
delete_entity(struct m0_entity *entity)
{
	int             rc = 0;
	struct m0_op *ops[1] = { NULL };

	m0_entity_delete(entity, &ops[0]);
	m0_op_launch(ops, 1);
	rc = m0_op_wait(ops[0], M0_BITS(M0_OS_FAILED,
					       M0_OS_STABLE), M0_TIME_NEVER);
	m0_op_fini(ops[0]);
	m0_op_free(ops[0]);
	return rc;
}

static int
write_data_to_object(struct m0_uint128 id, struct m0_indexvec *ext,
		     struct m0_bufvec *data, struct m0_bufvec *attr)
{
	int             rc = 0;
	struct m0_obj obj;
	struct m0_op *ops[1] = { NULL };

	memset(&obj, 0, sizeof(struct m0_obj));
	m0_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_client_layout_id(clovis_instance));
	open_entity(&obj.ob_entity);
	m0_obj_op(&obj, M0_OC_WRITE, ext, data, attr, 0, 0, &ops[0]);
	m0_op_launch(ops, 1);
	rc = m0_op_wait(ops[0], M0_BITS(M0_OS_FAILED,
					       M0_OS_STABLE), M0_TIME_NEVER);

	if (ops[0]->op_sm.sm_state != M0_OS_STABLE
	    || ops[0]->op_sm.sm_rc != 0)
		rc = EPERM;

	m0_op_fini(ops[0]);
	m0_op_free(ops[0]);

	m0_entity_fini(&obj.ob_entity);
	return rc;
}

static int
read_data_from_object(struct m0_uint128 id, struct m0_indexvec *ext,
		      struct m0_bufvec *data, struct m0_bufvec *attr)
{
	int             rc = 0;
	struct m0_obj obj;
	struct m0_op *ops[1] = { NULL };

	memset(&obj, 0, sizeof(struct m0_obj));

	m0_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_client_layout_id(clovis_instance));

	rc = open_entity(&obj.ob_entity);
	if (rc < 0)		// object not found
		return rc;

	m0_obj_op(&obj, M0_OC_READ, ext, data, attr, 0, 0, &ops[0]);

	if (ops[0] == NULL || ops[0]->op_sm.sm_rc != 0) {
		rc = EPERM;
		goto read_exit1;
	}

	m0_op_launch(ops, 1);

	rc = m0_op_wait(ops[0],
			       M0_BITS(M0_OS_FAILED, M0_OS_STABLE),
			       M0_TIME_NEVER);

	if (ops[0]->op_sm.sm_state != M0_OS_STABLE
	    || ops[0]->op_sm.sm_rc != 0)
		rc = -EPERM;

	m0_op_fini(ops[0]);
	m0_op_free(ops[0]);
read_exit1:
	m0_entity_fini(&obj.ob_entity);
	return rc;
}

uint64_t c0appz_m0bs(uint64_t idhi, uint64_t idlo, uint64_t obj_sz)
{
	int                     rc;
	int                     k;
	unsigned long           usz; /* unit size */
	unsigned long           gsz; /* data units in parity group */
	uint64_t                max_bs;
	unsigned                lid;
	struct m0_reqh         *reqh = &clovis_instance->m0c_reqh;
	struct m0_pool_version *pver;
	struct m0_pdclust_attr *pa;
	struct m0_fid          *pool = TIERS[tier_selection];
	struct m0_obj           obj;
	struct m0_uint128       id = {idhi, idlo};

	if (obj_sz > MAX_M0_BUFSZ)
		obj_sz = MAX_M0_BUFSZ;

	rc = m0_pool_version_get(reqh->rh_pools, pool, &pver);
	if (rc != 0) {
		//ERR("m0_pool_version_get(pool=%s) failed: rc=%d\n",
		//    fid2str(pool), rc);
		return 0;
	}

	//m0_obj_init(&obj, &clovis_uber_realm, &id, M0_DEFAULT_LAYOUT_ID);
	memset(&obj, 0, sizeof(struct m0_obj));
	m0_obj_init(&obj, &clovis_uber_realm, &id, m0_client_layout_id(clovis_instance));
	rc = open_entity(&obj.ob_entity);

	if (rc)
		lid = obj.ob_attr.oa_layout_id;
	//else if (4096) /* set explicitly via -u option ? */
	//	lid = m0_client_layout_id(clovis_instance);
	else
		lid = m0_layout_find_by_buffsize(&reqh->rh_ldom, &pver->pv_id,
						 obj_sz);

	usz = m0_obj_layout_id_to_unit_size(lid);
	pa = &pver->pv_attr;
	gsz = usz * pa->pa_N;
	/*
	 * The buffer should be max 4-times pool-width deep counting by
	 * 1MB units, otherwise we may get -E2BIG from Motr BE subsystem
	 * when there are too many units in one fop and the transaction
	 * grow too big.
	 *
	 * Also, the bigger is unit size - the bigger transaction it may
	 * require. LNet maximum frame size is 1MB, so, for example, 2MB
	 * unit will be sent by two LNet frames and will make bigger BE
	 * transaction.
	 */
	k = 8 / (usz / 0x80000 ?: 1);
	max_bs = usz * k * pa->pa_P * pa->pa_N / (pa->pa_N + 2 * pa->pa_K);
	max_bs = max_bs / gsz * gsz; /* make it multiple of group size */

	//DBG("usz=%lu (N,K,P)=(%u,%u,%u) max_bs=%lu\n",
	//    usz, pa->pa_N, pa->pa_K, pa->pa_P, max_bs);
#ifdef DEBUG
	fprintf(stderr, "usz=%ld max_bs=%ld gsz=%ld pow(obj_sz, 2)=%ld\n", usz, max_bs, gsz, roundup_power2(obj_sz));
#endif
	if (obj_sz >= max_bs)
		return max_bs;
	else if (obj_sz <= gsz)
		return gsz;
	else
		return roundup_power2(obj_sz);
}


int motr_create_object_metadata(const char *object_name, uint64_t *high_id, const size_t num, const size_t size)
{
	int             rc = 0;
	char sha256_digest[SHA256_DIGEST_LENGTH * 2 + 1];
	char serialized_metadata[METADATA_BLOCK_SIZE];

	// Get High ID
	sha256(sha256_digest, object_name, strlen(object_name));
   	*high_id = hex_to_uint64(sha256_digest);

	// Prepare metadata JSON
	json_object *root = json_object_new_object();
	json_object_object_add(root, "name", json_object_new_string(object_name));
	json_object_object_add(root, "sha256", json_object_new_string(sha256_digest));
	json_object_object_add(root, "mero_id", json_object_new_uint64(*high_id));
	json_object_object_add(root, "count", json_object_new_uint64(num));
	json_object_object_add(root, "datatype", json_object_new_uint64(size));

	const char *json_string = json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY);
#ifdef DEBUG
	fprintf(stderr, "tring to create: %s\n\n", json_string);
#endif
	if (strlen(json_string) < METADATA_BLOCK_SIZE) {
		strncpy(serialized_metadata, json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY), METADATA_BLOCK_SIZE-1);
	}
	else {
		fprintf(stderr, "Failed to create metadata, too long!\n");
		return -1;
	}

	// Overwrite if exist
	if (motr_exist_object(*high_id, _METADATA_CHUNK_ID)) {
		fprintf(stderr, "motr_create_object_metadata(): Object already exists, overwrite!\n");
	}
	else {
		// Create object with one block
		rc = motr_create_object(*high_id, _METADATA_CHUNK_ID, METADATA_BLOCK_SIZE);
		if (rc) { fprintf(stderr, "Failed to create metadata\n"); motr_delete_object(*high_id, _METADATA_CHUNK_ID); return rc; }
	}

	// Write metadata to object
	rc = motr_write_object(*high_id, _METADATA_CHUNK_ID, serialized_metadata, METADATA_BLOCK_SIZE, METADATA_BLOCK_SIZE);
	if (rc) { fprintf(stderr, "Failed to write metadata\n"); motr_delete_object(*high_id, _METADATA_CHUNK_ID); return rc; }

#ifdef DEBUG
	fprintf(stderr, "metadata creation successful.\n\n");
#endif
	json_object_put(root);
	return rc;
}

int motr_get_object_metadata(const char *object_name, uint64_t *high_id, size_t *num, size_t *size)
{
	int             rc = 0;
	char sha256_digest[SHA256_DIGEST_LENGTH * 2 + 1];
	char serialized_metadata[METADATA_BLOCK_SIZE];
	json_object *root, *temp;

	// Get High ID
	sha256(sha256_digest, object_name, strlen(object_name));
   	*high_id = hex_to_uint64(sha256_digest);

	if (!motr_exist_object(*high_id, _METADATA_CHUNK_ID)) {
		fprintf(stderr, "motr_get_object_metadata() with high id %"PRIu64": Object does not exist or corrupted!\n", *high_id);
		return -1;
	}

	rc = motr_read_object(*high_id, _METADATA_CHUNK_ID, serialized_metadata, METADATA_BLOCK_SIZE, METADATA_BLOCK_SIZE);
	if (rc) { fprintf(stderr, "Failed to read metadata\n"); return rc; }

	root = json_tokener_parse(serialized_metadata);
	if (!root) { fprintf(stderr, "Parsing metadata failed!\n"); return -1; }

#ifdef DEBUG
	const char *json_string = json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY);
	fprintf(stderr, "Trying to get: %s\n\n", json_string);
#endif

	temp = json_object_object_get(root, "count");
	*num = json_object_get_uint64(temp);
	temp = json_object_object_get(root, "datatype");
	*size = json_object_get_uint64(temp);

	// release object
	json_object_put(root);

	return rc;
}

int
motr_create_object(uint64_t high_id, uint64_t low_id, size_t bsz)
{
	int                     rc;
	unsigned                lid;
	struct m0_reqh         *reqh = &clovis_instance->m0c_reqh;
	struct m0_pool_version *pver;
	struct m0_op    *op = NULL;
	struct m0_fid          *pool = TIERS[tier_selection];
	struct m0_uint128       id = {high_id, low_id};
	struct m0_obj    obj = {};

	//DBG("tier=%d pool=%s\n", tier, fid2str(pool));
	rc = m0_pool_version_get(reqh->rh_pools, pool, &pver);
	if (rc != 0) {
		fprintf(stderr, "Error: m0_pool_version_get(pool=%d) failed: rc=%d\n", tier_selection, rc);
		//ERR("m0_pool_version_get(pool=%s) failed: rc=%d\n",
		//    fid2str(pool), rc);
		return rc;
	}

	lid = m0_layout_find_by_buffsize(&reqh->rh_ldom, &pver->pv_id,
					 bsz);

	//DBG("pool="FID_F" width=%u parity=(%u+%u) unit=%dK m0bs=%luK\n",
	//    FID_P(&pver->pv_pool->po_id), pver->pv_attr.pa_P,
	//    pver->pv_attr.pa_N, pver->pv_attr.pa_K,
	//    m0_obj_layout_id_to_unit_size(lid) / 1024, bsz / 1024);

	memset(&obj, 0, sizeof(struct m0_obj));
	m0_obj_init(&obj, &clovis_uber_realm, &id, lid);

	rc = open_entity(&obj.ob_entity);
	if (rc >= 0)		// object already exists
		return 1;

	m0_entity_create(pool, &obj.ob_entity, &op);
	//m0_op_launch(&op, ARRAY_SIZE(op));
	m0_op_launch(&op, 1);

	rc = m0_op_wait(op,
		       M0_BITS(M0_OS_FAILED, M0_OS_STABLE),
		       M0_TIME_NEVER);
	m0_op_fini(op);
	m0_op_free(op);
	m0_entity_fini(&obj.ob_entity);

	return rc;
}

int
motr_delete_object(uint64_t high_id, uint64_t low_id)
{
	int             rc = 0;
	struct m0_obj obj;

	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	memset(&obj, 0, sizeof(struct m0_obj));

	m0_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_client_layout_id(clovis_instance));

	rc = open_entity(&obj.ob_entity);
	if (rc < 0) // object not found
		return rc;

	rc = delete_entity(&obj.ob_entity);

	m0_entity_fini(&obj.ob_entity);

	return rc;
}

/*
 * read the data contained in the object with the id specified and
 * write it to buffer.
 * it returns 0 if the operation was correct
 */
int
motr_read_object(uint64_t high_id, uint64_t low_id, char *buffer, size_t length, size_t clovis_block_size)
{
	int             rc = 0;

	int             n_full_blocks = length / clovis_block_size;
	int             n_blocks = n_full_blocks + (length % clovis_block_size != 0);
	int             byte_count = 0;

#ifdef DEBUG
	fprintf(stderr, "motr_read_object(): MAX_BLOCK_CNT_PER_OP:%d\n", MAX_BLOCK_CNT_PER_OP);
	fprintf(stderr, "motr_read_object(): read block size:     %ld\n", clovis_block_size);
	fprintf(stderr, "motr_read_object(): read n_blocks:       %d\n", n_blocks);
	fprintf(stderr, "motr_read_object(): read n_full_blocks:  %d\n", n_full_blocks);
#endif

	struct m0_indexvec ext;
	struct m0_bufvec data;
	struct m0_bufvec attr;
	int             i;

	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	long cnt = 0;
	int block_count = (n_full_blocks - cnt) > MAX_BLOCK_CNT_PER_OP ? MAX_BLOCK_CNT_PER_OP : (n_full_blocks - cnt);
	if (block_count == 0) block_count = 1;

	rc = m0_indexvec_alloc(&ext, block_count);
	if (rc) return rc;

	rc = m0_bufvec_alloc(&data, block_count, clovis_block_size);
	if (rc) { m0_indexvec_free(&ext); return rc; };

	rc = m0_bufvec_alloc(&attr, block_count, 1);
	if (rc) { m0_bufvec_free(&data); m0_indexvec_free(&ext); return rc; }

	while (cnt < n_full_blocks) {
		if (block_count != MAX_BLOCK_CNT_PER_OP) {
			m0_bufvec_free(&attr);
			m0_bufvec_free(&data);
			m0_indexvec_free(&ext);

			rc = m0_indexvec_alloc(&ext, block_count);
			if (rc) return rc;
			rc = m0_bufvec_alloc(&data, block_count, clovis_block_size);
			if (rc) { m0_indexvec_free(&ext); return rc; };
			rc = m0_bufvec_alloc(&attr, block_count, 1);
			if (rc) { m0_bufvec_free(&data); m0_indexvec_free(&ext); return rc; }
		}

		for (i = 0; i < block_count; i++) {
			ext.iv_index[i] = byte_count;
			ext.iv_vec.v_count[i] = clovis_block_size;
			byte_count += clovis_block_size;
			attr.ov_vec.v_count[i] = 0;
		}
	
		rc = read_data_from_object(id, &ext, &data, &attr);
		if (rc) { fprintf(stderr, "Read operation failed!\n"); m0_bufvec_free(&attr); m0_bufvec_free(&data); m0_indexvec_free(&ext); return rc; }
	
		for (i = 0; i < block_count; i++) {
			memcpy(buffer + ext.iv_index[i],
				   data.ov_buf[i],
				   clovis_block_size);
		}

		cnt += block_count;
		block_count = n_full_blocks - cnt;
		if (block_count > MAX_BLOCK_CNT_PER_OP)
			block_count = MAX_BLOCK_CNT_PER_OP;
	}

	m0_bufvec_free(&attr);
	m0_bufvec_free(&data);
	m0_indexvec_free(&ext);

	if (n_blocks > n_full_blocks) {
		block_count = 1;
		rc = m0_indexvec_alloc(&ext, block_count);
		if (rc) return rc;

		rc = m0_bufvec_alloc(&data, block_count, clovis_block_size);
		if (rc) { m0_indexvec_free(&ext); return rc; }

		rc = m0_bufvec_alloc(&attr, block_count, 1);
		if (rc) { m0_indexvec_free(&ext); m0_bufvec_free(&data); return rc; }

		ext.iv_index[0] = byte_count;
		ext.iv_vec.v_count[0] = clovis_block_size;
		byte_count += clovis_block_size;
		attr.ov_vec.v_count[0] = 0;
	
		rc = read_data_from_object(id, &ext, &data, &attr);
		if (rc) { fprintf(stderr, "Read operation failed!\n"); m0_bufvec_free(&attr); m0_bufvec_free(&data); m0_indexvec_free(&ext); return rc; }
	
		memcpy(buffer + n_full_blocks * clovis_block_size,
		       data.ov_buf[0],
		        length % clovis_block_size);

		m0_bufvec_free(&attr);
		m0_bufvec_free(&data);
		m0_indexvec_free(&ext);
	}

	return rc;
}

/*
 * write data from buffer to object specified by the id.
 * It returns 0 if the operation was correct.
 */
int
motr_write_object(uint64_t high_id, uint64_t low_id, char *buffer, size_t length, size_t clovis_block_size)
{
	int             rc = 0;
	int             n_full_blocks = length / clovis_block_size;
	int             n_blocks = n_full_blocks + (length % clovis_block_size != 0 ? 1 : 0);
	int             byte_count = 0;
	struct m0_indexvec ext;
	struct m0_bufvec data;
	struct m0_bufvec attr;
	int             i;

#ifdef DEBUG
	fprintf(stderr, "motr_write_object(): MAX_BLOCK_CNT_PER_OP:%d\n", MAX_BLOCK_CNT_PER_OP);
	fprintf(stderr, "motr_write_object(): write block size:    %ld\n", clovis_block_size);
	fprintf(stderr, "motr_write_object(): write n_blocks:      %d\n", n_blocks);
	fprintf(stderr, "motr_write_object(): write n_full_blocks: %d\n", n_full_blocks);
#endif

	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	long cnt = 0;
	int block_count = (n_full_blocks - cnt) > MAX_BLOCK_CNT_PER_OP ? MAX_BLOCK_CNT_PER_OP : (n_full_blocks - cnt);
	if (block_count == 0) block_count = 1;
#ifdef DEBUG
	fprintf(stderr, "!!!! (%d - %ld) > %d : %ld -> %d\n", n_full_blocks, cnt, MAX_BLOCK_CNT_PER_OP, (n_full_blocks - cnt), block_count);
	fprintf(stderr, "to allocate block count: %d\n", block_count);
#endif
	rc = m0_indexvec_alloc(&ext, block_count);
	if (rc) return rc;

	rc = m0_bufvec_alloc(&data, block_count, clovis_block_size);
	if (rc) { m0_indexvec_free(&ext); return rc; };

	rc = m0_bufvec_alloc(&attr, block_count, 1);
	if (rc) { m0_bufvec_free(&data); m0_indexvec_free(&ext); return rc; }

	while (cnt < n_full_blocks) {
		if (block_count != MAX_BLOCK_CNT_PER_OP) {
			m0_bufvec_free(&attr);
			m0_bufvec_free(&data);
			m0_indexvec_free(&ext);

			rc = m0_indexvec_alloc(&ext, block_count);
			if (rc) return rc;
			rc = m0_bufvec_alloc(&data, block_count, clovis_block_size);
			if (rc) { m0_indexvec_free(&ext); return rc; };
			rc = m0_bufvec_alloc(&attr, block_count, 1);
			if (rc) { m0_bufvec_free(&data); m0_indexvec_free(&ext); return rc; }
		}

		for (i = 0; i < block_count; i++) {
			ext.iv_index[i] = byte_count;
			ext.iv_vec.v_count[i] = clovis_block_size;
			byte_count += clovis_block_size;
			attr.ov_vec.v_count[i] = 0;
			memcpy(data.ov_buf[i],
				   buffer + cnt * clovis_block_size + i * clovis_block_size,
				   clovis_block_size);
		}
	
		rc = write_data_to_object(id, &ext, &data, &attr);
		if (rc) { fprintf(stderr, "Write operation failed!\n"); m0_bufvec_free(&attr); m0_bufvec_free(&data); m0_indexvec_free(&ext); return rc; }

		cnt += block_count;
		block_count = n_full_blocks - cnt;
		if (block_count > MAX_BLOCK_CNT_PER_OP)
			block_count = MAX_BLOCK_CNT_PER_OP;
	}

	m0_bufvec_free(&attr);
	m0_bufvec_free(&data);
	m0_indexvec_free(&ext);

	if (n_blocks > n_full_blocks) {
		block_count = 1;
		rc = m0_indexvec_alloc(&ext, block_count);
		if (rc) return rc;

		rc = m0_bufvec_alloc(&data, block_count, clovis_block_size);
		if (rc) { m0_indexvec_free(&ext); return rc; };

		rc = m0_bufvec_alloc(&attr, block_count, 1);
		if (rc) { m0_bufvec_free(&data); m0_indexvec_free(&ext); return rc; }

		ext.iv_index[0] = byte_count;
		ext.iv_vec.v_count[0] = clovis_block_size;
		attr.ov_vec.v_count[0] = 0;
		memcpy(data.ov_buf[0],
		       buffer + n_full_blocks * clovis_block_size,
		       length % clovis_block_size);
		rc = write_data_to_object(id, &ext, &data, &attr);
		if (rc) fprintf(stderr, "writing to object failed!\n");

		m0_bufvec_free(&attr);
		m0_bufvec_free(&data);
		m0_indexvec_free(&ext);
#ifdef DEBUG
		fprintf(stderr, "written last block\n");
#endif
	}

#ifdef DEBUG
	printf("Object creation successful, now try to retrieve it...\n\n");
        void *verify_data = malloc(length);
        rc = motr_read_object(high_id, low_id, verify_data, length, clovis_block_size);
	char path_buf[1024]; snprintf(path_buf, 1024, "%ld_%ld.bin", high_id, low_id);
	FILE *fp = fopen(path_buf, "wb");
	fwrite(verify_data, sizeof(void), length, fp);
	fclose(fp);
	free(verify_data);
	printf("Object retrieved at %ld_%ld.bin ..\n\n", high_id, low_id);
#endif
	return rc;
}

int
motr_exist_object(uint64_t high_id, uint64_t low_id)
{
	int             rc;
	struct m0_obj obj;
	struct m0_uint128 id = {
		.u_hi = high_id,
		.u_lo = low_id
	};

	memset(&obj, 0, sizeof(struct m0_obj));

	m0_obj_init(&obj, &clovis_uber_realm, &id,
			   m0_client_layout_id(clovis_instance));
	rc = open_entity(&obj.ob_entity);

	m0_entity_fini(&obj.ob_entity);
	return rc >= 0;
}

int motr_put(const char *object_name, const char *data, const size_t num, const size_t size)
{
	int rc;
	uint64_t high_id;

	rc = motr_create_object_metadata(object_name, &high_id, num, size);
	if (rc) { fprintf(stderr, "PUT: Failed to create metadata!\n"); return rc; }

	// Overwrite if exist
	if (motr_exist_object(high_id, _DATA_CHUNK_ID)) {
		fprintf(stderr, "PUT: Object already exists, overwrite!\n");
	}
	else {
		rc = motr_create_object(high_id, _DATA_CHUNK_ID, 4096 /*TODO*/);
		if (rc) { motr_delete_object(high_id, _METADATA_CHUNK_ID); fprintf(stderr, "PUT: Failed to create data object!\n"); return rc; }
	}

	rc = motr_write_object(high_id, _DATA_CHUNK_ID, (char*)data, num * size, METADATA_BLOCK_SIZE);
	if (rc) { motr_delete_object(high_id, _METADATA_CHUNK_ID); motr_delete_object(high_id, _DATA_CHUNK_ID); fprintf(stderr, "PUT: Failed to write data!\n"); return rc; }

	return rc;
}

int motr_get(const char *object_name, char **data, size_t *num, size_t *size)
{
	int rc;
	uint64_t high_id;

	rc = motr_get_object_metadata(object_name, &high_id, num, size);
	if (rc) { fprintf(stderr, "GET: Failed to get object metadata!\n"); return rc; }

	*data = malloc((*size) * (*num));
	if (data == NULL) { fprintf(stderr, "GET: Memory alloc failed!\n"); return -1; }

	rc = motr_read_object(high_id, _DATA_CHUNK_ID, *data, (*size) * (*num), METADATA_BLOCK_SIZE);
	if (rc) { free(data); fprintf(stderr, "GET: Fail to get object data!\n"); }

	return rc;
}

int motr_delete(const char *object_name)
{
	int             rc = 0;
	uint64_t high_id;
	char sha256_digest[SHA256_DIGEST_LENGTH * 2 + 1];

	// Get High ID
	sha256(sha256_digest, object_name, strlen(object_name));
   	high_id = hex_to_uint64(sha256_digest);

	rc = motr_delete_object(high_id, _METADATA_CHUNK_ID);
	if (rc) { fprintf(stderr, "DELETE: Failed to delete object metadata!\n"); }

	rc = motr_delete_object(high_id, _DATA_CHUNK_ID);
	if (rc) { fprintf(stderr, "DELETE: Failed to delete object data!\n"); }

	return rc;
}
