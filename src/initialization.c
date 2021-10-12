#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "noa.h"

#ifdef USE_MERO
#include "aoi_functions.h"
#define MAX_LINE_LEN 1024

// from https://stackoverflow.com/questions/122616/how-do-i-trim-leading-trailing-whitespace-in-a-standard-way
static void trim(char * s) {
    char * p = s;
    int l = strlen(p);

    while(isspace(p[l - 1])) p[--l] = 0;
    while(* p && isspace(* p)) ++p, --l;

    memmove(s, p, l + 1);
}

static int my_aoi_init(const char *mero_config_filename, size_t block_size, int socket, int tier) {
  char key_string_buffer[MAX_LINE_LEN];
  char value_string_buffer[MAX_LINE_LEN];

  char m0_pool_tier[strlen("0x6f00000000000001:0x23d") + 1];
  char m0_pool_tier_name[14];
  snprintf(m0_pool_tier_name, 14, "M0_POOL_TIER%d", tier);

  char local_endpoint_addr[strlen("172.18.1.40@o2ib:12345:4:1") + 1];
  char local_endpoint_addr_name[21];
  snprintf(local_endpoint_addr_name, 21, "LOCAL_ENDPOINT_ADDR%d", socket);

  char local_proc_fid[strlen("0x7200000000000001:0x1f6") + 1];
  char local_proc_fid_name[16];
  snprintf(local_proc_fid_name, 16, "LOCAL_PROC_FID%d", socket);

  const char *ha_endpoint_addr_name = "HA_ENDPOINT_ADDR";
  char ha_endpoint_addr[strlen("172.18.1.40@o2ib:12345:1:1") + 1];

  const char *profile_fid_name      = "PROFILE_FID";
  char profile_fid[strlen("0x7000000000000001:0x2f1") + 1];

  char line_buffer[1024];
  size_t len;
  FILE *fp = fopen(mero_config_filename, "r");
  size_t ret;

printf("opening %s...\n", mero_config_filename);
  while (fgets(line_buffer, 1024, fp)) {
    trim(line_buffer);
    if (line_buffer[0] == '#' || line_buffer[0] == '\n') continue; // skip comment

    sscanf(line_buffer, "%s = %s", key_string_buffer, value_string_buffer);
    if (strcmp(key_string_buffer, m0_pool_tier_name) == 0) {
      strcpy(m0_pool_tier, value_string_buffer);
printf("%s\n", m0_pool_tier);
    }
    else if (strcmp(key_string_buffer, ha_endpoint_addr_name) == 0) {
      strcpy(ha_endpoint_addr, value_string_buffer);
printf("%s\n", ha_endpoint_addr);
    }
    else if (strcmp(key_string_buffer, local_endpoint_addr_name) == 0) {
      strcpy(local_endpoint_addr, value_string_buffer);
printf("%s\n", local_endpoint_addr);
    }
    else if (strcmp(key_string_buffer, profile_fid_name) == 0) {
      strcpy(profile_fid, value_string_buffer);
printf("%s\n", profile_fid);
    }
    else if (strcmp(key_string_buffer, local_proc_fid_name) == 0) {
      strcpy(local_proc_fid, value_string_buffer);
printf("%s\n", local_proc_fid);
    }
  }
  fclose(fp);

  printf("Mero Local Endpoint Address: %s HA Endpoint Address: %s Profile FID: %s Local Proc FID: %s\n", local_endpoint_addr, ha_endpoint_addr, profile_fid, local_proc_fid);

  int rc = aoi_init(local_endpoint_addr, ha_endpoint_addr, profile_fid,
                    local_proc_fid, block_size, tier);
  return rc;
}
#endif /* end if using mero */

int noa_init(const char *mero_config_filename, size_t block_size, int socket, int tier)
{
  int rc = 0;
#ifdef USE_MERO
  rc = my_aoi_init(mero_config_filename, block_size, socket, tier);
#endif
  return rc;
}

int noa_finalize()
{
#ifdef USE_MERO
  aoi_fini();
#endif
  return 0;
}
