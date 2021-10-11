#include <stdio.h>

#include "noa.h"

int noa_init() {
#ifdef USE_MERO
#else
  printf("Init libnoa.\n");
#endif
}

int noa_finalize() {
#ifdef USE_MERO
#else
  printf("Close down libnoa.\n");
#endif
}
