#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern "C" void _adex_taas_reg(void);
extern "C" void _fhn_taas_reg(void);
extern "C" void _hh_taas_reg(void);
extern "C" void _izh_taas_reg(void);
extern "C" void _lif_taas_reg(void);
extern "C" void _ml_taas_reg(void);

extern "C" void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"adex_taas.mod\"");
    fprintf(stderr, " \"fhn_taas.mod\"");
    fprintf(stderr, " \"hh_taas.mod\"");
    fprintf(stderr, " \"izh_taas.mod\"");
    fprintf(stderr, " \"lif_taas.mod\"");
    fprintf(stderr, " \"ml_taas.mod\"");
    fprintf(stderr, "\n");
  }
  _adex_taas_reg();
  _fhn_taas_reg();
  _hh_taas_reg();
  _izh_taas_reg();
  _lif_taas_reg();
  _ml_taas_reg();
}
