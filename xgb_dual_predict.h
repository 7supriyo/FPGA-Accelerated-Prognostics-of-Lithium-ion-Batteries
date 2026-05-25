#ifndef XGB_DUAL_PREDICT_H
#define XGB_DUAL_PREDICT_H

#include "xgb_hls_config.h"

void xgb_dual_predict(const float features_in[XGB_NUM_FEATURES], float *rul_out, float *soh_out);

#endif
