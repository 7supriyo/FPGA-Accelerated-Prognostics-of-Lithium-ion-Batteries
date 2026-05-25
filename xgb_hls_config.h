#ifndef XGB_HLS_CONFIG_H
#define XGB_HLS_CONFIG_H

#include <ap_fixed.h>
#include <ap_int.h>

#define XGB_NUM_FEATURES 9
#define XGB_LEAF_FLAG 255
#define XGB_MAX_TRAVERSAL_DEPTH 16
#define XGB_COMPARE_FRAC_BITS 20
#define XGB_SCORE_FRAC_BITS 12
#define XGB_COMPARE_SCALE 1048576
#define XGB_SCORE_SCALE 4096

typedef ap_fixed<22, 2, AP_RND, AP_SAT> compare_t;
typedef ap_fixed<24, 12, AP_RND, AP_SAT> score_t;
typedef ap_int<22> compare_raw_t;
typedef ap_int<24> score_raw_t;
typedef ap_int<32> acc_raw_t;

static const float XGB_SCORE_SCALE_INV = 1.0f / 4096.0f;

struct XGBNode {
    ap_uint<8>  feature_idx;
    compare_raw_t threshold_raw;
    score_raw_t leaf_raw;
    ap_uint<16> left_child;
    ap_uint<16> right_child;
    ap_uint<16> missing_child;
};

void xgb_dual_predict(const float features_in[XGB_NUM_FEATURES], float *rul_out, float *soh_out);

#endif
