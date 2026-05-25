#include <cmath>

#include "xgb_dual_predict.h"
#include "xgb_rul_fixed_weights.h"
#include "xgb_soh_fixed_weights.h"

static compare_raw_t quantize_feature(float value) {
#pragma HLS INLINE
    compare_t fixed_value = compare_t(value);
    return compare_raw_t(fixed_value * XGB_COMPARE_SCALE);
}

template <int N_TREES, int ARRAY_SIZE>
static acc_raw_t predict_forest(
    const XGBNode (&forest)[N_TREES][ARRAY_SIZE],
    const score_raw_t base_score_raw,
    const compare_raw_t features[XGB_NUM_FEATURES],
    const bool feature_is_nan[XGB_NUM_FEATURES]
) {
#pragma HLS INLINE off

    acc_raw_t sum = acc_raw_t(base_score_raw);

TREE_LOOP:
    for (int tree = 0; tree < N_TREES; ++tree) {
        ap_uint<16> nid = 0;

NODE_LOOP:
        for (int depth = 0; depth < XGB_MAX_TRAVERSAL_DEPTH; ++depth) {
#pragma HLS PIPELINE II=1
            XGBNode node = forest[tree][nid];
            if (node.feature_idx == XGB_LEAF_FLAG) {
                sum += acc_raw_t(node.leaf_raw);
                break;
            }

            if (feature_is_nan[node.feature_idx]) {
                nid = node.missing_child;
            } else if (features[node.feature_idx] < node.threshold_raw) {
                nid = node.left_child;
            } else {
                nid = node.right_child;
            }
        }
    }

    return sum;
}

void xgb_dual_predict(const float features_in[XGB_NUM_FEATURES], float *rul_out, float *soh_out) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE s_axilite port=features_in bundle=CTRL
#pragma HLS INTERFACE s_axilite port=rul_out bundle=CTRL
#pragma HLS INTERFACE s_axilite port=soh_out bundle=CTRL

    compare_raw_t features[XGB_NUM_FEATURES];
    bool feature_is_nan[XGB_NUM_FEATURES];
#pragma HLS ARRAY_PARTITION variable=features complete dim=1
#pragma HLS ARRAY_PARTITION variable=feature_is_nan complete dim=1

LOAD_FEATURES:
    for (int i = 0; i < XGB_NUM_FEATURES; ++i) {
#pragma HLS UNROLL
        bool is_nan = std::isnan(features_in[i]);
        feature_is_nan[i] = is_nan;
        features[i] = is_nan ? compare_raw_t(0) : quantize_feature(features_in[i]);
    }

    acc_raw_t rul_raw = predict_forest(rul_forest, BASE_SCORE_RUL_RAW, features, feature_is_nan);
    acc_raw_t soh_raw = predict_forest(soh_forest, BASE_SCORE_SOH_RAW, features, feature_is_nan);

    *rul_out = float(rul_raw) * XGB_SCORE_SCALE_INV;
    *soh_out = float(soh_raw) * XGB_SCORE_SCALE_INV;
}
