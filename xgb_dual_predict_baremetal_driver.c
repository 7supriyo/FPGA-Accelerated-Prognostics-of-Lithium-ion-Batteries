#include <stdio.h>
#include <string.h>

#include "xparameters.h"
#include "sleep.h"
#include "xxgb_dual_predict.h"

/*
 * Update this macro if your Vitis-generated xparameters.h uses a different
 * instance name for the HLS IP.
 */
#ifndef XGB_DUAL_PREDICT_DEVICE_ID
#define XGB_DUAL_PREDICT_DEVICE_ID XPAR_XGB_DUAL_PREDICT_0_DEVICE_ID
#endif

static u32 float_to_u32(float value) {
    u32 bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float u32_to_float(u32 bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

int main(void) {
    int status;
    XXgb_dual_predict ip;
    word_type feature_words[9];

    static const float features[9] = {
        0.00024390244f,
        0.68916070f,
        0.50000000f,
        0.27780849f,
        0.79837841f,
        0.53616017f,
        0.12190533f,
        0.01561738f,
        0.00325733f
    };

    for (int i = 0; i < 9; ++i) {
        feature_words[i] = float_to_u32(features[i]);
    }

    printf("Initializing xgb_dual_predict...\n");
    status = XXgb_dual_predict_Initialize(&ip, XGB_DUAL_PREDICT_DEVICE_ID);
    if (status != XST_SUCCESS) {
        printf("ERROR: XXgb_dual_predict_Initialize failed: %d\n", status);
        return status;
    }

    XXgb_dual_predict_DisableAutoRestart(&ip);

    if (XXgb_dual_predict_Write_features_in_Words(&ip, 0, feature_words, 9) != 9) {
        printf("ERROR: failed to write all 9 input features\n");
        return XST_FAILURE;
    }

    printf("Starting inference...\n");
    XXgb_dual_predict_Start(&ip);

    while (!XXgb_dual_predict_IsDone(&ip)) {
        usleep(100);
    }

    {
        u32 rul_raw = XXgb_dual_predict_Get_rul_out(&ip);
        u32 soh_raw = XXgb_dual_predict_Get_soh_out(&ip);
        float rul = u32_to_float(rul_raw);
        float soh = u32_to_float(soh_raw);

        printf("Inference complete.\n");
        printf("rul_out raw = 0x%08lx\n", (unsigned long)rul_raw);
        printf("soh_out raw = 0x%08lx\n", (unsigned long)soh_raw);
        printf("RUL = %.6f\n", rul);
        printf("SOH = %.6f\n", soh);
        printf("Expected approx: RUL = 557.810242, SOH = 91.259720\n");
    }

    return 0;
}
