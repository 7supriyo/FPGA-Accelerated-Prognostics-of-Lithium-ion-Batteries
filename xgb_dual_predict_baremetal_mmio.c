#include <stdio.h>
#include <string.h>

#include "xil_io.h"
#include "xparameters.h"
#include "sleep.h"

/*
 * Update this macro if your Vitis-generated xparameters.h uses a different
 * instance name for the HLS IP.
 */
#ifndef XGB_DUAL_PREDICT_BASEADDR
#define XGB_DUAL_PREDICT_BASEADDR XPAR_XGB_DUAL_PREDICT_0_BASEADDR
#endif

#define XGB_AP_CTRL          0x00
#define XGB_RUL_OUT_DATA     0x10
#define XGB_SOH_OUT_DATA     0x20
#define XGB_FEATURES_BASE    0x40

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

static void xgb_write_feature(int index, float value) {
    Xil_Out32(XGB_DUAL_PREDICT_BASEADDR + XGB_FEATURES_BASE + (index * 4), float_to_u32(value));
}

static void xgb_start(void) {
    Xil_Out32(XGB_DUAL_PREDICT_BASEADDR + XGB_AP_CTRL, 0x01);
}

static int xgb_is_done(void) {
    u32 ctrl = Xil_In32(XGB_DUAL_PREDICT_BASEADDR + XGB_AP_CTRL);
    return (ctrl >> 1) & 0x1;
}

int main(void) {
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

    printf("Writing 9 input features...\n");
    for (int i = 0; i < 9; ++i) {
        xgb_write_feature(i, features[i]);
    }

    printf("Starting inference...\n");
    xgb_start();

    while (!xgb_is_done()) {
        usleep(100);
    }

    {
        u32 rul_raw = Xil_In32(XGB_DUAL_PREDICT_BASEADDR + XGB_RUL_OUT_DATA);
        u32 soh_raw = Xil_In32(XGB_DUAL_PREDICT_BASEADDR + XGB_SOH_OUT_DATA);
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
