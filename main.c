#include <stdio.h>
#include <string.h>

#include "sleep.h"
#include "xil_io.h"
#include "xil_types.h"
#include "xparameters.h"

/*
 * Standalone PS-side bring-up example for the implemented xgb_dual_predict IP.
 *
 * Flow:
 *   1. Write 9 float features into AXI-Lite registers at 0x40..0x60
 *   2. Start the accelerator by writing ap_start = 1 to 0x00
 *   3. Poll ap_done in the control register
 *   4. Read RUL from 0x10 and SOH from 0x20
 *
 * If your xparameters.h uses a different instance name, only update the base
 * address selection block below.
 */

#if defined(XPAR_XGB_DUAL_PREDICT_0_BASEADDR)
#define XGB_DUAL_PREDICT_BASEADDR XPAR_XGB_DUAL_PREDICT_0_BASEADDR
#elif defined(XPAR_DESIGN_1_XGB_DUAL_PREDICT_0_0_BASEADDR)
#define XGB_DUAL_PREDICT_BASEADDR XPAR_DESIGN_1_XGB_DUAL_PREDICT_0_0_BASEADDR
#else
#error "Could not find the xgb_dual_predict base address macro in xparameters.h"
#endif

#define XGB_AP_CTRL       0x00
#define XGB_RUL_OUT_DATA  0x10
#define XGB_SOH_OUT_DATA  0x20
#define XGB_FEATURE_BASE  0x40

static u32 float_to_u32(float value) {
    u32 bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float u32_to_float(u32 bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void write_feature(int index, float value) {
    Xil_Out32(XGB_DUAL_PREDICT_BASEADDR + XGB_FEATURE_BASE + (index * 4),
              float_to_u32(value));
}

static void start_accelerator(void) {
    Xil_Out32(XGB_DUAL_PREDICT_BASEADDR + XGB_AP_CTRL, 0x01);
}

static int is_done(void) {
    u32 ctrl = Xil_In32(XGB_DUAL_PREDICT_BASEADDR + XGB_AP_CTRL);
    return (int)((ctrl >> 1) & 0x1U);
}

int main(void) {
    /*
     * Known-good test vector from the project notebook.
     * Replace these 9 values with your own normalized features if needed.
     */
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

    printf("xgb_dual_predict PS bring-up\n");
    printf("Base address: 0x%08lx\n", (unsigned long)XGB_DUAL_PREDICT_BASEADDR);

    printf("Writing 9 input features to AXI-Lite registers...\n");
    for (int i = 0; i < 9; ++i) {
        write_feature(i, features[i]);
        printf("  feature[%d] = %.8f\n", i, features[i]);
    }

    printf("Starting accelerator...\n");
    start_accelerator();

    while (!is_done()) {
        usleep(100);
    }

    {
        u32 rul_raw = Xil_In32(XGB_DUAL_PREDICT_BASEADDR + XGB_RUL_OUT_DATA);
        u32 soh_raw = Xil_In32(XGB_DUAL_PREDICT_BASEADDR + XGB_SOH_OUT_DATA);
        float rul = u32_to_float(rul_raw);
        float soh = u32_to_float(soh_raw);

        printf("Accelerator finished.\n");
        printf("rul_out raw = 0x%08lx\n", (unsigned long)rul_raw);
        printf("soh_out raw = 0x%08lx\n", (unsigned long)soh_raw);
        printf("RUL = %.6f\n", rul);
        printf("SOH = %.6f\n", soh);
    }

    return 0;
}
