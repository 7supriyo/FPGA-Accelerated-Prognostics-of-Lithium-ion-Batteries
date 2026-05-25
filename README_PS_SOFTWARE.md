## PS-Side Software Bring-Up

These examples show how to control the implemented `xgb_dual_predict` IP from the Zynq PS:

1. write `9` float input features into AXI-Lite registers
2. start the IP
3. poll `ap_done`
4. read `rul_out` and `soh_out`

The examples match the generated HLS register map:

- `0x00`: `ap_start`, `ap_done`, `ap_idle`, `ap_ready`
- `0x10`: `rul_out`
- `0x20`: `soh_out`
- `0x40` to `0x60`: `features_in[0..8]`

### Files

- `xgb_dual_predict_baremetal_driver.c`
  Uses the generated HLS driver API from `xxgb_dual_predict.h`.
- `xgb_dual_predict_baremetal_mmio.c`
  Uses direct `Xil_Out32` / `Xil_In32` accesses to AXI-Lite registers.

### Known-Good Test Vector

These examples use the first row from:
[X_test_9feat.csv](C:/Users/supri/OneDrive/Documents/BMS_ML_FINAL/Final%20ML/hls_paper/vitis_hls_project/tb/X_test_9feat.csv)

Feature values:

```text
[0] 0.00024390243925154209
[1] 0.68916070461273193
[2] 0.5
[3] 0.27780848741531372
[4] 0.79837840795516968
[5] 0.5361601710319519
[6] 0.12190533429384232
[7] 0.015617376193404198
[8] 0.0032573288772255182
```

Expected software reference outputs:

- RUL about `557.810242`
- SOH about `91.259720`

These reference values come from:

- [y_rul_pred_fp32.csv](C:/Users/supri/OneDrive/Documents/BMS_ML_FINAL/Final%20ML/hls_paper/vitis_hls_project/tb/y_rul_pred_fp32.csv)
- [y_soh_pred_fp32.csv](C:/Users/supri/OneDrive/Documents/BMS_ML_FINAL/Final%20ML/hls_paper/vitis_hls_project/tb/y_soh_pred_fp32.csv)

### Vitis Setup

1. In Vivado, export hardware including bitstream and create an `.xsa`.
2. In Vitis, create a platform from that `.xsa`.
3. Create a standalone application for the PS.
4. Add one of the example `.c` files to the application.

If you use the driver-based example:

1. Make sure `xparameters.h` contains the `xgb_dual_predict` instance.
2. If Vitis does not automatically import the HLS driver, add these files to the app or BSP:
   `C:\Users\supri\xh\cli_workspace_20260323_222141\battery_xgb_hls_comp\battery_xgb_hls_comp\hls\impl\ip\drivers\xgb_dual_predict_v1_0\src\xxgb_dual_predict.c`
   `C:\Users\supri\xh\cli_workspace_20260323_222141\battery_xgb_hls_comp\battery_xgb_hls_comp\hls\impl\ip\drivers\xgb_dual_predict_v1_0\src\xxgb_dual_predict.h`
   `C:\Users\supri\xh\cli_workspace_20260323_222141\battery_xgb_hls_comp\battery_xgb_hls_comp\hls\impl\ip\drivers\xgb_dual_predict_v1_0\src\xxgb_dual_predict_hw.h`

### Macro Name Note

The exact macro in `xparameters.h` depends on how Vitis names the block design instance.

Common examples are:

- `XPAR_XGB_DUAL_PREDICT_0_BASEADDR`
- `XPAR_DESIGN_1_XGB_DUAL_PREDICT_0_0_BASEADDR`
- `XPAR_XGB_DUAL_PREDICT_0_DEVICE_ID`
- `XPAR_DESIGN_1_XGB_DUAL_PREDICT_0_0_DEVICE_ID`

If your macro name differs, update the single `#define` near the top of the example file.

### Output Printing Note

The examples use `printf` to print floating-point values. If your standalone toolchain does not print floats by default, you can still:

- inspect the raw `0xXXXXXXXX` output words, or
- enable float formatting in the BSP/linker settings

### Driver vs MMIO

Use the driver example first if possible. It is cleaner and uses the generated HLS API.

Use the raw MMIO example if:

- the driver is not imported into Vitis yet, or
- you want the smallest possible example with no extra driver files.
