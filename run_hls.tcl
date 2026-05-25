set project_name "battery_xgb_hls"
set solution_name "sol1"

if {[info exists ::env(HLS_PART)]} {
    set part_name $::env(HLS_PART)
} else {
    set part_name "xc7z020clg400-1"
}

if {[info exists ::env(HLS_CLOCK_PERIOD)]} {
    set clock_period $::env(HLS_CLOCK_PERIOD)
} else {
    set clock_period "10.0"
}

open_project -reset $project_name
set_top xgb_dual_predict

add_files src/xgb_hls_config.h
add_files src/xgb_dual_predict.h
add_files src/xgb_dual_predict.cpp
add_files src/xgb_rul_fixed_weights.h
add_files src/xgb_soh_fixed_weights.h

add_files -tb tb/xgb_dual_predict_tb.cpp -cflags "-std=c++14"
add_files -tb tb/X_test_9feat.csv
add_files -tb tb/y_rul_pred_fp32.csv
add_files -tb tb/y_soh_pred_fp32.csv

open_solution -reset $solution_name
set_part $part_name
create_clock -period $clock_period -name default

csim_design
csynth_design
cosim_design
export_design -format ip_catalog

exit
