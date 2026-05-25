#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../src/xgb_dual_predict.h"

static std::ifstream open_first_existing(const std::vector<std::string> &candidates) {
    for (const std::string &path : candidates) {
        std::ifstream fh(path.c_str());
        if (fh.good()) {
            return fh;
        }
    }
    return std::ifstream();
}

static std::vector<std::array<float, XGB_NUM_FEATURES>> load_feature_rows() {
    std::ifstream fh = open_first_existing({
        "X_test_9feat.csv",
        "tb/X_test_9feat.csv",
        "../tb/X_test_9feat.csv",
        "../../tb/X_test_9feat.csv",
        "../../../tb/X_test_9feat.csv"
    });

    if (!fh.good()) {
        throw std::runtime_error("Could not open X_test_9feat.csv");
    }

    std::vector<std::array<float, XGB_NUM_FEATURES>> rows;
    std::string line;

    std::getline(fh, line);
    while (std::getline(fh, line)) {
        if (line.empty()) {
            continue;
        }

        std::array<float, XGB_NUM_FEATURES> sample = {};
        std::stringstream ss(line);
        std::string cell;
        int idx = 0;
        while (std::getline(ss, cell, ',') && idx < XGB_NUM_FEATURES) {
            sample[idx++] = std::stof(cell);
        }
        if (idx == XGB_NUM_FEATURES) {
            rows.push_back(sample);
        }
    }
    return rows;
}

static std::vector<float> load_scalar_csv(const char *filename) {
    std::ifstream fh = open_first_existing({
        filename,
        std::string("tb/") + filename,
        std::string("../tb/") + filename,
        std::string("../../tb/") + filename,
        std::string("../../../tb/") + filename
    });

    if (!fh.good()) {
        throw std::runtime_error(std::string("Could not open ") + filename);
    }

    std::vector<float> values;
    std::string line;
    while (std::getline(fh, line)) {
        if (line.empty()) {
            continue;
        }
        values.push_back(std::stof(line));
    }
    return values;
}

int main() {
    const float rul_tol = 0.050000f;
    const float soh_tol = 0.020000f;

    auto features = load_feature_rows();
    auto rul_ref = load_scalar_csv("y_rul_pred_fp32.csv");
    auto soh_ref = load_scalar_csv("y_soh_pred_fp32.csv");

    if (features.size() != rul_ref.size() || features.size() != soh_ref.size()) {
        std::cerr << "Test vector size mismatch" << std::endl;
        return 1;
    }

    float max_rul_err = 0.0f;
    float max_soh_err = 0.0f;
    double sum_rul_err = 0.0;
    double sum_soh_err = 0.0;

    for (std::size_t i = 0; i < features.size(); ++i) {
        float rul_hw = 0.0f;
        float soh_hw = 0.0f;
        xgb_dual_predict(features[i].data(), &rul_hw, &soh_hw);

        float rul_err = std::fabs(rul_hw - rul_ref[i]);
        float soh_err = std::fabs(soh_hw - soh_ref[i]);
        if (rul_err > max_rul_err) {
            max_rul_err = rul_err;
        }
        if (soh_err > max_soh_err) {
            max_soh_err = soh_err;
        }

        sum_rul_err += rul_err;
        sum_soh_err += soh_err;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Samples        : " << features.size() << std::endl;
    std::cout << "Max RUL error  : " << max_rul_err << std::endl;
    std::cout << "Mean RUL error : " << (sum_rul_err / features.size()) << std::endl;
    std::cout << "Max SOH error  : " << max_soh_err << std::endl;
    std::cout << "Mean SOH error : " << (sum_soh_err / features.size()) << std::endl;

    if (max_rul_err > rul_tol || max_soh_err > soh_tol) {
        std::cerr << "Tolerance check failed" << std::endl;
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
