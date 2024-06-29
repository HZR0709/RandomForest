#include "DataAugmentation.h"
#include <algorithm>
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void DataAugmentation::add_random_noise(std::vector<std::vector<double>>& data, double noise_level) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, noise_level);

    for (auto& sample : data) {
        for (auto& value : sample) {
            value += distribution(generator);
        }
    }
}

void DataAugmentation::scale(std::vector<std::vector<double>>& data, double scale_factor) {
    for (auto& sample : data) {
        for (auto& value : sample) {
            value *= scale_factor;
        }
    }
}

void DataAugmentation::shift(std::vector<std::vector<double>>& data, double shift_value) {
    for (auto& sample : data) {
        for (auto& value : sample) {
            value += shift_value;
        }
    }
}

void DataAugmentation::rotate(std::vector<std::vector<double>>& data, double angle_degrees) {
    double angle_radians = angle_degrees * M_PI / 180.0;
    double cos_angle = std::cos(angle_radians);
    double sin_angle = std::sin(angle_radians);

    for (auto& sample : data) {
        if (sample.size() < 2) {
            continue; // 确保样本有足够的维度进行旋转
        }
        double x = sample[0];
        double y = sample[1];
        sample[0] = x * cos_angle - y * sin_angle;
        sample[1] = x * sin_angle + y * cos_angle;
    }
}


void DataAugmentation::crop(std::vector<std::vector<double>>& data, double crop_factor) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-crop_factor, crop_factor);

    for (auto& sample : data) {
        for (auto& value : sample) {
            value *= (1 + distribution(generator));
        }
    }
}
