#ifndef DATA_AUGMENTATION_H
#define DATA_AUGMENTATION_H

#include <vector>

class DataAugmentation {
public:
    static void add_random_noise(std::vector<std::vector<double>>& data, double noise_level);
    static void scale(std::vector<std::vector<double>>& data, double scale_factor);
    static void shift(std::vector<std::vector<double>>& data, double shift_value);
    static void rotate(std::vector<std::vector<double>>& data, double angle_degrees);
    static void crop(std::vector<std::vector<double>>& data, double crop_factor);
};

#endif // DATA_AUGMENTATION_H
