//
// Created by liqinbin on 10/14/20.
// ThunderGBM common.cpp: https://github.com/Xtra-Computing/thundergbm/blob/master/src/thundergbm/util/common.cpp
// Under Apache-2.0 license
// copyright (c) 2020 jiashuai
//

#include "FedTree/common.h"
INITIALIZE_EASYLOGGINGPP

//#define EPSILON (std::numeric_limits<float_type>::epsilon())


std::ostream &operator<<(std::ostream &os, const int_float &rhs) {
    os << string_format("%d/%f", thrust::get<0>(rhs), thrust::get<1>(rhs));
    return os;
}

#define EPSILON 1e-9

bool ft_eq(float_type a, float_type b) {
    return ((a - b) < EPSILON) && ((b - a) < EPSILON);
}

bool ft_ge(float_type a, float_type b) {
    return a > b - EPSILON;
}

bool ft_le(float_type a, float_type b) {
    return a < b + EPSILON;
}