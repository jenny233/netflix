#ifndef __SVD_with_bias_H__
#define __SVD_with_bias_H__

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#define READ_IN_LINES 94362235
#define TRAIN_SIZE  94362233
// #define TRAIN_SIZE 500000
#define VALID_SIZE  1965045
#define TEST_SIZE  2749898
#define USER_SIZE  458293
#define MOVIE_SIZE  17770
#define TRAINING_DATA_AVERAGE 3.60861
using namespace std;
using namespace std::chrono;

struct svd_ans {
    double** U;
    double** V;
    double E_in;
    double E_val;
} ;
svd_ans complete_training(double eta, double reg);
#endif
