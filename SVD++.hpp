#ifndef __SVDPP_H__
#define __SVDPP_H__

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <Eigen/Dense>
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

// Max number of int is 65535
#define TRAIN_SIZE  94362233   // Use long while looping
// #define TRAIN_SIZE 500000
#define VALID_SIZE  1965045    // Use long while looping
#define TEST_SIZE  2749898     // Use long while looping
#define USER_SIZE  458293      // Use long while looping
#define MOVIE_SIZE  17770      // Use int while looping
#define TRAINING_DATA_AVERAGE 3.60861
using namespace std;
using namespace std::chrono;
using namespace Eigen;


struct svd_ans {
    double** U;
    double** V;
    double E_in;
    double E_val;
} ;

void populate_movies_to_array(int user_matrix[], short movie_matrix[], double rating_matrix[]);

double predict_score(double** U, double** V, double** SumMW, int userId, int itemId, double b_u, double b_i, double sqrt_r);

double get_err(double** U, double** V,
              int* user_matrix, short* movie_matrix,
              short* date_matrix, double* rating_matrix,
              double size, double reg, double** SumMW,
              double* user_bias, double* movie_bias);

double** read_matrix_from_file(long n_rows, int n_cols, string filename) {
    ifstream inFile;
    inFile.open(filename);
    if (!inFile) {
      cout << "File not opened." << endl;
      exit(1);
    }
    double** matrix = new double*[n_rows];
    for (long r = 0; r < n_rows; r++) {
        matrix[r] = new double[n_cols];
        for (int c = 0; c < n_cols; c++) {
            inFile >> matrix[r][c];
            cout << matrix[r][c] << "\t";
        }
        cout << endl;
    }
    inFile.close();
    return matrix;
}

void checkpoint_U_V(MatrixXd U, MatrixXd V, int epoch);

svd_ans train_model_from_UV(double eta, double reg,
                            int* user_matrix, short* movie_matrix,
                            short* date_matrix, double* rating_matrix,
                            int* user_matrix_val, short* movie_matrix_val,
                            short* date_matrix_val, double* rating_matrix_val,
                            MatrixXd U, MatrixXd V, MatrixXd Y, MatrixXd SumMW);

svd_ans complete_training(double eta, double reg);

#endif
