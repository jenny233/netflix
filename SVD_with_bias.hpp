#ifndef __SVD_with_bias_H__
#define __SVD_with_bias_H__

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
using namespace Eigen;


struct svd_ans {
    MatrixXd U;
    MatrixXd V;
    double E_in;
    double E_val;
} ;

void populate_movies_to_array(int user_matrix[], short movie_matrix[], double rating_matrix[]);

double predict_score(VectorXd Ui, VectorXd Vj, VectorXd SumMWi, int r_u, double b_u, double b_i);

VectorXd grad_U(VectorXd Ui, double Yij, VectorXd Vj, double reg, double eta, double score);

VectorXd grad_V(VectorXd Vj, double Yij, VectorXd Ui, double reg, double eta, double score, double r_u, VectorXd SumMWi);

double get_err(MatrixXd U, MatrixXd V,
              int* user_matrix, short* movie_matrix,
              short* date_matrix, double* rating_matrix,
              double size, double reg, MatrixXd SumMW,
              double* user_bias, double* movie_bias);

MatrixXd read_matrix_from_file(int n_rows, int n_cols, string filename) {
    MatrixXd matrix(n_rows, n_cols);
    ifstream inFile;
    inFile.open(filename);
    if (!inFile) {
      std::cout << "File not opened." << endl;
      exit(1);
    }
    for (long r = 0; r < n_rows; r++) {
      for (int c = 0; c < n_cols; c++) {
          inFile >> matrix(r, c);
      }
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
