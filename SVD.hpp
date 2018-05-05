#ifndef __SVD_H__
#define __SVD_H__

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

#define TRAIN_SIZE  94362233
// #define TRAIN_SIZE 500000
#define VALID_SIZE  1965045
#define TEST_SIZE  2749898
#define USER_SIZE  458293
#define MOVIE_SIZE  17770

using namespace std;
using namespace std::chrono;
using namespace Eigen;


struct svd_ans {
    MatrixXd U;
    MatrixXd V;
    double E_in;
    double E_val;
} ;

VectorXd grad_U(VectorXd Ui, double Yij, VectorXd Vj, double reg, double eta);

VectorXd grad_V(VectorXd Vj, double Yij, VectorXd Ui, double reg, double eta);

double get_err(MatrixXd U, MatrixXd V,
              int* user_matrix, short* movie_matrix,
              short* date_matrix, char* rating_matrix, double size, double reg=0.0);

svd_ans train_model_from_UV(int M, int N, int K, double eta, double reg,
                          int* user_matrix, short* movie_matrix,
                          short* date_matrix, char* rating_matrix,
                          int* user_matrix_val, short* movie_matrix_val,
                          short* date_matrix_val, char* rating_matrix_val,
                          MatrixXd U, MatrixXd V, int max_epochs);

svd_ans complete_training(int M, int N, int K, double eta, double reg, int max_epochs);

void predict_from_UV(int M, int N, int K, MatrixXd U, MatrixXd V);


#endif
