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

#define TRAIN_SIZE  94362234
// #define TRAIN_SIZE 500000
#define VALID_SIZE  1965045
#define USER_SIZE  458293
#define MOVIE_SIZE  17770

using namespace std;
using namespace Eigen;


struct svd_ans {
    MatrixXd U;
    MatrixXd V;
    double E_val;
} ;

VectorXd grad_U(VectorXd Ui, double Yij, VectorXd Vj, double reg, double eta);

VectorXd grad_V(VectorXd Vj, double Yij, VectorXd Ui, double reg, double eta);

double get_err(MatrixXd U, MatrixXd V,
              int* user_matrix, short* movie_matrix,
              short* date_matrix, char* rating_matrix, int size, double reg=0.0);

svd_ans train_model(int M, int N, int K, double eta, double reg,
                    int* user_matrix, short* movie_matrix,
                    short* date_matrix, char* rating_matrix,
                    int* user_matrix_val, short* movie_matrix_val,
                    short* date_matrix_val, char* rating_matrix_val,
                    int max_epochs);


#endif
