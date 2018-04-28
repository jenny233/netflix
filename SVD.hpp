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

#define TRAIN_SIZE  500000
#define USER_SIZE  458294
#define MOVIE_SIZE  17771
#define EPOCH  300

using namespace std;
using namespace Eigen;


struct svd_ans {
    MatrixXd U;
    MatrixXd V;
    float error;
} ;

VectorXd grad_U(VectorXd Ui, float Yij, VectorXd Vj, float reg, float eta);

VectorXd grad_V(VectorXd Vj, float Yij, VectorXd Ui, float reg, float eta);

float get_err(MatrixXd U, MatrixXd V,
              int* user_matrix, short* movie_matrix,
              short* date_matrix, char* rating_matrix, float reg=0.0);

svd_ans train_model(int M, int N, int K, float eta, float reg,
                    int* user_matrix, short* movie_matrix,
                    short* date_matrix, char* rating_matrix,
                    float eps=0.0001, int max_epochs=EPOCH);


#endif
