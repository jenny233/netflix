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


struct ans {
    MatrixXd U;
    MatrixXd V;
    float error;
} ;

// Four arrays to store all the data read in
int* user_matrix = new int[TRAIN_SIZE];
short* movie_matrix = new short[TRAIN_SIZE];
short* date_matrix = new short[TRAIN_SIZE];
char* rating_matrix = new char[TRAIN_SIZE];


VectorXd grad_U(VectorXd Ui, float Yij, VectorXd Vj, float reg, float eta) {
    /*
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    */
    return (1-reg * eta) * Ui + eta * (Yij - Ui.dot(Vj)) * Vj;
}

VectorXd grad_V(VectorXd Vj, float Yij, VectorXd Ui, float reg, float eta) {
    /*
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    */
    return (1-reg*eta)*Vj + eta * Ui * (Yij - Ui.dot(Vj));
}

float get_err(MatrixXd U, MatrixXd V,
              int* user_matrix, short* movie_matrix,
              short* date_matrix, char* rating_matrix, float reg=0.0) {
    /*
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    */
    // Compute mean squared error on each data point in Y; include
    // regularization penalty in error calculations.
    // We first compute the total squared squared error
    float err = 0.0;

    for (int r=0; r<TRAIN_SIZE; r++) {
        int i = user_matrix[r];
        int j = movie_matrix[r];
        int date = date_matrix[r];
        int Yij = rating_matrix[r] - '0';
        err += 0.5 * pow(Yij - U.row(i-1).dot( V.col(j-1) ) , 2.0);
    }
    // Add error penalty due to regularization if regularization
    // parameter is nonzero
    if (reg != 0) {
        float U_frobenius_norm = U.squaredNorm();
        float V_frobenius_norm = V.squaredNorm();
        err += 0.5 * reg * pow(U_frobenius_norm, 2.0);
        err += 0.5 * reg * pow(V_frobenius_norm, 2.0);
    }
    // Return the mean of the regularized error
    return err / TRAIN_SIZE;
}


ans train_model(int M, int N, int K, float eta, float reg,
                int* user_matrix, short* movie_matrix,
                short* date_matrix, char* rating_matrix,
                float eps=0.0001, int max_epochs=EPOCH) {
    /*
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    */
    // Initialize U, V
    MatrixXd U = MatrixXd::Random(M, K) - MatrixXd::Constant(M, K, 0.5);
    MatrixXd V = MatrixXd::Random(K, N) - MatrixXd::Constant(K, N, 0.5);
    float delta = 0;
    int indices[TRAIN_SIZE];
    for (int i = 0; i <= TRAIN_SIZE; i++) indices[i] = i;

    for (int epoch = 0; epoch < max_epochs; epoch++) {

        // Progress bar
        // if (count % 1000000 == 0) {
        //     cout << "\r" << to_string(count * 100 / 102416306) << "%%" << flush;
        // }

        // Run an epoch of SGD
        float before_E_in = get_err(U, V, user_matrix, movie_matrix,
                                    date_matrix, rating_matrix, reg);
        srand ( unsigned ( time(0) ) );
        random_shuffle ( indices, indices+TRAIN_SIZE );
        for (int a = 0; a < TRAIN_SIZE; a++) {
            int ind = indices[a];
            int i = user_matrix[ind];
            int j = movie_matrix[ind];
            int date = date_matrix[ind];
            int Yij = rating_matrix[ind] - '0';
            // Update U[i], V[j]
            U.row(i-1) = grad_U(U.row(i-1), Yij, V.col(j-1), reg, eta);
            V.col(j-1) = grad_V(V.col(j-1), Yij, U.row(i-1), reg, eta);
        }
        // At end of epoch, print E_in
        float E_in = get_err(U, V, user_matrix, movie_matrix,
                             date_matrix, rating_matrix, reg);
        cout << "Epoch" << to_string(epoch+1)
             << ", E_in (regularized MSE): " << E_in << endl;

        // Compute change in E_in for first epoch
        if (epoch == 0) {
            delta = before_E_in - E_in;
        }
        // If E_in doesn't decrease by some fraction <eps>
        // of the initial decrease in E_in, stop early
        else if (before_E_in - E_in < eps * delta) {
            break;
        }
    }

    cout << "U:" << endl;
    cout << U.block<5,5>(0, 0) << endl;
    cout << "V:" << endl;
    cout << V.block<5,5>(0, 0) << endl;


    ans result = {U, V, get_err(U, V, user_matrix, movie_matrix,
                                date_matrix, rating_matrix)};
    return result;
}


int main() {
    cout << "Reading input." << endl;
    ifstream inFile;

    inFile.open("dataset1_random_samples_500000.dta");

    if (!inFile) {
        std::cout << "File not opened." << endl;
        exit(1);
    }
    int number;
    for (int i=0; i<TRAIN_SIZE; i++) {
        inFile >> user_matrix[i];
		inFile >> movie_matrix[i];
		inFile >> date_matrix[i];
		inFile >> rating_matrix[i];
    }
    inFile.close();

    cout << "Training model." << endl;
    ans result = train_model(USER_SIZE, MOVIE_SIZE, 5, 0.03, 0.0,
                             user_matrix, movie_matrix,
                             date_matrix, rating_matrix);

    cout << "E_in final: " << result.error << endl;

    return 0;
}
