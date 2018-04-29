#include "SVD.hpp"




VectorXd grad_U(VectorXd Ui, double Yij, VectorXd Vj, double reg, double eta) {
    /*
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    */
    return (1-reg * eta) * Ui + eta * (Yij - Ui.dot(Vj)) * Vj;
}

VectorXd grad_V(VectorXd Vj, double Yij, VectorXd Ui, double reg, double eta) {
    /*
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    */
    return (1-reg*eta)*Vj + eta * Ui * (Yij - Ui.dot(Vj));
}

double get_err(MatrixXd U, MatrixXd V,
              int* user_matrix, short* movie_matrix,
              short* date_matrix, char* rating_matrix,
              int size, double reg) {
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
    double err = 0.0;

    for (long r=0; r<size; r++) {
        int i = user_matrix[r];
        int j = movie_matrix[r];
        int Yij = rating_matrix[r] - '0';
        err += 0.5 * pow(Yij - U.row(i-1).dot( V.col(j-1) ) , 2.0);
    }
    // Add error penalty due to regularization if regularization
    // parameter is nonzero
    if (reg != 0) {
        double U_frobenius_norm = U.squaredNorm();
        double V_frobenius_norm = V.squaredNorm();
        err += 0.5 * reg * pow(U_frobenius_norm, 2.0);
        err += 0.5 * reg * pow(V_frobenius_norm, 2.0);
    }
    // Return the mean of the regularized error
    return err / size;
}


svd_ans train_model(int M, int N, int K, double eta, double reg,
                int* user_matrix, short* movie_matrix,
                short* date_matrix, char* rating_matrix,
                int* user_matrix_val, short* movie_matrix_val,
                short* date_matrix_val, char* rating_matrix_val,
                int max_epochs) {
    /*
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV)_ij.

    Uses an initial learning rate of <eta> and regularization of <reg>. Stops
    after <max_epochs> epochs, or MSE of validation set stops decreasing.
    Learning rate decreases by 10% every epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    */
    // Initialize U, V
    MatrixXd U = MatrixXd::Random(M, K) - MatrixXd::Constant(M, K, 0.5);
    MatrixXd V = MatrixXd::Random(K, N) - MatrixXd::Constant(K, N, 0.5);
    // long* indices = new long[TRAIN_SIZE];
    // for (long i = 0; i <= TRAIN_SIZE; i++) indices[i] = i;

    double E_val, before_E_val;
    before_E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                                date_matrix_val, rating_matrix_val,
                                VALID_SIZE, reg);
    cout << "Initial E_val: " << before_E_val << endl;
    for (int epoch = 0; epoch < max_epochs; epoch++) {

        cout << "Epoch " << epoch << ":" << endl;

        // Shuffle the data
        // srand ( unsigned ( time(0) ) );
        // random_shuffle ( indices, indices+TRAIN_SIZE );

        for (long ind = 0; ind < TRAIN_SIZE; ind++) {
            // Progress bar
            if (ind % 1000000 == 0) {
                cout << "\r" << to_string(ind * 100 / TRAIN_SIZE) << "%%" << flush;
            }

            int i = user_matrix[ind];
            int j = movie_matrix[ind];
            int date = date_matrix[ind];
            int Yij = rating_matrix[ind] - '0';
            // Update U[i], V[j]
            if (ind * 100 / TRAIN_SIZE == 77) {
                cout << ind << " " << i << " " << j << endl
                 << U.row(i-1) << endl << V.col(j-1) << endl << endl;
            }
            U.row(i-1) = grad_U(U.row(i-1), Yij, V.col(j-1), reg, eta);
            V.col(j-1) = grad_V(V.col(j-1), Yij, U.row(i-1), reg, eta);
        }
        // At end of epoch, print E_val
        // E_in = get_err(U, V, user_matrix, movie_matrix,
        //                      date_matrix, rating_matrix,
        //                      TRAIN_SIZE, reg);
        E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                              date_matrix_val, rating_matrix_val,
                              VALID_SIZE, reg);
        cout << endl;
        cout << "E_val: " << E_val << endl;
        cout << endl;


        // If E_val doesn't decrease, stop early
        if (before_E_val < E_val) {
            break;
        }
        before_E_val = E_val;
        eta = 0.9 * eta;
    }
    cout << endl;
    // cout << "U:" << endl;
    // cout << U.block<5,5>(0, 0) << endl;
    // cout << "V:" << endl;
    // cout << V.block<5,5>(0, 0) << endl;

    // delete[] indices;

    svd_ans result = {U, V, E_val};
    return result;
}


int main() {

    // Four arrays to store all the training data read in
    int* user_matrix = new int[TRAIN_SIZE];
    short* movie_matrix = new short[TRAIN_SIZE];
    short* date_matrix = new short[TRAIN_SIZE];
    char* rating_matrix = new char[TRAIN_SIZE];

    // Four arrays to store all the validation data read in
    int* user_matrix_val = new int[VALID_SIZE];
    short* movie_matrix_val = new short[VALID_SIZE];
    short* date_matrix_val = new short[VALID_SIZE];
    char* rating_matrix_val = new char[VALID_SIZE];

    ifstream inFile;

    // Read training data
    cout << "Reading training input." << endl;
    inFile.open("../dataset1_random_samples_all.dta");
    if (!inFile) {
        std::cout << "File not opened." << endl;
        exit(1);
    }
    for (long i=0; i<TRAIN_SIZE; i++) {
        inFile >> user_matrix[i];
		inFile >> movie_matrix[i];
		inFile >> date_matrix[i];
		inFile >> rating_matrix[i];
        if (i % 1000000 == 0) {
            cout << "\r" << to_string(i * 100 / TRAIN_SIZE) << "%%" << flush;
        }
        if (i >= 73599333 && i <= 73599337) {
            cout << endl << i << ": " << user_matrix[i] << " " << movie_matrix[i] << " " << date_matrix[i] << " " << rating_matrix[i] << endl;
        }
    }
    cout << endl;
    inFile.close();

    // Read validation data
    cout << "Reading validation input." << endl;
    inFile.open("../dataset2_random_samples_all.dta");
    if (!inFile) {
        std::cout << "File not opened." << endl;
        exit(1);
    }
    for (long i=0; i<VALID_SIZE; i++) {
        inFile >> user_matrix_val[i];
		inFile >> movie_matrix_val[i];
		inFile >> date_matrix_val[i];
		inFile >> rating_matrix_val[i];
        if (i % 100000 == 0) {
            cout << "\r" << to_string(i * 100 / VALID_SIZE) << "%%" << flush;
        }
    }
    cout << endl;
    inFile.close();

    cout << "Training model." << endl;
    cout << "data at index 73599334: " << user_matrix[73599334] << " " << movie_matrix[73599334] << endl;
    svd_ans result = train_model(USER_SIZE, MOVIE_SIZE, 5, 0.007, 0.05,
        user_matrix, movie_matrix, date_matrix, rating_matrix,
        user_matrix_val, movie_matrix_val ,date_matrix_val, rating_matrix_val, 150);

    cout << "Final E_val: " << result.E_val << endl;

    delete[] rating_matrix;
	delete[] user_matrix;
	delete[] movie_matrix;
    delete[] date_matrix;
    delete[] rating_matrix_val;
	delete[] user_matrix_val;
	delete[] movie_matrix_val;
    delete[] date_matrix_val;

    return 0;
}
