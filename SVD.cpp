#include "SVD.hpp"
#include <string>
#define LATENT_FACTORS 10
#define REGULARIZATION 0.07
#define LEARNING_RATE  0.05
#define MAX_EPOCH      400
#define PRED_FILENAME ("../predictions" + to_string(LATENT_FACTORS) + "lf_.dta")


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
              short* date_matrix, double* rating_matrix,
              double size, double reg) {
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
        double Yij = rating_matrix[r];
        err += pow(Yij - U.row(i-1).dot( V.col(j-1) ) , 2.0);
    }
    // Return the RMSE
    return sqrt(err / size);
}

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

svd_ans train_model_from_UV(int M, int N, int K, double eta, double reg,
                            int* user_matrix, short* movie_matrix,
                            short* date_matrix, double* rating_matrix,
                            int* user_matrix_val, short* movie_matrix_val,
                            short* date_matrix_val, double* rating_matrix_val,
                            MatrixXd U, MatrixXd V, int max_epochs) {
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

    double E_in, E_val, init_E_in, init_E_val, delta;

    // Calculate E_in only on a portion of the in data.

    system_clock::time_point start_time, end_time; // initialize timers
    start_time = system_clock::now();

    srand ( unsigned ( time(0) ) );
    long rand_n = rand() % (TRAIN_SIZE-1000001);
    init_E_in = get_err(U, V, user_matrix+rand_n, movie_matrix+rand_n,
                        date_matrix+rand_n, rating_matrix+rand_n,
                        1000000, reg);

    end_time = system_clock::now();
    auto duration = duration_cast<seconds>( end_time - start_time ).count();

    cout << "Initial E_in: " << init_E_in
         << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << endl;

    start_time = system_clock::now();
    init_E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                         date_matrix_val, rating_matrix_val,
                         VALID_SIZE, reg);
    end_time = system_clock::now();
    duration = duration_cast<seconds>( end_time - start_time ).count();
    cout << "Initial E_val: " << init_E_val
         << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << endl;


    // Stochastic gradient descent
    ofstream outFile;
    for (int epoch = 0; epoch < max_epochs; epoch++) {

        cout << "Epoch " << epoch << ":" << endl;
        start_time = system_clock::now();

        // Checkpoint every 10 epochs
		if (epoch % 10 == 0 )
		{
			cout<<"printing checkpoint"<<endl;
			string filename = ("../svd_U_matrix_"+to_string(K)+"lf_"+to_string(epoch)+"ep.txt");
			// Write U and V to a file
			outFile.open(filename);
			if (outFile.is_open()) {
				outFile << U;
			}
			outFile.close();
			filename = ("../svd_V_matrix_"+to_string(K)+"lf_"+to_string(epoch)+"ep.txt");
			outFile.open(filename);
			if (outFile.is_open()) {
				outFile << V;
			}
			outFile.close();
		}
        for (long ind = 0; ind < TRAIN_SIZE; ind++) {

            // Progress bar
            if (ind % 1000000 == 0) {
                end_time = system_clock::now();
                duration = duration_cast<seconds>( end_time - start_time ).count();
                cout << "\r" << to_string(ind * 100 / TRAIN_SIZE) << "%%"
                     << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << flush;
            }

            // Update U[i], V[j]
            int i = user_matrix[ind];
            int j = movie_matrix[ind];
            int date = date_matrix[ind];
            double Yij = rating_matrix[ind];

            U.row(i-1) = grad_U(U.row(i-1), Yij, V.col(j-1), reg, eta);
            V.col(j-1) = grad_V(V.col(j-1), Yij, U.row(i-1), reg, eta);
        }

        // At end of epoch, print E_in, E_val
        srand ( unsigned ( time(0) ) );
        rand_n = rand() % (TRAIN_SIZE-1000001);
        E_in = get_err(U, V, user_matrix+rand_n, movie_matrix+rand_n,
                            date_matrix+rand_n, rating_matrix+rand_n,
                            1000000, reg);
        E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                              date_matrix_val, rating_matrix_val,
                              VALID_SIZE, reg);
        cout << endl << "E_in: " << E_in << "  E_val: " << E_val << endl;


        // If E_val doesn't decrease, stop early
        if (init_E_val <= E_val) {
            cout<<"E_val is increasing! Printing checkpoint"<<endl;
			string filename = ("../svd_U_matrix_"+to_string(K)+"lf_"+to_string(epoch)+"ep.txt");
			// Write U and V to a file
			outFile.open(filename);
			if (outFile.is_open()) {
				outFile << U;
			}
			outFile.close();
			filename = ("../svd_V_matrix_"+to_string(K)+"lf_"+to_string(epoch)+"ep.txt");
			outFile.open(filename);
			if (outFile.is_open()) {
				outFile << V;
			}
			outFile.close();
            break;
        }
        init_E_val = E_val;
        // eta = 0.9 * eta;
    }
    cout << endl;

    svd_ans result = {U, V, E_in, E_val};
    return result;
}

svd_ans complete_training(int M, int N, int K, double eta, double reg, int max_epochs) {
    // Four arrays to store all the training data read in
    int* user_matrix = new int[TRAIN_SIZE];
    short* movie_matrix = new short[TRAIN_SIZE];
    short* date_matrix = new short[TRAIN_SIZE];
    double* rating_matrix = new double[TRAIN_SIZE];

    // Four arrays to store all the validation data read in
    int* user_matrix_val = new int[VALID_SIZE];
    short* movie_matrix_val = new short[VALID_SIZE];
    short* date_matrix_val = new short[VALID_SIZE];
    double* rating_matrix_val = new double[VALID_SIZE];

    // IO
    ifstream inFile;
    ofstream outFile;


    // Read training data
    cout << "Reading training input." << endl;
    inFile.open("../dataset1_shuffled_all.dta");
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
    }
    cout << endl;
    inFile.close();



    // Read validation data
    cout << "Reading validation input." << endl;
    inFile.open("../dataset2_shuffled_all.dta");
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


    // Train SVD
    cout << "Training model." << endl;

    MatrixXd U = MatrixXd::Random(M, K);
    MatrixXd V = MatrixXd::Random(K, N);
    svd_ans result = train_model_from_UV(M, N, K, eta, reg,
                                        user_matrix, movie_matrix,
                                        date_matrix, rating_matrix,
                                        user_matrix_val, movie_matrix_val,
                                        date_matrix_val, rating_matrix_val,
                                        U, V, max_epochs);

    cout << "Final E_in: " << result.E_in << "  E_val: " << result.E_val << endl;

    delete[] rating_matrix;
	delete[] user_matrix;
	delete[] movie_matrix;
    delete[] date_matrix;
    delete[] rating_matrix_val;
	delete[] user_matrix_val;
	delete[] movie_matrix_val;
    delete[] date_matrix_val;

    return result;


    // Read in test data
    int* user_matrix_test = new int[TEST_SIZE];
    short* movie_matrix_test = new short[TEST_SIZE];
    short* date_matrix_test = new short[TEST_SIZE];
    cout << "Reading testing input." << endl;
    inFile.open("dataset5_unshuffled_all.dta");
    if (!inFile) {
        cout << "File not opened." << endl;
        exit(1);
    }
    int garbage_zero_rating;
    for (long i=0; i<TEST_SIZE; i++) {
        inFile >> user_matrix_test[i];
        inFile >> movie_matrix_test[i];
        inFile >> date_matrix_test[i];
        inFile >> garbage_zero_rating;
        if (i % 1000000 == 0) {
            cout << "\r" << to_string(i * 100 / TEST_SIZE) << "%%" << flush;
        }
    }
    cout << endl;
    inFile.close();

    // Make predictions
    outFile.open(PRED_FILENAME);
    for (long r=0; r<TEST_SIZE; r++) {
        int i = user_matrix_test[r];
        int j = movie_matrix_test[r];
        double prediction = result.U.row(i-1).dot( result.V.col(j-1) );
        outFile << prediction << endl;
    }
    outFile.close();

    delete[] user_matrix_test;
    delete[] movie_matrix_test;
    delete[] date_matrix_test;
}



void predict_from_UV(int M, int N, int K, MatrixXd U, MatrixXd V) {

    // Initialize

    ifstream inFile;
    ofstream outFile;

    int* user_matrix_test = new int[TEST_SIZE];
    short* movie_matrix_test = new short[TEST_SIZE];
    short* date_matrix_test = new short[TEST_SIZE];

    // Read in test data
    cout << "Reading testing input." << endl;
    inFile.open("../dataset5_unshuffled_all.dta");
    if (!inFile) {
        cout << "File not opened." << endl;
        exit(1);
    }
    int garbage_zero_rating;
    for (long i=0; i<TEST_SIZE; i++) {
        inFile >> user_matrix_test[i];
        inFile >> movie_matrix_test[i];
        inFile >> date_matrix_test[i];
        inFile >> garbage_zero_rating;
        if (i % 1000000 == 0) {
            cout << "\r" << to_string(i * 100 / TEST_SIZE) << "%%" << flush;
        }
    }
    cout << endl;
    inFile.close();


    // Make predictions
    cout << "Making predictions." << endl;
    outFile.open(PRED_FILENAME);
    for (long r=0; r<TEST_SIZE; r++) {
        int i = user_matrix_test[r];
        int j = movie_matrix_test[r];
        double prediction = U.row(i-1).dot( V.col(j-1) );
        if (prediction < 1)
        {
			prediction = 1;
		}
		if (prediction > 5)
		{
			prediction = 5;
		}
        outFile << prediction << endl;
    }
    outFile.close();

    delete[] user_matrix_test;
    delete[] movie_matrix_test;
    delete[] date_matrix_test;

}


int main() {

    // To do training from the very beginning
    svd_ans result = complete_training(USER_SIZE, MOVIE_SIZE, LATENT_FACTORS,
                                       LEARNING_RATE, REGULARIZATION, MAX_EPOCH);
    predict_from_UV(USER_SIZE, MOVIE_SIZE, LATENT_FACTORS, result.U, result.V);



    // To do training from saved U V matrices

    // MatrixXd U = read_matrix_from_file(M, K, U_filename);
    // MatrixXd V = read_matrix_from_file(K, N, V_filename);
    // svd_ans result = train_model_from_UV(M, N, K, eta, reg,
    //                                     user_matrix, movie_matrix,
    //                                     date_matrix, rating_matrix,
    //                                     user_matrix_val, movie_matrix_val,
    //                                     date_matrix_val, rating_matrix_val,
    //                                     U, V, max_epochs);
    // predict_from_UV(USER_SIZE, MOVIE_SIZE, LATENT_FACTORS, result.U, result.V);



    // To predict from saved U V matrices

    // MatrixXd U = read_matrix_from_file(M, K, U_filename);
    // MatrixXd V = read_matrix_from_file(K, N, V_filename);
    // predict_from_UV(USER_SIZE, MOVIE_SIZE, LATENT_FACTORS, U, V);


    return 0;
}
