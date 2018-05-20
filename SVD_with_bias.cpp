#include "b.h"
#include <string>
#include <cmath>
#define LATENT_FACTORS 1000
#define REGULARIZATION 0.015
#define LEARNING_RATE  0.007
#define MAX_EPOCH      400
#define REG_BIAS       0.005
#define PRED_FILENAME ("../predictions_svd++_" + to_string(LATENT_FACTORS) + "lf_5_16.dta")

/*
 * IMPORTANT:
 * All arrays and matrices hold user and movie indices instead of ids
 * userId = user index + 1
 * Except for the data read from the training, validation, testing files.
 */

// Arrays of vectors that store movie indices (not ids) rated by each user
vector<int>* movies_rated_by_user = new vector<int> [USER_SIZE];
vector<int>* ratings_by_user = new vector<int> [USER_SIZE];

void populate_movies_to_array(int user_matrix[], short movie_matrix[], double rating_matrix[]) {
	/*
	 * This array takes the user and movie matrix and for each user it
	 * pushes back the movies that that user has rated in the
	 * movies _rated by_user_array.
	 */
	cout<< "Populating user rating arrays"<<endl;
	// populate user array with empty vectors
	for (long i = 0; i < USER_SIZE; i++)
	{
		vector <int> m;
		movies_rated_by_user[i] = m;
		ratings_by_user[i] = m;
	}
	// Fill the array of vectors with the movies that each user rated
    system_clock::time_point start_time, end_time;
    start_time = system_clock::now();
	for (long long r = 0; r < TRAIN_SIZE ; r++)
	{
        // Progress bar
        if (r % 100 == 0) {
            end_time = system_clock::now();
            auto duration = duration_cast<seconds>( end_time - start_time ).count();
            cout << "\r" << to_string(r * 100 / TRAIN_SIZE) << "%%"
                 << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << flush;
        }

        long userId = user_matrix[r];
		int movieId = movie_matrix[r];
		int rating = rating_matrix[r];
		(movies_rated_by_user[userId - 1]).push_back(movieId - 1);
		(ratings_by_user[userId - 1]).push_back(rating);
	}
    cout << endl;
}

double predict_score(double** U, double** V,
                     int i, int j, double b_u, double b_i){
    /*
    PARAMETERS
    U:      USER_SIZE by LATENT_FACTORS matrix
    V:      MOVIE_SIZE by LATENT_FACTORS matrix
    SumMW:  USER_SIZE by LATENT_FACTORS matrix
    i:      user index, userId - 1
    j:      movie index, movieId - 1
    b_u:    bias from the user
    b_i:    bias from the movie
    sqrt_r: (number of movies rated by this user)^-1/2

    RETURNS
    prediction for this user and movie combo
    */

    double tmp = 0.0;
    for(int k = 0; k < LATENT_FACTORS; k++){
        tmp += (U[i][k]) * V[j][k];
    }
    double score = TRAINING_DATA_AVERAGE + b_u + b_i + tmp;
    if(score > 5){
        score = 5;
    }
    if(score < 1){
        score = 1;
    }
    return score;
}

double get_err(double** U, double** V,
               int* user_matrix, short* movie_matrix,
               short* date_matrix, double* rating_matrix,
               double size, double reg,
               double* user_bias, double* movie_bias) {

    /*
    PARAMETERS
    U:          USER_SIZE by LATENT_FACTORS matrix
    V:          MOVIE_SIZE by LATENT_FACTORS matrix
    (user/movie/date/rating)_matrix:
                the dataset for error to be calculated oncould be training or validation
    size:       the size of said dataset
    reg:        regularization parameter
    SumMW:      USER_SIZE by LATENT_FACTORS matrix
    user_bias:  bias from the user
    movie_bias: bias from the movie

    RETURNS
    The mean squared-error of predictions


    Compute mean squared error on each data point; include
    regularization penalty in error calculations.
    We first compute the total squared squared error
    */
    double err = 0.0;
	{
		for (long r=0; r<size; r++) {
			int i = user_matrix[r] - 1;  // user index
			int j = movie_matrix[r] - 1; // movie index
			double Yij = rating_matrix[r];
            double score = predict_score(U, V, i, j, user_bias[i], movie_bias[j]);
			err += pow(Yij - score , 2.0);
		}
	}
    // Return the RMSE
    return sqrt(err / size);
}


void predict(double** U, double** V, double* user_bias,
			double* movie_bias, int* user_matrix_test, short* movie_matrix_test, short* date_matrix_test)
{
	cout<<"printing checkpoint"<<endl;
    ofstream outFile;
	outFile.open(PRED_FILENAME);
    // Make predictions
    for (long r=0; r<TEST_SIZE; r++) {
        int i = user_matrix_test[r] - 1;
        int j = movie_matrix_test[r] - 1;
		double prediction = predict_score( U, V, i, j, user_bias[i], movie_bias[j]);
        outFile << prediction << endl;
    }
    outFile.close();


}

void checkpoint_U_V(double** U, double** V, int epoch) {
    ofstream outFile;
    cout<<"  printing checkpoint"<<endl;
    string filename = ("../svd++_bias_U_matrix_"+to_string(LATENT_FACTORS)+"lf_"+to_string(epoch)+"ep.txt");
    // Write U and V to a file
    outFile.open(filename);
    if (outFile.is_open()) {
        for (int i = 0; i < USER_SIZE; i++) {
            for (int k = 0; k < LATENT_FACTORS; k++) {
                outFile << U[i][k];
                if (k < LATENT_FACTORS - 1) {
                    outFile << "\t";
                } else {
                    outFile << endl;
                }
            }
        }
    }
    outFile.close();
    filename = ("../svd++_bias_V_matrix_"+to_string(LATENT_FACTORS)+"lf_"+to_string(epoch)+"ep.txt");
    outFile.open(filename);
    if (outFile.is_open()) {
        for (int j = 0; j < MOVIE_SIZE; j++) {
            for (int k = 0; k < LATENT_FACTORS; k++) {
                outFile << V[j][k];
                if (k < LATENT_FACTORS - 1) {
                    outFile << "\t";
                } else {
                    outFile << endl;
                }
            }
        }
    }
    outFile.close();
}


svd_ans train_model_from_UV(double eta, double reg,
                            int* user_matrix, short* movie_matrix,
                            short* date_matrix, double* rating_matrix,
                            int* user_matrix_val, short* movie_matrix_val,
                            short* date_matrix_val, double* rating_matrix_val,
                            double** U, double** V,
                            int* user_matrix_test, short* movie_matrix_test,
                            short* date_matrix_test, double* bu, double* bi) {
    /*
    Given a training data Y_ij is user i's rating on movie j, learns an
    USER_SIZE x LATENT_FACTORS matrix U and MOVIE_SIZE x LATENT_FACTORS matrix V such that rating Y_ij is approximated
    by (UV)_ij.

    y is the second set of latent factors, MOVIE_SIZE by LATENT_FACTORS

    Uses an initial learning rate of <eta> and regularization of <reg>. Stops
    after <MAX_EPOCH> epochs, or MSE of validation set stops decreasing.
    Learning rate decreases by 10% every epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    */

    double E_in, E_val;
    double init_E_in = 100, init_E_val = 100;




    // Initialize timers
    system_clock::time_point start_time, end_time;

    // Stochastic gradient descent
    ofstream outFile;
    for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {

        cout << "Epoch " << epoch << ":" << endl;
        start_time = system_clock::now();

        // Checkpoint every 10 epochs
		if (epoch % 10 == 0 ) {
            predict( U, V, bu, bi,user_matrix_test, movie_matrix_test, date_matrix_test);

		}

		// Loop through the users, i is the user id - 1
        for (long i = 0; i < USER_SIZE; i++) {

            // Progress bar
            if (i % 1000 == 0) {
                end_time = system_clock::now();
                auto duration = duration_cast<seconds>( end_time - start_time ).count();
                cout << "\r" << to_string(i * 100 / USER_SIZE) << "%%"
                     << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << flush;
            }

           
			int sz = (movies_rated_by_user[i]).size();
			// Update the U and V matricies using the gradient
            // Loop through all the movies rated by a user
			for (int t = 0; t < sz; t++)
			{
				double Yij = ratings_by_user[i][t]; // the actual rating
				int j = movies_rated_by_user[i][t]; // the movie index

				// get the predicted score
                double score = predict_score(U, V, i, j, bu[i], bi[j]);
                double error = Yij - score;

                // Update U for user i and V for movie j
                for(int k = 0; k < LATENT_FACTORS; k++){
                    double uf = U[i][k]; // The U latent factor for this user i
                    double mf = V[j][k]; // The V latent factor for this movie j
                    U[i][k] += eta * (error * mf - reg * uf);
                    V[j][k] += eta * (error * uf - reg * mf);
                    bi[j] += eta * (error - REG_BIAS * bi[j]);
                    bu[i] += eta * (error - REG_BIAS * bu[i]);
                }
			}

        }


        // At end of epoch, print E_in, E_val
        srand ( unsigned ( time(0) ) );
        long rand_n = rand() % (TRAIN_SIZE-1000001);
        E_in = get_err(U, V, user_matrix+rand_n, movie_matrix+rand_n,
                            date_matrix+rand_n, rating_matrix+rand_n,
                            1000000, reg,bu,bi);
        E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                              date_matrix_val, rating_matrix_val,
                              VALID_SIZE, reg, bu, bi);

        cout << endl << "E_in: " << E_in << "  E_val: " << E_val << endl;


        // If E_val doesn't decrease, stop early
        if (init_E_val <= E_val) {
            cout<<"E_val is increasing! Printing checkpoint"<<endl;
            checkpoint_U_V(U, V, epoch);
            break;
        }
        init_E_val = E_val;
        eta *= (0.9 + 0.1 * rand() / RAND_MAX);
    }
    cout << endl;

    svd_ans result = {U, V, E_in, E_val};
    return result;
}

svd_ans complete_training(double eta, double reg) {
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
    cout << "\nReading training input." << endl;
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
        if (i % 100000 == 0) {
            cout << "\r" << to_string(i * 100 / TRAIN_SIZE) << "%%" << flush;
        }
    }
    cout << endl;
    inFile.close();


	populate_movies_to_array(user_matrix, movie_matrix, rating_matrix);

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

	cout<<" Reading out test data " <<endl;
	// Read in training data
    ifstream inFile_test;
	inFile_test.open("../dataset5_unshuffled_all.dta");
    int* user_matrix_test = new int[TEST_SIZE];
    short* movie_matrix_test = new short[TEST_SIZE];
    short* date_matrix_test = new short[TEST_SIZE];
    int garbage_zero_rating;
    for (long i=0; i<TEST_SIZE; i++) {
        inFile_test >> user_matrix_test[i];
        inFile_test >> movie_matrix_test[i];
        inFile_test >> date_matrix_test[i];
        inFile_test >> garbage_zero_rating;
        if (i % 100 == 0) {
            cout << "\r" << to_string(i * 100 / TEST_SIZE) << "%%" << flush;
        }
    }
    cout << endl;
    inFile_test.close();


    // Create U matrix
    double** U = new double* [USER_SIZE];
    // Create V matrix
    double** V = new double* [MOVIE_SIZE];

	double* bi = new double [MOVIE_SIZE];
	double* bu = new double [USER_SIZE];
	
	
	
    // Initialize the matrices
    for(int j = 0; j < MOVIE_SIZE; j++){
        V[j] = new double[LATENT_FACTORS];
        bi[j] = 0.0;
        for(int k = 0; k < LATENT_FACTORS; k++){
            srand ( unsigned ( time(0) ) );
            V[j][k] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(LATENT_FACTORS);
        }
    }
    for(int i = 0; i < USER_SIZE; i++){
        U[i] = new double[LATENT_FACTORS];
        bu[i] = 0.0;
        for(int k = 0; k < LATENT_FACTORS; k++){
            U[i][k] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(LATENT_FACTORS);
        }
    }

    // Train SVD
    cout << "Training model." << endl;

    // MatrixXd U = MatrixXd::Random(USER_SIZE, LATENT_FACTORS);
    // MatrixXd V = MatrixXd::Random(LATENT_FACTORS, MOVIE_SIZE);
	// MatrixXd y = MatrixXd::Random(MOVIE_SIZE, LATENT_FACTORS);
	// MatrixXd SumMW = MatrixXd::Random(USER_SIZE, LATENT_FACTORS);

    svd_ans result = train_model_from_UV(eta, reg,
                                        user_matrix, movie_matrix,
                                        date_matrix, rating_matrix,
                                        user_matrix_val, movie_matrix_val,
                                        date_matrix_val, rating_matrix_val,
                                        U, V,user_matrix_test,
                                        movie_matrix_test, date_matrix_test,
                                        bu,bi);

    cout << "Final E_in: " << result.E_in << "  E_val: " << result.E_val << endl;

	delete[] user_matrix_test;
    delete[] movie_matrix_test;
    delete[] date_matrix_test;
    delete[] rating_matrix;
	delete[] user_matrix;
	delete[] movie_matrix;
    delete[] date_matrix;
    delete[] rating_matrix_val;
	delete[] user_matrix_val;
	delete[] movie_matrix_val;
    delete[] date_matrix_val;

    for (long r = 0; r < USER_SIZE;  r++) { delete[] U[r]; }
    for (long r = 0; r < MOVIE_SIZE; r++) { delete[] V[r]; }
    delete[] U;
    delete[] V;

    return result;
}



int main() {

    // To do training from the very beginning
    svd_ans result = complete_training(LEARNING_RATE, REGULARIZATION);

    // To do training from saved U V matrices

    // double** U = read_matrix_from_file(U_filename);
    // double** V = read_matrix_from_file(V_filename);
    // svd_ans result = train_model_from_UV(eta, reg,
    //                                     user_matrix, movie_matrix,
    //                                     date_matrix, rating_matrix,
    //                                     user_matrix_val, movie_matrix_val,
    //                                     date_matrix_val, rating_matrix_val,
    //                                     U, V);



    return 0;
}
