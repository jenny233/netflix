#include "SVD_with_bias.hpp"
#include <string>
#include <cmath>
#define LATENT_FACTORS 30
#define REGULARIZATION 0.02
#define LEARNING_RATE  0.005
#define MAX_EPOCH      400
#define PRED_FILENAME ("../predictions" + to_string(LATENT_FACTORS) + "lf_.dta")

// Two arrays for holding the user and movie bias
double* user_bias = new double[USER_SIZE + 1];
double* movie_bias = new double[MOVIE_SIZE + 1];
vector<int>* movies_rated_by_user = new vector<int> [USER_SIZE];
vector<int>* ratings_by_user = new vector<int> [USER_SIZE];

void populate_movies_to_array(int user_matrix[], short movie_matrix[], double rating_matrix[]) {
	/*
	 * This array takes the user and movie matrix and for each user it
	 * pushes back the movies that that user has rated in the
	 * movies _rated by_user_array.
	 */
	 cout<< "populating user rating arrays"<<endl;
	// populate user array with empty vectors
	for (long i = 0; i < USER_SIZE; i++)
	{
		vector <int> m;
		movies_rated_by_user[i] = m;
		ratings_by_user[i] = m;
	}
	// Fill the array of vectors with the movies that each user rated
	for (long long i = 0; i < READ_IN_LINES ; i++)
	{
		int movie = movie_matrix[i];
		int rating = rating_matrix[i] ;
		{
			(movies_rated_by_user[user_matrix[i]]).push_back(movie);
			(ratings_by_user[user_matrix[i]]).push_back(rating);
		}
	}
}


double predict_score(VectorXd Ui, VectorXd Vj, VectorXd SumMWi, int r_u, double b_u, double b_i) {
	/*
	 * This function perdicts the score of a rating for a user and movie
	 * using the implicit factors. It incorporates the Y matrix and the
	 * biases
	 */
	double temp = 0.0;
    for (int i = 0; i < LATENT_FACTORS; i++)
    {
		 temp = (Ui(i) + SumMWi(i) * r_u) * Vj(i);
	}
	double score = TRAINING_DATA_AVERAGE + b_u + b_i + temp;
	if(score > 5)
	{
		score = 5;
	}
	if (score < 1)
	{
		score = 1;
	}
	return score;
}


VectorXd grad_U(VectorXd Ui, double Yij, VectorXd Vj, double reg, double eta, double score) {
    /*
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    */
    return eta * ((Yij - score) * Vj - reg * Ui);
    // return (1-reg * eta) * Ui + eta * (Yij - score) * Vj;
}

VectorXd grad_V(VectorXd Vj, double Yij, VectorXd Ui, double reg, double eta, double score, double r_u, VectorXd SumMWi) {
    /*
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    */
    return eta * ((Yij - score) * (Ui + r_u * SumMWi) - reg * Vj);
    // return (1 - reg * eta) * Vj + eta * (Ui + r_u * SumMWi) * (Yij - score);
}

double get_err(MatrixXd U, MatrixXd V,
              int* user_matrix, short* movie_matrix,
              short* date_matrix, double* rating_matrix,
              double size, double reg, MatrixXd SumMW,
              double* user_bias, double* movie_bias) {

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
	{
		double r_u = 0.0;
		for (long r=0; r<size; r++) {
			int i = user_matrix[r];
			int j = movie_matrix[r];
			double Yij = rating_matrix[r];
			int sz = (movies_rated_by_user[i]).size();
			// Want to make sure it is not 0 before we divide and take the square root
			if (sz > 0)
			{
				r_u = 1 / sqrt(sz);
			}
			double score = predict_score(U.row(i-1), V.col(j-1), SumMW.row(i-1), r_u, user_bias[i], movie_bias[j]);
			err += pow(Yij - score , 2.0);
		}
	}
    // Return the RMSE
    return sqrt(err / size);
}



void checkpoint_U_V(MatrixXd U, MatrixXd V, int epoch) {
    ofstream outFile;
    cout<<"  printing checkpoint"<<endl;
    string filename = ("../svd_U_matrix_"+to_string(LATENT_FACTORS)+"lf_"+to_string(epoch)+"ep.txt");
    // Write U and V to a file
    outFile.open(filename);
    if (outFile.is_open()) {
        outFile << U;
    }
    outFile.close();
    filename = ("../svd_V_matrix_"+to_string(LATENT_FACTORS)+"lf_"+to_string(epoch)+"ep.txt");
    outFile.open(filename);
    if (outFile.is_open()) {
        outFile << V;
    }
    outFile.close();
}


svd_ans train_model_from_UV(double eta, double reg,
                            int* user_matrix, short* movie_matrix,
                            short* date_matrix, double* rating_matrix,
                            int* user_matrix_val, short* movie_matrix_val,
                            short* date_matrix_val, double* rating_matrix_val,
                            MatrixXd U, MatrixXd V, MatrixXd Y, MatrixXd SumMW) {
    /*
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    USER_SIZE x LATENT_FACTORS matrix U and MOVIE_SIZE x LATENT_FACTORS matrix V such that rating Y_ij is approximated
    by (UV)_ij.

    Uses an initial learning rate of <eta> and regularization of <reg>. Stops
    after <MAX_EPOCH> epochs, or MSE of validation set stops decreasing.
    Learning rate decreases by 10% every epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    */

    double E_in, E_val, init_E_in, init_E_val;

    // Initialize timers
    system_clock::time_point start_time, end_time;

    // Calculate initial E_in only on a portion of the in data.
    start_time = system_clock::now();
    srand ( unsigned ( time(0) ) );
    long rand_n = rand() % (TRAIN_SIZE-1000001);
    init_E_in = get_err(U, V, user_matrix+rand_n, movie_matrix+rand_n,
                        date_matrix+rand_n, rating_matrix+rand_n,
                        1000000, reg, SumMW, user_bias, movie_bias);
    end_time = system_clock::now();
    auto duration = duration_cast<seconds>( end_time - start_time ).count();
    cout << "Initial E_in: " << init_E_in
         << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << endl;

    // Calculate initial E_val
    start_time = system_clock::now();
    init_E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                         date_matrix_val, rating_matrix_val,
                         VALID_SIZE, reg, SumMW, user_bias, movie_bias);
    end_time = system_clock::now();
    duration = duration_cast<seconds>( end_time - start_time ).count();
    cout << "Initial E_val: " << init_E_val
         << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << endl;

    // Stochastic gradient descent
    ofstream outFile;
    for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {

        cout << "Epoch " << epoch << ":" << endl;
        start_time = system_clock::now();

        // Checkpoint every 10 epochs
		if (epoch % 10 == 0 ) {
            checkpoint_U_V(U, V, epoch);
		}

		// Loop through the users
        for (long user = 1; user < USER_SIZE; user++) {

            // Progress bar
            if (user % 1000 == 0) {
                end_time = system_clock::now();
                duration = duration_cast<seconds>( end_time - start_time ).count();
                cout << "\r" << to_string(user * 100 / USER_SIZE) << "%%"
                     << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << flush;
            }

            // Update U[i], V[j]
            int i = user;
            double r_u = 0.0;
			// calculate R(u) ^ -1/2
			int sz = (movies_rated_by_user[user]).size();
			// Want to make sure it is not 0 before we divide and take the square root
			if (sz > 0)
			{
				r_u = 1 / sqrt(sz);
			}

			// Intialize a vector of 0's that have the same number of latent factors like in github code
			VectorXd tmpSum = VectorXd::Zero(LATENT_FACTORS);
			// List of movies rated by this particular user
			vector<int> user_movies = movies_rated_by_user[user];
			// List of ratings made by this particular user, is in the same order as the movies rated by this user
			vector<int> user_ratings = ratings_by_user[user];
			// populate the SumMW matrix
			for (int lat = 0; lat < LATENT_FACTORS; lat++)
			{
				double sumy = 0;
				// get the sum for all the movies rated by the user the
				// latent factors associated with the movie rated by the user
				for (int j = 0; j < sz; j++)
				{
					sumy += Y(user_movies[j], lat);
				}
				SumMW(i, lat) = sumy;
			}

			// update the U and V matricies using the gradient
			for (int f = 0; f < sz; f++)
			{
				//loop through all the movies rated by a user
				double Yij = user_ratings[f];
				int j = user_movies[f];
				// get the predicted score -> I created the predicted _score function
				double score = predict_score(U.row(i-1), V.col(j-1), SumMW.row(i-1), r_u, user_bias[i], movie_bias[j]);
				U.row(i-1) += grad_U(U.row(i-1), Yij, V.col(j-1), reg, eta, score);
				V.col(j-1) += grad_V(V.col(j-1), Yij, U.row(i-1), reg, eta, score, r_u, SumMW.row(i-1));
				// U.row(i-1) = grad_U(U.row(i-1), Yij, V.col(j-1), reg, eta, score);
				// V.col(j-1) = grad_V(V.col(j-1), Yij, U.row(i-1), reg, eta, score, r_u, SumMW.row(i-1));

                // This was done in the github code with a variable of the same name so I did it too
				tmpSum = tmpSum + (Yij - score) * r_u * V.col(j-1);
			}

			// Update the Y matrice for each movie a user rated
			for (int t = 0; t < sz; t++)
			{
				int j = user_movies[t];
				VectorXd tmpMW = Y.row(j-1);
				Y.row(j-1) = Y.row(j-1) + eta * (tmpSum - reg * tmpMW);
				SumMW.row(i-1) = SumMW.row(i-1) + Y.row(j-1) - tmpMW;
			}
        }

        // update part of the y matrix again in order to save us from
        // wasting too much time recalculaing the y matrix? idk they did this in their code
        for (int user = 0; user < USER_SIZE; user++)
        {
			vector<int> user_movies = movies_rated_by_user[user];
			int sz = (movies_rated_by_user[user]).size();
			double r_u = 0.0;
			// Want to make sure it is not 0 before we divide and take the square root
			if (sz > 0)
			{
				r_u = 1 / sqrt(sz);
			}
			for (int lat = 0; lat < LATENT_FACTORS; lat++)
			{
				double sumy = 0;
				// get the sum for all the movies rated by the user the
				// latent factors associated with the movie rated by the user
				for (int j = 0; j < sz; j++)
				{
					sumy += Y(user_movies[j], lat);
				}
				SumMW(user, lat) = sumy;
			}
		}

        // At end of epoch, print E_in, E_val
        srand ( unsigned ( time(0) ) );
        rand_n = rand() % (TRAIN_SIZE-1000001);
        E_in = get_err(U, V, user_matrix+rand_n, movie_matrix+rand_n,
                            date_matrix+rand_n, rating_matrix+rand_n,
                            1000000, reg, SumMW,user_bias,movie_bias);
        E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                              date_matrix_val, rating_matrix_val,
                              VALID_SIZE, reg,SumMW, user_bias, movie_bias);

        cout << endl << "E_in: " << E_in << "  E_val: " << E_val << endl;


        // If E_val doesn't decrease, stop early
        if (init_E_val <= E_val) {
            cout<<"E_val is increasing! Printing checkpoint"<<endl;
            checkpoint_U_V(U, V, epoch);
            break;
        }
        init_E_val = E_val;
        // eta = 0.9 * eta;
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
	ifstream inFile_bias;

	// Read in bias /////
	inFile_bias.open("bias_checkpoint.txt");
	cout << "Reading bias data for users" << endl;
	for (long i = 0; i <=USER_SIZE; i++) {
        inFile_bias >> user_bias[i];
        if (i % 100 == 0) {
            cout << "\r" << to_string(i * 100 / (USER_SIZE+1)) << "%%" << flush;
        }
    }
    cout << "\nReading bias data for movies" << endl;
	for (long i = 0; i <= MOVIE_SIZE; i++) {
        inFile_bias >> movie_bias[i];
        if (i % 100 == 0) {
            cout << "\r" << to_string(i * 100 / (MOVIE_SIZE+1)) << "%%" << flush;
        }
    }
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
        if (i % 100 == 0) {
            cout << "\r" << to_string(i * 100 / TRAIN_SIZE) << "%%" << flush;
        }
    }
    cout << endl;
    inFile.close();
    inFile_bias.close();

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

	// Create y array
	MatrixXd Y = MatrixXd::Random(MOVIE_SIZE, LATENT_FACTORS);
	// Create the part of the Y array that will be used for the gradient
	MatrixXd sumMW = MatrixXd::Random(USER_SIZE, LATENT_FACTORS);
    // Train SVD
    cout << "Training model." << endl;

    MatrixXd U = MatrixXd::Random(USER_SIZE, LATENT_FACTORS);
    MatrixXd V = MatrixXd::Random(LATENT_FACTORS, MOVIE_SIZE);
    svd_ans result = train_model_from_UV(eta, reg,
                                        user_matrix, movie_matrix,
                                        date_matrix, rating_matrix,
                                        user_matrix_val, movie_matrix_val,
                                        date_matrix_val, rating_matrix_val,
                                        U, V, Y, sumMW);

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
}



int main() {

    // To do training from the very beginning
    svd_ans result = complete_training(LEARNING_RATE, REGULARIZATION);



    // To do training from saved U V matrices

    // MatrixXd U = read_matrix_from_file(U_filename);
    // MatrixXd V = read_matrix_from_file(V_filename);
    // svd_ans result = train_model_from_UV(eta, reg,
    //                                     user_matrix, movie_matrix,
    //                                     date_matrix, rating_matrix,
    //                                     user_matrix_val, movie_matrix_val,
    //                                     date_matrix_val, rating_matrix_val,
    //                                     U, V);



    return 0;
}
