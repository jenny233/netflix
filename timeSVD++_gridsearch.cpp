#include "timeSVD++.h"
#include <string>
#include <cmath>

#define RUN_NUMBER 1

#define TRAIN_DATASET_SIZE 94362233
#define PROBE_DATASET_SIZE 1374739
#define VALID_DATASET_SIZE 1965045
#define TEST_DATASET_SIZE  2749898

#define TRAIN_IN_FILENAME "../dataset1_shuffled_all.dta"
#define TRAIN_SIZE        TRAIN_DATASET_SIZE
#define VALID_IN_FILENAME "../probe.dta"
#define VALID_SIZE        PROBE_DATASET_SIZE
#define TEST_IN_FILENAME  "../probe.dta"
#define TEST_SIZE         PROBE_DATASET_SIZE
// #define TEST_IN_FILENAME  "../dataset5_unshuffled_all.dta"
// #define TEST_SIZE         TEST_DATASET_SIZE
// #define VALID_IN_FILENAME "../dataset2_shuffled_all.dta"
// #define VALID_SIZE        VALID_DATASET_SIZE

int    LATENT_FACTORS      = 250;
double REGULARIZATION_UF   = 0.008;
double REGULARIZATION_MF   = 0.0006;
double REGULARIZATION_Y    = 0.003;
double REGULARIZATION_MB   = 0;
double REGULARIZATION_UB   = 0.003;
double REGULARIZATION_BINS = 0.008;
double REGULARIZATION      = 0.0060;
double LEARNING_RATE_UF    = 0.006;
double LEARNING_RATE_MF    = 0.011;
double LEARNING_RATE_Y     = 0.001;
double LEARNING_RATE_MB    = 0.003;
double LEARNING_RATE_UB    = 0.012;
double LEARNING_RATE_BINS  = 0.0012;
double LEARNING_RATE       = 0.012;
double reg_alpha      = 0.00001;
double eta_alpha      = 0.0004;
int    MAX_EPOCH      = 7;
string PRED_FILENAME = "../probe_predictions_timesvd++_"+to_string(RUN_NUMBER)+"_"+to_string(LATENT_FACTORS)+"lf_";

/*
 * IMPORTANT:
 * All arrays and matrices hold user and movie indices instead of ids
 * userId = user index + 1
 * Except for the data read from the training, validation, testing files.
 */

// Two arrays for holding the user and movie bias

// Arrays of vectors that store movie indices (not ids) rated by each user
vector<int>* movies_rated_by_user = new vector<int> [USER_SIZE];
vector<int>* ratings_by_user = new vector<int> [USER_SIZE];
// Create Tu, which is an array containinf the average time over all
// the ratings that a user did
vector<int>* times_of_user_ratings = new vector<int> [USER_SIZE];

// Create vector that holds a map for the deviations of time
vector<map<int, double> > Dev;
vector<map<int,double> > Bu_t;
void calculate_mean_times(double* Tu)
{
	/*
	 * This function is used to populate the Tu array, which stores the
	 * average day that each user rated on. This is done by simply taking
	 * the array of vectors that holds all the times that a user rated
	 * a movie adding it together and dividing the sum by the total number
	 * of movies that a user rated
	 */
	for (int i = 0; i < USER_SIZE; i++)
	{
		double total_times = 0;
		/* If the user did not rating any movies set their mean to 0 */
		if (times_of_user_ratings[i].size() == 0)
		{
			Tu[i] = 0;
			continue;
		}

		/*
		 * Loop through all the times that a user rated, add them up
		 *  and divide by the total number of ratings.
		 */
		for (unsigned j = 0; j < times_of_user_ratings[i].size(); j++)
		{
			total_times += (times_of_user_ratings[i])[j];
		}
		Tu[i] = total_times/times_of_user_ratings[i].size();
	}
}

int CalBin (int timeArg)
{
	/*
	 * This function takes in a time and calcuates which bin that this
	 * time corresponds to. This is used in the Bi,Bin(t) bias for
	 * movies
	 */
	int binsize = TIME_SIZE/ BIN_NUMBER + 1;
	return timeArg/binsize;
}

double CalDev( int user, int timeArg, double* Tu)
{
	/*
	 * This function calculates the deviative of time, which we will use
	 * to model the user's tastes over long periods of time.
	 */
	if(Dev[user].count(timeArg) != 0)
	{
		return Dev[user][timeArg];
	}
	double tmp = sign(timeArg - Tu[user]) * pow(double(abs(timeArg - Tu[user])),0.4);
	Dev[user][timeArg] = tmp;
	return tmp;
}

void populate_movies_to_array(int user_matrix[], short movie_matrix[], double rating_matrix[], short date_matrix []) {
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
		int date = date_matrix[r];
		int rating = rating_matrix[r];
		(movies_rated_by_user[userId - 1]).push_back(movieId - 1);
		(ratings_by_user[userId - 1]).push_back(rating);
		/* NEW */
		(times_of_user_ratings[userId - 1]).push_back(date);

	}
    cout << endl;
}

double predict_score(double** U, double** V, double** SumMW,
                     int i, short j, short time, double b_u, double b_i, double sqrt_r,
                     double bin_bias, double alpha, double* Tu){

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
        tmp += (U[i][k] + SumMW[i][k] * sqrt_r) * V[j][k];
    }
    double score = TRAINING_DATA_AVERAGE + b_u + b_i + bin_bias + alpha * CalDev(i, time, Tu) + Bu_t[i][time] +  tmp;
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
               double size, double reg, double** SumMW,
               double* bu, double* bi,
               double** Bi_Bin, double* Tu, double* Alpha_u) {

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
		double sqrt_r = 0.0;
		for (long r=0; r<size; r++) {
			int i = user_matrix[r] - 1;  // user index
			int j = movie_matrix[r] - 1; // movie index
			int time = date_matrix[r] - 1; //time index
			double Yij = rating_matrix[r];
			int sz = (movies_rated_by_user[i]).size();
			// Want to make sure it is not 0 before we divide and take the square root
			if (sz > 0)
			{
				sqrt_r = 1 / sqrt(sz);
			}
			if (i < 0)
			{
				cout<<"hi"<<endl;
				continue;
			}
            double score = predict_score(U, V, SumMW, i, j, time,bu[i], bi[j], sqrt_r, Bi_Bin[j][CalBin(time)], Alpha_u[i], Tu);

			err += pow(Yij - score , 2.0);
		}
	}
    // Return the RMSE
    return sqrt(err / size);
}


void predict(double** U, double** V, double** SumMW, double* bu,
			double* bi, int* user_matrix_test, short* movie_matrix_test,
			short* date_matrix_test, double** Bi_Bin, double* Tu,
			double* Alpha_u, int epoch)
{

	cout<<"printing predictions"<<endl;
    ofstream outFile;
	outFile.open(PRED_FILENAME + to_string(epoch)+ "ep.dta");
    // Make predictions
    for (long r=0; r<TEST_SIZE; r++) {
        int i = user_matrix_test[r] - 1;
        int j = movie_matrix_test[r] - 1;
        short time = date_matrix_test[r] - 1;
         // Update U[i], V[j]
		double sqrt_r = 0.0;
		// calculate R(u) ^ -1/2
		int sz = (movies_rated_by_user[i]).size();
		// Want to make sure it is not 0 before we divide and take the square root
		if (sz > 0)
		{
			sqrt_r = 1 / sqrt(sz);
		}
		double prediction = predict_score(U, V, SumMW, i, j, time, bu[i], bi[j], sqrt_r, Bi_Bin[j][CalBin(time)], Alpha_u[i], Tu);

        outFile << prediction << endl;
    }
    outFile.close();


}

void checkpoint_U_V(double** U, double** V, int epoch) {
    ofstream outFile;
    cout<<"  printing U, V matrices"<<endl;
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
                            double** U, double** V, double** y, double** SumMW,
                            int* user_matrix_test, short* movie_matrix_test,
                            short* date_matrix_test, double* bu, double* bi,
                            double** Bi_Bin, double* Tu, double* Alpha_u) {
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

    double E_in = 100, E_val = 100;
    double init_E_in = 100, init_E_val = 100;




    // Initialize timers
    system_clock::time_point start_time, end_time;

    // Stochastic gradient descent
	calculate_mean_times(Tu);
	// Populate the short term bias that will see if any movies were rated on the same day
	for ( int i = 0;  i < USER_SIZE; i++)
	{
		map<int,double> tmp;
		for (unsigned j = 0; j < times_of_user_ratings[i].size(); j++)
		{
			if(tmp.count((times_of_user_ratings[i])[j]) == 0)
			{
				tmp[(times_of_user_ratings[i])[j]] = 0.0000001;
			}
			else
			{
				continue;
			}
		}
		Bu_t.push_back(tmp);
	}
	for ( int i = 0;  i < USER_SIZE; i++)
	{
		map<int,double> tmp;
		Dev.push_back(tmp);
	}

    int epoch;
    for (epoch = 0; epoch < MAX_EPOCH; epoch++) {

        cout << "Epoch " << epoch << ":" << endl;
        start_time = system_clock::now();

        // Checkpoint every 10 epochs
		// if (epoch % 10 == 0 ) {
        //     predict( U, V, SumMW, bu, bi,user_matrix_test, movie_matrix_test, date_matrix_test, Bi_Bin, Tu, Alpha_u, epoch);
		// }
		// Loop through the users, i is the user id - 1
        for (long i = 0; i < USER_SIZE; i++) {
            // Progress bar
            if (i % 1000 == 0) {
                end_time = system_clock::now();
                auto duration = duration_cast<seconds>( end_time - start_time ).count();
                cout << "\r" << to_string(i * 100 / USER_SIZE) << "%%"
                     << "  Time: " << duration / 60 << "m" << duration % 60 << "s" << flush;
            }
            // Update U[i], V[j]
            double sqrt_r = 0.0;
			// calculate R(u) ^ -1/2
			int sz = (movies_rated_by_user[i]).size();
			// Want to make sure it is not 0 before we divide and take the square root
			if (sz > 0)
			{
				sqrt_r = 1 / sqrt(sz);
			}

			// tmpSum is used to update the y factors for movies the user rated
            // It's the same for one user
			vector <double> tmpSum(LATENT_FACTORS, 0);

            // SumMW stores the sum of y factors for the movies rated by each user
            // This is only updated for one user in this loop
			for (int k = 0; k < LATENT_FACTORS; k++) {
				double sumy = 0;
                // Loop through the t^th movie rated by this user
				for (int t = 0; t < sz; t++) {
					sumy += y[ movies_rated_by_user[i][t] ] [k];
				}
				SumMW[i][k] = sumy;
			}

			// Update the U and V matricies using the gradient
            // Loop through all the movies rated by a user
			for (int t = 0; t < sz; t++)
			{
				double Yij = ratings_by_user[i][t]; // the actual rating
				int j = movies_rated_by_user[i][t]; // the movie index
				int time = times_of_user_ratings[i][t]; // the time of the rating

				// get the predicted score
                double score = predict_score(U, V, SumMW, i, j, time, bu[i], bi[j], sqrt_r, Bi_Bin[j][CalBin(time)], Alpha_u[i],Tu);
                double error = Yij - score;

                bu[i] += REGULARIZATION_UB * (error - LEARNING_RATE_UB * bu[i]);
                bi[j] += REGULARIZATION_MB * (error - LEARNING_RATE_MB * bi[j]);
                Bi_Bin[j][CalBin(time)] += REGULARIZATION_BINS * (error - LEARNING_RATE_BINS * Bi_Bin[j][CalBin(time)]);
                Alpha_u[i] += reg_alpha * (error * CalDev(i,time,Tu) - eta_alpha * Alpha_u[i]);
                Bu_t[i][time] += reg * ( error - eta * Bu_t[i][time]);

                // Update U for user i and V for movie j
                for(int k = 0; k < LATENT_FACTORS; k++){
                    double uf = U[i][k]; // The U latent factor for this user i
                    double mf = V[j][k]; // The V latent factor for this movie j
                    U[i][k] += LEARNING_RATE_UF * (error * mf - REGULARIZATION_UF * uf);
                    V[j][k] += LEARNING_RATE_MF * (error * (uf + sqrt_r*SumMW[i][k]) - REGULARIZATION_MF * mf);
                    tmpSum[k] += error * sqrt_r * mf;
                }
			}
			// Update the y factors for each movie a user rated
			for (int t = 0; t < sz; t++)
			{
				int j = movies_rated_by_user[i][t];
                for (int k = 0; k < LATENT_FACTORS; ++k) {
                    y[j][k] += LEARNING_RATE_Y * (tmpSum[k] - REGULARIZATION_Y * y[j][k]);
                }
			}
        }

        // Update SumMW for all users
        for (int i = 0; i < USER_SIZE; i++)
        {
			int sz = (movies_rated_by_user[i]).size();
			for (int k = 0; k < LATENT_FACTORS; k++)
			{
				double sumy = 0;
				// get the sum for all the movies rated by the user the
				// latent factors associated with the movie rated by the user
				for (int t = 0; t < sz; t++)
				{
					sumy += y[ movies_rated_by_user[i][t] ] [k];
				}
				SumMW[i][k] = sumy;
			}
		}
        // At end of epoch, print E_in, E_val
        // srand ( unsigned ( time(0) ) );
        // long rand_n = rand() % (TRAIN_SIZE-1000001);
        // E_in = get_err(U, V, user_matrix+rand_n, movie_matrix+rand_n,
                            // date_matrix+rand_n, rating_matrix+rand_n,
                            // 1000000, reg, SumMW,bu, bi, Bi_Bin, Tu, Alpha_u);
        // cout<<E_in<<endl;
        E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                              date_matrix_val, rating_matrix_val,
                              VALID_SIZE, reg,SumMW, bu, bi, Bi_Bin, Tu, Alpha_u);

        cout << endl << "E_in: " << E_in << "  E_val: " << E_val << endl;


        // If E_val doesn't decrease, stop early
        if (init_E_val <= E_val) {
            cout<<"E_val is increasing! Printing checkpoint"<<endl;
            // predict( U, V, SumMW, bu, bi,user_matrix_test, movie_matrix_test, date_matrix_test, Bi_Bin, Tu, Alpha_u, epoch);
            // checkpoint_U_V(U, V, epoch);
            break;
        }
        init_E_val = E_val;
        eta *= (0.9 + 0.1 * rand()/RAND_MAX);
    }
    cout << endl;
	// predict( U, V, SumMW, bu, bi,user_matrix_test, movie_matrix_test, date_matrix_test, Bi_Bin, Tu, Alpha_u, epoch);
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
    inFile.open(TRAIN_IN_FILENAME);
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

	populate_movies_to_array(user_matrix, movie_matrix, rating_matrix, date_matrix);

    // Read validation data
    cout << "Reading validation input." << endl;
    inFile.open(VALID_IN_FILENAME);
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

	// cout<<" Reading out test data " <<endl;
	// // Read in training data
    // ifstream inFile_test;
	// inFile_test.open(TEST_IN_FILENAME);
    // int* user_matrix_test = new int[TEST_SIZE];
    // short* movie_matrix_test = new short[TEST_SIZE];
    // short* date_matrix_test = new short[TEST_SIZE];
    // int garbage_zero_rating;
    // for (long i=0; i<TEST_SIZE; i++) {
    //     inFile_test >> user_matrix_test[i];
    //     inFile_test >> movie_matrix_test[i];
    //     inFile_test >> date_matrix_test[i];
    //     inFile_test >> garbage_zero_rating;
    //     if (i % 100 == 0) {
    //         cout << "\r" << to_string(i * 100 / TEST_SIZE) << "%%" << flush;
    //     }
    // }
    // cout << endl;
    // inFile_test.close();








	double reg_uf[] = {0.005, 0.01, 0.02, 0.06};
	double reg_mf[] = {0.007};
	double reg_y[]  = {0.015};
	double reg_mb[] = {0};
	double reg_ub[] = {0.005};
	double reg_bn[] = {0.0008};
	double reg_bu[] = {0.0006};
	// double[] eta_uf = {0.006};
	// double[] eta_mf = {0.011};
	// double[] eta_y  = {0.001};
	// double[] eta_mb = {0.003};
	// double[] eta_ub = {0.012};
	// double[] eta_bn = {0.0012};
	// double[] eta_bu = {0.012};

    outFile.open("Grid_Search_"+to_string(LATENT_FACTORS)+"_3.log");
	svd_ans result;

	double lowestRMSE = 100;
	double best_REGULARIZATION_UF = -1;
	double best_REGULARIZATION_MF = -1;
	double best_REGULARIZATION_Y  = -1;
	double best_REGULARIZATION_MB = -1;
	double best_REGULARIZATION_UB = -1;
	double best_REGULARIZATION_BINS = -1;
	double best_REGULARIZATION = -1;


	// for (int i6=0; i6<2; i6++) {
	// 	REGULARIZATION_BINS = reg_bn[i6];
	// 	for (int i5=0; i5<1; i5++) {
	// 		REGULARIZATION_UB = reg_ub[i5];
	// 		for (int i4=0; i4<1; i4++) {
	// 			REGULARIZATION_MB = reg_mb[i4];
	// 			for (int i3=0; i3<3; i3++) {
	// 				REGULARIZATION_Y = reg_y[i3];
	// 				for (int i2=0; i2<1; i2++) {
	// 					REGULARIZATION_MF = reg_mf[i2];
	// 					for (int i1=0; i1< 2; i1++) {
	// 						REGULARIZATION_UF = reg_uf[i1];
	// 						for (int i7=0; i7<1; i7++) {
	// 							REGULARIZATION = reg_bu[i7];

	for (int i1=0; i1<4; i1++) {
		REGULARIZATION_UF = reg_uf[i1];
	for (int i2=0; i2<1; i2++) {
		REGULARIZATION_MF = reg_mf[i2];
	for (int i3=0; i3<1; i3++) {
		REGULARIZATION_Y = reg_y[i3];
	for (int i4=0; i4<1; i4++) {
		REGULARIZATION_MB = reg_mb[i4];
	for (int i5=0; i5<1; i5++) {
		REGULARIZATION_UB = reg_ub[i5];
	for (int i6=0; i6<1; i6++) {
		REGULARIZATION_BINS = reg_bn[i6];
	for (int i7=0; i7<1; i7++) {
		REGULARIZATION = reg_bu[i7];

		// Create U matrix
	    double** U = new double* [USER_SIZE];
	    // Create V matrix
	    double** V = new double* [MOVIE_SIZE];
	    // Create y array
	    double** y = new double* [MOVIE_SIZE];
	    // Create the part of the y array that will be used for the gradient
	    double** SumMW = new double* [USER_SIZE];


	    // Create the user bias array
	    double* bu = new double [USER_SIZE];
	    // Create the movie bias array
	    double* bi = new double [MOVIE_SIZE];
	    // Create alpha array
	    double* Alpha_u = new double [USER_SIZE];
	    // Create bi_bin, which represents a movies ratings over time
	    double** Bi_Bin = new double* [MOVIE_SIZE];
	    // Create Tu, which will hold the mean movies, I will make index 0 be user 1
		double* Tu = new double [USER_SIZE];

	    // Initialize the matrices
	    for(int j = 0; j < MOVIE_SIZE; j++){
			bi[j] = 0.0;
	        V[j] = new double[LATENT_FACTORS];
	        y[j] = new double[LATENT_FACTORS];
	        Bi_Bin[j] = new double[BIN_NUMBER];
	        for(int k = 0; k < LATENT_FACTORS; k++){
	            srand ( unsigned ( time(0) ) );
	            V[j][k] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(LATENT_FACTORS);
	            y[j][k] = 0;
	        }
	    }

	    for (int i = 0; i < MOVIE_SIZE; i++)
	    {
			for (int k = 0; k < BIN_NUMBER; k++)
				{
					Bi_Bin[i][k] = 0;
				}
		}

	    for(int i = 0; i < USER_SIZE; i++){
			bu[i] = 0.0;
			Alpha_u[i] = 0.0;
	        U[i] = new double[LATENT_FACTORS];
	        SumMW[i] = new double[LATENT_FACTORS];
	        for(int k = 0; k < LATENT_FACTORS; k++){
	            U[i][k] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(LATENT_FACTORS);
	            SumMW[i][k] = 0.1 * (rand() / (RAND_MAX + 1.0)) / sqrt(LATENT_FACTORS);
	        }
	    }


	    // Train SVD
	    cout << "Training model." << endl;

		result = train_model_from_UV(LEARNING_RATE, REGULARIZATION,
	                                        user_matrix, movie_matrix,
	                                        date_matrix, rating_matrix,
	                                        user_matrix_val, movie_matrix_val,
	                                        date_matrix_val, rating_matrix_val,
	                                        U, V, y, SumMW,user_matrix_val,
	                                        movie_matrix_val, date_matrix_val,
	                                        bu, bi, Bi_Bin, Tu, Alpha_u);
		if (result.E_val < lowestRMSE) {
			best_REGULARIZATION_UF   = REGULARIZATION_UF;
			best_REGULARIZATION_MF   = REGULARIZATION_MF;
			best_REGULARIZATION_Y    = REGULARIZATION_Y;
			best_REGULARIZATION_MB   = REGULARIZATION_MB;
			best_REGULARIZATION_UB   = REGULARIZATION_UB;
			best_REGULARIZATION_BINS = REGULARIZATION_BINS;
			best_REGULARIZATION      = REGULARIZATION;
			lowestRMSE = result.E_val;
		}
		outFile << "REGULARIZATION_UF   " << REGULARIZATION_UF << endl;
		outFile << "REGULARIZATION_MF   " << REGULARIZATION_MF << endl;
		outFile << "REGULARIZATION_Y    " << REGULARIZATION_Y << endl;
		outFile << "REGULARIZATION_MB   " << REGULARIZATION_MB << endl;
		outFile << "REGULARIZATION_UB   " << REGULARIZATION_UB << endl;
		outFile << "REGULARIZATION_BINS " << REGULARIZATION_BINS << endl;
		outFile << "REGULARIZATION      " << REGULARIZATION << endl;
		outFile << "e_val: " << result.E_val << endl;
		outFile << "----------------------------" << endl << endl;


	    for (long r = 0; r < USER_SIZE;  r++) { delete[] U[r]; }
	    for (long r = 0; r < MOVIE_SIZE; r++) { delete[] V[r]; }
	    for (long r = 0; r < MOVIE_SIZE; r++) { delete[] y[r]; }
	    for (long r = 0; r < USER_SIZE;  r++) { delete[] SumMW[r]; }
	    for (long r = 0; r < MOVIE_SIZE;  r++) { delete[] Bi_Bin[r]; }

	    delete[] bi;
	    delete[] bu;
	    delete[] Alpha_u;
	    delete[] Tu;
	    delete[] U;
	    delete[] V;
	    delete[] y;
	    delete[] SumMW;
		delete[] Bi_Bin;
		}
		}
		}
		}
		}
		}
	}
	outFile << "BEST PARAMETERS" << endl;
	outFile << "REGULARIZATION_UF   " << best_REGULARIZATION_UF << endl;
	outFile << "REGULARIZATION_MF   " << best_REGULARIZATION_MF << endl;
	outFile << "REGULARIZATION_Y    " << best_REGULARIZATION_Y << endl;
	outFile << "REGULARIZATION_MB   " << best_REGULARIZATION_MB << endl;
	outFile << "REGULARIZATION_UB   " << best_REGULARIZATION_UB << endl;
	outFile << "REGULARIZATION_BINS " << best_REGULARIZATION_BINS << endl;
	outFile << "REGULARIZATION      " << best_REGULARIZATION << endl;
	outFile << "e_val: " << lowestRMSE << endl << endl;
	outFile.close();






    // svd_ans result = train_model_from_UV(eta, reg,
    //                                     user_matrix, movie_matrix,
    //                                     date_matrix, rating_matrix,
    //                                     user_matrix_val, movie_matrix_val,
    //                                     date_matrix_val, rating_matrix_val,
    //                                     U, V, y, SumMW,user_matrix_test,
    //                                     movie_matrix_test, date_matrix_test,
    //                                     bu, bi, Bi_Bin, Tu, Alpha_u);
	//
    // cout << "Final E_in: " << result.E_in << "  E_val: " << result.E_val << endl;

	// delete[] user_matrix_test;
    // delete[] movie_matrix_test;
    // delete[] date_matrix_test;
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
