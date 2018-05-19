#include "timeSVD++.h"
#include <string>
#include <cmath>
#define LATENT_FACTORS 30
#define REGULARIZATION 0.007
#define LEARNING_RATE  0.005
#define reg_alpha  0.00001
#define eta_alpha  0.0004
#define MAX_EPOCH      400
#define PRED_FILENAME ("../predictions_svd++_" + to_string(LATENT_FACTORS) + "lf_5_16.dta")

/*
 * IMPORTANT:
 * All arrays and matrices hold user and movie indices instead of ids
 * userId = user index + 1
 * Except for the data read from the training, validation, testing files.
 */

// Two arrays for holding the user and movie bias
double* user_bias = new double[USER_SIZE];
double* movie_bias = new double[MOVIE_SIZE];

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
	cout<<endl;
    // Return the RMSE
    return sqrt(err / size);
}


void predict(double** U, double** V, double** SumMW, double* bu,
			double* bi, int* user_matrix_test, short* movie_matrix_test,
			short* date_matrix_test, double** Bi_Bin, double* Tu, 
			double* Alpha_u)
{
	
	cout<<"printing checkpoint"<<endl;
    ofstream outFile;
	outFile.open(PRED_FILENAME);
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

    double E_in, E_val;
    double init_E_in = 100, init_E_val = 100;
	



    // Initialize timers
    system_clock::time_point start_time, end_time;

    // Stochastic gradient descent
    ofstream outFile;
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
	
	
    for (int epoch = 0; epoch < MAX_EPOCH; epoch++) {

        cout << "Epoch " << epoch << ":" << endl;
        start_time = system_clock::now();
		
        // Checkpoint every 10 epochs
		if (epoch % 10 == 0 ) {
            predict( U, V, SumMW, user_bias, movie_bias,user_matrix_test, movie_matrix_test, date_matrix_test, Bi_Bin, Tu, Alpha_u);
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
                
                bu[i] += reg * (error - eta * bu[i]);
                bi[j] += reg * (error - eta * bi[j]);
                Bi_Bin[j][CalBin(time)] += reg * (error - eta * Bi_Bin[j][CalBin(time)]);
                Alpha_u[i] += reg_alpha * (error * CalDev(i,time,Tu) - eta_alpha * Alpha_u[i]);
                Bu_t[i][time] += reg * ( error - eta * Bu_t[i][time]);                
                
                // Update U for user i and V for movie j
                for(int k = 0; k < LATENT_FACTORS; k++){
                    double uf = U[i][k]; // The U latent factor for this user i
                    double mf = V[j][k]; // The V latent factor for this movie j
                    U[i][k] += eta * (error * mf - reg * uf);
                    V[j][k] += eta * (error * (uf + sqrt_r*SumMW[i][k]) - reg * mf);
                    tmpSum[k] += error * sqrt_r * mf;
                }
			}
			// Update the y factors for each movie a user rated
			for (int t = 0; t < sz; t++)
			{
				int j = movies_rated_by_user[i][t];
                for (int k = 0; k < LATENT_FACTORS; ++k) {
                    y[j][k] += eta * (tmpSum[k] - reg * y[j][k]);
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
        srand ( unsigned ( time(0) ) );
        long rand_n = rand() % (TRAIN_SIZE-1000001);
        E_in = get_err(U, V, user_matrix+rand_n, movie_matrix+rand_n,
                            date_matrix+rand_n, rating_matrix+rand_n,
                            1000000, reg, SumMW,bu, bi, Bi_Bin, Tu, Alpha_u);
        cout<<E_in<<endl;
        E_val = get_err(U, V, user_matrix_val, movie_matrix_val,
                              date_matrix_val, rating_matrix_val,
                              VALID_SIZE, reg,SumMW, bu, bi, Bi_Bin, Tu, Alpha_u);

        cout << endl << "E_in: " << E_in << "  E_val: " << E_val << endl;


        // If E_val doesn't decrease, stop early
        if (init_E_val <= E_val) {
            cout<<"E_val is increasing! Printing checkpoint"<<endl;
            checkpoint_U_V(U, V, epoch);
            break;
        }
        init_E_val = E_val;
        eta *= (0.9 + 0.1 * rand()/RAND_MAX);
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
    inFile_bias.close();

	populate_movies_to_array(user_matrix, movie_matrix, rating_matrix, date_matrix);

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

    // MatrixXd U = MatrixXd::Random(USER_SIZE, LATENT_FACTORS);
    // MatrixXd V = MatrixXd::Random(LATENT_FACTORS, MOVIE_SIZE);
	// MatrixXd y = MatrixXd::Random(MOVIE_SIZE, LATENT_FACTORS);
	// MatrixXd SumMW = MatrixXd::Random(USER_SIZE, LATENT_FACTORS);

    svd_ans result = train_model_from_UV(eta, reg,
                                        user_matrix, movie_matrix,
                                        date_matrix, rating_matrix,
                                        user_matrix_val, movie_matrix_val,
                                        date_matrix_val, rating_matrix_val,
                                        U, V, y, SumMW,user_matrix_test,
                                        movie_matrix_test, date_matrix_test,
                                        bu, bi, Bi_Bin, Tu, Alpha_u);

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
    for (long r = 0; r < MOVIE_SIZE; r++) { delete[] y[r]; }
    for (long r = 0; r < USER_SIZE;  r++) { delete[] SumMW[r]; }
    for (long r = 0; r < USER_SIZE;  r++) { delete[] Bi_Bin[r]; }
    
    delete[] bi;
    delete[] bu;
    delete[] Alpha_u;
    delete[] Tu;
    delete[] U;
    delete[] V;
    delete[] y;
    delete[] SumMW;

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
