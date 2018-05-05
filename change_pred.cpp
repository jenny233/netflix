#include "SVD_with_bias.hpp"
#include <string>
#define PRED_FILENAME "../predictions_5_2_100lf.dta"
#define LANTENT_FACTORS 20
void predict(int M, int N, int K) {

    // Initialize
    MatrixXd U(M, K);
    MatrixXd V(K, N);
    ifstream inFile;
    ofstream outFile;
    int* user_matrix_test = new int[TEST_SIZE];
    short* movie_matrix_test = new short[TEST_SIZE];
    short* date_matrix_test = new short[TEST_SIZE];

    short* date_matrix_val = new short[VALID_SIZE];
    double* rating_matrix_val = new double[VALID_SIZE];


	// IO

	ifstream inFile_bias;
    // Two arrays for holding the user and movie bias
    double* user_bias = new double[USER_SIZE + 1];
    double* movie_bias = new double[MOVIE_SIZE + 1];
	// Read in bias /////
	inFile_bias.open("../bias_checkpoint.txt");
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
    inFile_bias.close();
    // Read in U
    cout << "Reading U matrix." << endl;
    inFile.open("../svd_U_matrix_20lf_50ep.txt");
    if (!inFile) {
        std::cout << "File not opened." << endl;
        exit(1);
    }
    for (long r = 0; r < M; r++) {
        for (int c = 0; c < K; c++) {
            inFile >> U(r, c);
        }
    }
    inFile.close();

    // Read in V
    cout << "Reading V matrix." << endl;
    inFile.open("../svd_V_matrix_20lf_50ep.txt");
    if (!inFile) {
        std::cout << "File not opened." << endl;
        exit(1);
    }
    for (long r = 0; r < K; r++) {
        for (int c = 0; c < N; c++) {
            inFile >> V(r, c);
        }
    }
    inFile.close();



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
        if (i % 100 == 0) {
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
        prediction = prediction + TRAINING_DATA_AVERAGE + user_bias[i] + movie_bias[j];
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
int main()
{
	 predict(USER_SIZE, MOVIE_SIZE, LANTENT_FACTORS);
}
