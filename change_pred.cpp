#include "SVD.hpp"
#include <string>
#define PRED_FILENAME "predictions_5_2_100lf.dta"
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

    // Read in U
    cout << "Reading U matrix." << endl;
    inFile.open("svd_U_matrix_5_1_1.txt");
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
    inFile.open("svd_V_matrix_5_1_1.txt");
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
