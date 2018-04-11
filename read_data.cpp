#include <iostream>
#include <fstream>
using namespace std;

int main()
{
	ifstream inFile;
    inFile.open("mu/all.dta");
	if (!inFile) {
		cout << "File not opened." << endl;
		exit(1);
	}

	// The matrix of all of the ratings
	int rating_matrix[10][4];


    for (int i=0; i<10; i++) {
		for (int j=0; j<4; j++) {
			inFile >> rating_matrix[i][j];
		}
    }
	inFile.close();
	
	for (int i=0; i<10; i++) {
		for (int j=0; j<4; j++) {
			cout << rating_matrix[i][j] << " ";
		}
		cout << endl;
    }
	return 0;
}
