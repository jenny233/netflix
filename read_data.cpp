#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#define SIZE  10000
using namespace std;

int main()
{
	ifstream inFile;
	inFile.open("all.dta");
	if (!inFile) {
		cout << "File not opened." << endl;
		exit(1);
	}
	
	// The matrix of all of the ratings
	int rating_matrix[SIZE][4];


	for (int i = 0; i<SIZE; i++) {
		for (int j = 0; j<4; j++) {
			inFile >> rating_matrix[i][j];
		}
	}

	map<int, double> users;
	for (int i = 0; i < SIZE; i++)
	{
		int user = rating_matrix[i][0];

		if (users.find(user) == users.end())
		{
			
			users[user] = (double) rating_matrix[i][3];
		}
		else
		{
			double temp = users[user];
			users[user] = ((double)rating_matrix[i][3] + temp) / 2.0;
		}
	}
	
	return 0;
}
