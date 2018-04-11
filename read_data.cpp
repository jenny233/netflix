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

	map<int, vector<int>> users;
	for (int i = 0; i < SIZE; i++)
	{
		int user = rating_matrix[i][0];

		if (users.find(user) == users.end())
		{
			vector<int> movie;
			movie.push_back(rating_matrix[i][3]);
			users[user] = movie;
		}
		else
		{
			vector<int> movies;
			movies = users[user];
			movies.push_back(rating_matrix[i][3]);
			users[user] = movies;

		}
	}


	for (map<int, vector<int> >::iterator ii = users.begin(); ii != users.end(); ++ii) {
		cout << (*ii).first << ": ";
		vector <int> inVect = (*ii).second;
		for (unsigned j = 0; j<inVect.size(); j++) {
			cout << inVect[j] << " ";
		}
		cout << endl;
	}
	cout << "Map size: " << users.size() << endl;
	return 0;
}
