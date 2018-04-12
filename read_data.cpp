#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#define USER_SIZE  32000
#define MOVIE_SIZE  32000
using namespace std;

map<int, double> users;
map<int, double> movies;

void populate_hashmap(int size, int rating_matrix[][4], map<int, double> &temp_map, char type)
{
	if (type = 'u')
	{
		for (int i = 0; i < size; i++)
		{
			int user = rating_matrix[i][0];
			if (temp_map.find(user) == temp_map.end())
			{

				temp_map[user] = (double)rating_matrix[i][3];
			}
			else
			{
				double temp = temp_map.find(user)->second;
				temp_map[user] = ((double)rating_matrix[i][3] + temp) / 2.0;
			}
		}
	}
	else
	{
		for (int i = 0; i < size; i++)
		{
			int user = rating_matrix[i][3];

			if (temp_map.find(user) == temp_map.end())
			{

				temp_map[user] = (double)rating_matrix[i][0];
			}
			else
			{
				double temp = temp_map.find(user)->second;
				temp_map[user] = ((double)rating_matrix[i][0] + temp) / 2.0;
			}
		}
	}

}

double baseline_pred(int user, int movie)
{
	return (users[user] + movies[movie]) / 2.0;
}

int main()
{
	ifstream inFile;
	inFile.open("all.dta");
	if (!inFile) {
		cout << "File not opened." << endl;
		exit(1);
	}

	// The matrix of all of the ratings
	int rating_matrix_users[USER_SIZE][4];
	int rating_matrix_movies[MOVIE_SIZE][4];


	for (int i = 0; i < USER_SIZE; i++) {
		for (int j = 0; j<4; j++) {
			inFile >> rating_matrix_users[i][j];
		}
	}

	populate_hashmap(USER_SIZE, rating_matrix_users, users, 'u');


	for (int i = 0; i<MOVIE_SIZE; i++) {
		for (int j = 0; j<4; j++) {
			inFile >> rating_matrix_movies[i][j];
		}
	}
	populate_hashmap(MOVIE_SIZE, rating_matrix_movies, movies,'m');


	
	cout << "Map size: " << users.size() << endl;
	cout << " Map movie size: " << movies.size() << endl;
	/*
	for (auto elem : movies)
	{
		std::cout << elem.first << " " << elem.second << "\n";
	}
	for (auto elem : users)
	{
		std::cout << elem.first << " " << elem.second << "\n";
	}
	*/

 	return 0;
}
