#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#define USER_SIZE  1048576
#define MOVIE_SIZE  17000
using namespace std;

vector<int>* adj_user = new vector<int> [USER_SIZE];
vector<int>* adj_movie = new vector<int>[MOVIE_SIZE];
int num_of_unique_movie;
int num_of_unique_user;

void array_pop(int size, int* rating_matrix[], vector<int> arr [], char pop)
{
	int counter = 0;
	for (int i = 0; i < size; i++)
	{
		int type;
		if (pop == 'u')
			type = rating_matrix[i][0];
		else
			type = rating_matrix[i][2];
		int rating = rating_matrix[i][3];
		if ((arr[type])[0] == 0)
		{
			vector<int> r;
			r.push_back(rating);
			arr[type] = r;
			counter++;
		}
		else
		{
			(arr[type]).push_back(rating);
		}

	}
	if (pop == 'u')
		num_of_unique_user = counter;
	else 
		num_of_unique_movie = counter;

}

double baseline_pred(int user, int movie)
{
	/* Finds more simple predictor of movie baselines */
	double user_average = 0.0;
	double movie_average = 0.0;
	for (unsigned int i = 0; i < (adj_user[user]).size(); i++)
		user_average += (double)(adj_user[user])[i];

	for (unsigned int i = 0; i < (adj_movie[movie]).size(); i++)
		movie_average += (double)(adj_movie[movie])[i];

	return ((user_average/ (double) ((adj_user[user]).size())) + (movie_average / (double)(adj_movie[movie]).size())) / 2.0;
}

double baseline_pred_better(int user, int movie, double avg)
{
	double total = 0;
	double b_i, b_u;
	double lamdba1 = 25.0;
	double lamdba2 = 10.0;

	for (unsigned int i = 0; i < (adj_movie[movie]).size(); i++)
		total += (double)(adj_movie[movie])[i] - avg;
	b_i = total / (lamdba1 + ((adj_movie[movie]).size()));
	total = 0;
	for (unsigned int i = 0; i < (adj_user[user].size()); i++)
		total += (double)(adj_user[user])[i] - avg - b_i;
	b_u = total / (lamdba2 + ((adj_user[user]).size()));
	return (b_u + b_i + avg);
}

vector<double> find_bias(int user, int movie, double avg)
{
	double total = 0;
	double b_i, b_u;
	double lamdba1 = 25.0;
	double lamdba2 = 10.0;

	for (unsigned int i = 0; i < (adj_movie[movie]).size(); i++)
		total += (double)(adj_movie[movie])[i] - avg;

	b_i = total / (lamdba1 + ((adj_movie[movie]).size()));

	double total2 = 0;

	for (unsigned int i = 0; i < (adj_user[user]).size(); i++)
		total2 += (double)((adj_user[user])[i]) - avg - b_i;

	b_u = total2 / (lamdba2 + ((adj_user[user]).size()));
	vector <double> bias = { b_i,b_u };
	return bias;
}

double sum_bias(double arr[], int* rating_matrix[],  char type)
{
	double total = 0;
	if (type == 'u')
	{
		for (int i = 0; i < USER_SIZE; i++)
			total += pow(arr[rating_matrix[i][0]],2.0);
	}
	else
	{
		for (int i = 0; i < USER_SIZE; i++)
			total += pow(arr[rating_matrix[i][2]], 2.0);
	}
	return total;
}

double bias_error_function(double u_bias[], double m_bias[], int* rating_matrix[], double lamdba, double avg)
{
	double error = 0.0;
	for (int i = 0; i < USER_SIZE; i++)
	{
		error += pow((rating_matrix[i][3] - avg - u_bias[rating_matrix[i][0]] - m_bias[rating_matrix[i][2]]), 2.0) + lamdba * (sum_bias(u_bias, rating_matrix,'u') + sum_bias(m_bias,rating_matrix,'m'));
	}
	return error;
}
void do_learning()
{
	//TODO
}
void gradient_descent_bias(double u_bias [], double m_bias [], int* rating_matrix[], double lamdba, double avg)
{
	double error;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < USER_SIZE; j++)
		{
			error = bias_error_function(u_bias, m_bias, rating_matrix, lamdba, avg);
			do_learning();
		}
	}
}

int main()
{
	for (int i = 0; i < USER_SIZE; i++)
	{
		adj_user[i] = { 0 };
	}
	for (int i = 0; i < MOVIE_SIZE; i++)
	{
		adj_movie[i] = { 0 };
	}
	
	ifstream inFile;
	inFile.open("all.dta");
	if (!inFile) {
		std::cout << "File not opened." << endl;
		exit(1);
	}

	
	// The matrix of all of the ratings
	int **rating_matrix = new int*[USER_SIZE];
	for (int i = 0; i < USER_SIZE; ++i) {
		rating_matrix[i] = new int[4];
	}

	

	double total_average_movie_rating = 0;
	for (int i = 0; i < USER_SIZE; i++) {
		for (int j = 0; j<4; j++) {
			inFile >> rating_matrix[i][j];
		}
		total_average_movie_rating += (double) rating_matrix[i][3];

	}
	total_average_movie_rating = total_average_movie_rating / USER_SIZE;

	array_pop(USER_SIZE, rating_matrix, adj_user, 'u');
	array_pop(MOVIE_SIZE, rating_matrix, adj_movie, 'm');

	double error1 = 0;
	double error2 = 0;
	for (int i = 0; i < USER_SIZE; ++i)
	{
		error1 += pow((double) rating_matrix[i][3] - baseline_pred(rating_matrix[i][0], rating_matrix[i][2]), 2.0);
		error2 += pow((double) rating_matrix[i][3] - baseline_pred_better(rating_matrix[i][0], rating_matrix[i][2],total_average_movie_rating),2.0);
	}
	cout << error1 << endl;
	cout << error2 << endl;

	double* bias_user = new double [USER_SIZE];
	double* bias_movie = new double [MOVIE_SIZE];
	for (int i = 0; i < USER_SIZE; i++)
	{
		bias_user[i] = { 0 };
	}
	for (int i = 0; i < MOVIE_SIZE; i++)
	{
		bias_movie[i] = { 0 };
	}
	vector <double> bias;
	for (int i = 0; i < USER_SIZE; ++i)
	{
		bias = find_bias(rating_matrix[i][0], rating_matrix[i][2], total_average_movie_rating);

		if (bias_user[rating_matrix[i][0]] != 0)
		{
			bias_user[rating_matrix[i][0]] = (bias[0] + bias_user[rating_matrix[i][0]]) / 2.0;
		}
		else
			bias_user[rating_matrix[i][0]] = bias[0];

		if (bias_movie[rating_matrix[i][2]] != 0)
		{
			bias_movie[rating_matrix[i][2]] = (bias[1] + bias_movie[rating_matrix[i][2]]) / 2.0;
		}
		else
			bias_movie[rating_matrix[i][2]] = bias[1];
	}




	cin.get();
	for (int i = 0; i < USER_SIZE; ++i)
		delete[] rating_matrix[i];
	delete[] rating_matrix;
	delete[] bias_movie;
	delete[] bias_user;
	delete[] adj_user;
	delete[] adj_movie;

 	return 0;
}
