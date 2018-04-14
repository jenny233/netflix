#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#define USER_SIZE  500000
#define MOVIE_SIZE  5000
using namespace std;

vector<int>* adj_user = new vector<int>[USER_SIZE];
vector<int>* adj_movie = new vector<int>[MOVIE_SIZE];
int num_of_unique_movie;
int num_of_unique_user;

void array_pop(int size, int* rating_matrix[], vector<int> arr[], char pop)
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

	return ((user_average / (double)((adj_user[user]).size())) + (movie_average / (double)(adj_movie[movie]).size())) / 2.0;
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

double sum_bias(double arr[], int* rating_matrix[], char type)
{
	double total = 0;
	if (type == 'u')
	{
		for (int i = 0; i < USER_SIZE; i++)
			total += (arr[rating_matrix[i][0]]);
	}
	else
	{
		for (int i = 0; i < USER_SIZE; i++)
			total += (arr[rating_matrix[i][2]]);
	}
	return total;
}

double SVG_bias_u(double u_bias[], double m_bias[], int* rating_matrix[], double lamdba, double avg, int rating_index)
{
	double gradient  = 0.0;
	double learning_rate = 0.1;
	for (int i = 0; i < USER_SIZE; i++)
	{
		gradient += -2.0 * (rating_matrix[i][3] - avg - u_bias[rating_matrix[i][0]] - m_bias[rating_matrix[i][2]]) + lamdba * 2.0 * sum_bias(u_bias, rating_matrix, 'u');
	}
	u_bias[rating_matrix[rating_index][0]] -= learning_rate * gradient;
	return u_bias[rating_matrix[rating_index][0]];
}
double SVG_bias_m(double u_bias[], double m_bias[], int* rating_matrix[], double lamdba, double avg, int rating_index)
{
	double gradient = 0.0;
	double learning_rate = 0.1;
	for (int i = 0; i < USER_SIZE; i++)
	{
		gradient += -2.0 * (rating_matrix[i][3] - avg - u_bias[rating_matrix[i][0]] - m_bias[rating_matrix[i][2]]) + lamdba * 2.0 * sum_bias(m_bias, rating_matrix, 'm');
	}
	m_bias[rating_matrix[rating_index][2]] -= learning_rate * gradient;
	return m_bias[rating_matrix[rating_index][2]];
}

void gradient_descent_bias(double u_bias[], double m_bias[], int* rating_matrix[], double lamdba, double avg)
{
	double error;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < USER_SIZE; j++)
		{
			u_bias[rating_matrix[j][0]] = SVG_bias_u(u_bias, m_bias, rating_matrix, lamdba, avg, j);
			m_bias[rating_matrix[j][2]] = SVG_bias_m(u_bias, m_bias, rating_matrix, lamdba, avg, j);
		}
		cout << "Iteration " << i << endl;
	}
	for (int i = 0; i < USER_SIZE; ++i)
	{
		std::cout << u_bias[rating_matrix[i][0]] << endl;
		std::cout << m_bias[rating_matrix[i][2]] << endl;
	}
}

void checkpoint(double u_bias[], double m_bias[])
{
	ofstream outputFile;
	outputFile.open("bias_checkpoint.txt");
	outputFile << "movie_bias:" << endl;
	for (int i = 0; i < USER_SIZE; i++)
	{
		outputFile << u_bias[i] << endl;

	}
	outputFile << "movie_bias:" << endl<<endl;
	for (int i = 0; i < MOVIE_SIZE; i++)
	{
		outputFile << m_bias[i] << endl;

	}
	outputFile.close();
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
		total_average_movie_rating += (double)rating_matrix[i][3];

	}
	total_average_movie_rating = total_average_movie_rating / USER_SIZE;

	array_pop(USER_SIZE, rating_matrix, adj_user, 'u');
	array_pop(USER_SIZE, rating_matrix, adj_movie, 'm');

	double error1 = 0;
	double error2 = 0;
	/*
	for (int i = 0; i < USER_SIZE; ++i)
	{
		error1 += pow((double)rating_matrix[i][3] - baseline_pred(rating_matrix[i][0], rating_matrix[i][2]), 2.0);
		error2 += pow((double)rating_matrix[i][3] - baseline_pred_better(rating_matrix[i][0], rating_matrix[i][2], total_average_movie_rating), 2.0);
	}
	cout << error1 << endl;
	cout << error2 << endl;
	*/
	double* bias_user = new double[USER_SIZE];
	double* bias_movie = new double[MOVIE_SIZE];
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
	checkpoint(bias_user,bias_movie);
	cout << "goodluck" << endl;
	gradient_descent_bias(bias_user, bias_movie,  rating_matrix,  10, total_average_movie_rating);
	checkpoint(bias_user, bias_movie);
	std::cout << num_of_unique_movie << endl;
	std::cin.get();
	for (int i = 0; i < USER_SIZE; ++i)
		delete[] rating_matrix[i];
	delete[] rating_matrix;
	delete[] bias_movie;
	delete[] bias_user;
	delete[] adj_user;
	return 0;
}
