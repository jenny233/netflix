#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

function kfold(int data, number of splits )
{
	/*
	* Inputs: data (or data indices) has been shuffled, and the number of splits
	* Output: nothing
	* This function performs kfold validation on data.
	*/

	// split
	
	// train on every k-1 combination, then validate
	for (int i = 0; i < splits; i++)
	{
		// train SVD on data - split i 

		//

		/* validate on split i */

		// predict ratings for split i

		// save predicted ratings for split i

		// check ratings for split i

		
		printf ("The validation error of split %i is %f .\n", i, error )
	}


	// average the predicted ratings from each split

	// check ratings

	// return averaged ratings, and error




}