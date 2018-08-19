#include <maths_matrix.h>
#include <iostream>

using namespace std;

void print_matrix2x2(unsigned char **array, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << array[i][j] << ' ';
		}

		cout << endl;
	}
}
