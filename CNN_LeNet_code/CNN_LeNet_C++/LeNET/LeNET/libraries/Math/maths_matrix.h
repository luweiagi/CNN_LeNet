#pragma once

template <typename T>
void print_matrix(T *array, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << (int)*(array + i * col * sizeof(T) + j) << ' ';
		}
		cout << endl;
	}
}

