// 卷积运算
#include <maths_convolution.h>


// 采用数组来求卷积，而不是用vector，速度要快10%！
Array3Dd convolution(Array3Dd X, const Array2Dd &Ker, string shape)
{
	// 'full' - (default) returns the full N - D convolution
	// 'valid' - returns only the part of the result that can be
	//          computed without assuming zero - padded arrays.
	//          size(C, k) = max([nak - max(0, nbk - 1)], 0).

	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	if (X.size() <= 0)
	{
		cout << "Array3Dd is wrong!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	if (shape == "full")
	{
		X.expand_to_full_size(Ker_col, Ker_row);
	}

	int X_page = X.size();
	int X_row = X.at(0).at(0).size();
	int X_col = X.at(0).size();

	int i, j, k;

	if (shape == "valid" && (X_row < Ker_row || X_col < Ker_col))
	{
		cout << "X size is smaller than Ker size!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	// 创建卷积结果输出变量conv并初始化为0
	int conv_row = X.at(0).at(0).size() - Ker.at(0).size() + 1;
	int conv_col = X.at(0).size() - Ker.size() + 1;
	Array3Dd convn(X_page, conv_col, conv_row, 0);

	double *arr_X = new double[X_page * X_row * X_col]();
	double *arr_Ker = new double[Ker_row * Ker_col]();

	for (i = 0; i < X_page; i++)
	{
		for (j = 0; j < X_row; j++)
		{
			for (k = 0; k < X_col; k++)
			{
				// 对arr_X赋值
				arr_X[i * (X_row * X_col) + j * X_col + k] = X.at(i).at(k).at(j);

				// 对arr_Ker赋值
				if ((i == 0) && (j < Ker_row) && (k < Ker_col))
				{
					// x,y向同时翻转
					arr_Ker[j * Ker_col + k] = Ker.at(Ker_col - 1 - k).at(Ker_row - 1 - j);
				}
			}
		}
	}

	int row, col;
	for (i = 0; i < X_page; i++)
	{
		for (j = 0; j < conv_row; j++)
		{
			for (k = 0; k < conv_col; k++)
			{
				// 计算卷积矩阵第(j,k)点的值
				double sum_ijk = 0;
				for (row = j; row < j + Ker_row; row++)
				{
					for (col = k; col < k + Ker_col; col++)
					{
						sum_ijk += arr_X[i * (X_row * X_col) + row * X_col + col] * arr_Ker[(row - j) * Ker_col + (col - k)];
					}
				}
				convn.at(i).at(k).at(j) = sum_ijk;
			}
		}
	}

	delete[] arr_X;
	delete[] arr_Ker;

	return convn;
}


// 若size(a, 3) = size(b, 3), 则上式输出第三维为1, 表示参与训练样本的叠加和(批处理算法), 结果要对样本数做平均
// 就是对每一幅图像做卷积，然后加起来
Array2Dd convolution(const Array3Dd &X, const Array3Dd &Ker, string shape)
{
	// 'full' - (default) returns the full N - D convolution
	// 'valid' - returns only the part of the result that can be
	//          computed without assuming zero - padded arrays.
	//          size(C, k) = max([nak - max(0, nbk - 1)], 0).

	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	int page_X = X.size();
	int page_Ker = Ker.size();

	if (page_X != page_Ker)
	{
		cout << "page size not equal!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	Array2Dd sum;

	for (int i = 0; i < page_X; ++i)
	{
		sum.add(convolution(X.at(i), Ker.at(i), shape));
	}

	return sum;
}


// 采用数组来求卷积，而不是用vector，速度要快30倍！
Array2Dd convolution(Array2Dd X, Array2Dd Ker, string shape)
{
	// 'full' - (default) returns the full N - D convolution
	// 'valid' - returns only the part of the result that can be
	//          computed without assuming zero - padded arrays.
	//          size(C, k) = max([nak - max(0, nbk - 1)], 0).

	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	if (shape == "full")
	{
		X.expand_to_full_size(Ker_col, Ker_row);
	}

	int X_row = X.at(0).size();
	int X_col = X.size();

	if (shape == "valid" && (X_row < Ker_row || X_col < Ker_col))
	{
		cout << "X size is smaller than Ker size!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	// 创建卷积结果输出变量conv并初始化为0
	int conv_row = X.at(0).size() - Ker.at(0).size() + 1;
	int conv_col = X.size() - Ker.size() + 1;
	Array2Dd conv(conv_col, conv_row, 0);

	double *arr_X = new double[X_row * X_col]();
	double *arr_Ker = new double[Ker_row * Ker_col]();

	int i, j;

	for (i = 0; i < X_row; i++)
	{
		for (j = 0; j < X_col; j++)
		{
			// 对arr_X赋值
			arr_X[i * X_col + j] = X.at(j).at(i);

			// 对arr_Ker赋值
			if ((i < Ker_row) && (j < Ker_col))
			{
				// x,y向同时翻转
				arr_Ker[i * Ker_col + j] = Ker.at(Ker_col - 1 - j).at(Ker_row - 1 - i);
			}
		}
	}

	int row, col;
	for (i = 0; i < conv_row; i++)
	{
		for (j = 0; j < conv_col; j++)
		{
			// 计算卷积矩阵第(i,j)点的值
			double sum_ij = 0;
			for (row = i; row < i + Ker_row; row++)
			{
				for (col = j; col < j + Ker_col; col++)
				{
					sum_ij += arr_X[row * X_col + col] * arr_Ker[(row - i) * Ker_col + (col - j)];
				}
			}
			conv.at(j).at(i) = sum_ij;
		}
	}

	delete[] arr_X;
	delete[] arr_Ker;

	return conv;
}

