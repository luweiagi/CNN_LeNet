// 卷积运算
#include <maths_convolution.h>

/*
Array3Dd convolution_n_dim(const Array3Dd &X, const Array2Dd &Ker)
{
	Array3Dd convn;

	for (int i = 0; i < X.size(); i++)
	{
		Array2Dd conv = convolution_one_dim_fast(X.at(i), Ker);
		convn.push_back(conv);
	}

	return convn;
}
*/

// 采用数组来求卷积，而不是用vector，速度要快10%！
Array3Dd convolution_n_dim(const Array3Dd &X, const Array2Dd &Ker)
{
	int X_page = X.size();
	int X_row = X.at(0).at(0).size();
	int X_col = X.at(0).size();

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	int i, j, k;

	// 如果要卷积的对象的尺寸小于卷积核的尺寸，则将卷积对象的尺寸扩展（复制边缘）到卷积和的尺寸。
	if (X_row < Ker_row || X_col < Ker_col)
	{
		cout << "X size is smaller than Ker size!" << endl << "convolution_n_dim() failed!" << endl;
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
				if((i == 0) && (j < Ker_row) && (k < Ker_col))
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


/*
Array2Dd convolution_one_dim(Array2Dd X, Array2Dd Ker)
{
	int X_row = X.at(0).size();
	int X_col = X.size();

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	// 如果要卷积的对象的尺寸小于卷积核的尺寸，则将卷积对象的尺寸扩展（复制边缘）到卷积和的尺寸。
	if (X_row < Ker_row || X_col < Ker_col)
	{
		change_X_size_to_fit_Ker(X, Ker);
	}

	// 重新获取X的尺寸
	X_row = X.at(0).size();
	X_col = X.size();

	// 创建卷积结果输出变量conv并初始化为0
	int conv_row = X.at(0).size() - Ker.at(0).size() + 1;
	int conv_col = X.size() - Ker.size() + 1;
	Array2Dd conv(conv_col, conv_row, 0);

	// 对卷积核进行xy轴向的翻转
	Ker.flip_xy();

	// 这是用来和卷积核相乘的那一块X的区域
	Array2Dd X_patch;

	// 开始进行卷积
	for (int i = 0; i < conv_col; i++)
	{
		for (int j = 0; j < conv_row; j++)
		{
			X_patch.get_specific_patch(X, Ker_col, Ker_row, i, j);
			// conv[i][j] = sum(X_patch .* Ker);
			conv.at(i).at(j) = (X_patch * Ker).sum();
		}
	}

	return conv;
}
*/


// 采用数组来求卷积，而不是用vector，速度要快30倍！
Array2Dd convolution_one_dim(Array2Dd X, Array2Dd Ker)
{
	int X_row = X.at(0).size();
	int X_col = X.size();

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	// 如果要卷积的对象的尺寸小于卷积核的尺寸，则将卷积对象的尺寸扩展（复制边缘）到卷积和的尺寸。
	if (X_row < Ker_row || X_col < Ker_col)
	{
		cout << "X size is smaller than Ker size!" << endl << "convolution_n_dim() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	// 创建卷积结果输出变量conv并初始化为0
	int conv_row = X.at(0).size() - Ker.at(0).size() + 1;
	int conv_col = X.size() - Ker.size() + 1;
	Array2Dd conv(conv_col, conv_row, 0);

	double *arr_X = new double[X_row * X_col] ();
	double *arr_Ker = new double[Ker_row * Ker_col] ();

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
				arr_Ker[i * Ker_col + j] = Ker.at(Ker_col -1 - j).at(Ker_row - 1 - i);
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

