#include <maths_down_sample.h>
#include <maths.h>


// 采样的尺寸变化规则是：5x5的图片，以2的倍率降采样，其实相当于4x4的图片来降采样。
Array3Dd down_sample(const Array3Dd &vector_array, const int sample_num, down_sample_type sample_type)
{
	if (vector_array.size() <= 0)
	{
		cout << "vector_array is empty!" << endl << "down_sample() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	switch (sample_type)
	{
		case MeanPooling:
		{
			return down_sample_mean_pooling(vector_array, sample_num);
			break;
		}
		case MaxPooling:
		{
			return down_sample_max_pooling(vector_array, sample_num);
			break;
		}
		case StochasticPooling:
		{
			return down_sample_stochastic_pooling(vector_array, sample_num);
			break;
		}
		default:
		{
			cout << "down sample type is not defined yet!" << endl << "down_sample() failed!" << endl;
			Array3Dd temp;
			return temp;
			break;
		}
	}
}


Array3Dd down_sample_mean_pooling(const Array3Dd &vector_array, const int sample_num)
{
	if (vector_array.size() <= 0)
	{
		cout << "vector_array is empty!" << endl << "down_sample_mean_pooling() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	Array2Dd mean_pooling_convolution_kernel(sample_num, sample_num, double(1.0 / pow(sample_num, 2)));

	Array3Dd Z = convolution(vector_array, mean_pooling_convolution_kernel, "valid");

	Array3Dd Y = Z.sampling(sample_num);

	return Y;
}


Array3Dd down_sample_max_pooling(const Array3Dd &vector_array, const int sample_num)
{
	if (vector_array.size() <= 0)
	{
		cout << "vector_array is empty!" << endl << "down_sample_max_pooling() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	cout << "MaxPooling type down sample is not defined yet!" << endl << "down_sample_max_pooling() has stoped!" << endl;
	Array3Dd temp;
	return temp;
}


Array3Dd down_sample_stochastic_pooling(const Array3Dd &vector_array, const int sample_num)
{
	if (vector_array.size() <= 0)
	{
		cout << "vector_array is empty!" << endl << "down_sample_stochastic_pooling() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	cout << "StochasticPooling type down sample is not defined yet!" << endl << "down_sample_stochastic_pooling() has stoped!" << endl;
	Array3Dd temp;
	return temp;
}


Array3Dd up_sample(const Array3Dd &vector_array3D, const int sample_num, down_sample_type sample_type)
{
	if (vector_array3D.size() <= 0)
	{
		cout << "vector_array3D is empty!" << endl << "up_sample() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	switch (sample_type)
	{
		case MeanPooling:
		{
			return up_sample_mean_pooling(vector_array3D, sample_num);
			break;
		}
		case MaxPooling:
		case StochasticPooling:
		default:
		{
			cout << "down sample type is not defined yet!" << endl << "down_sample() failed!" << endl;
			Array3Dd temp;
			return temp;
			break;
		}
	}
}


Array3Dd up_sample_mean_pooling(const Array3Dd &vector_array3D, const int sample_num)
{

	int page = vector_array3D.size();
	int col = vector_array3D.at(0).size();
	int row = vector_array3D.at(0).at(0).size();

	Array3Dd vec_array3D;
	Array2Dd vec_array2D;
	vector<double> vec;
	int i,j,k,m,n;

	for (i = 0; i < page; i++)
	{
		vec_array2D.clear();
		for (j = 0; j < col; j++)
		{
			for (k = 0; k < sample_num; k++)
			{
				vec.clear();
				for (m = 0; m < row; m++)
				{
					double vec_val = vector_array3D.at(i).at(j).at(m);
					for (n = 0; n < sample_num; n++)
					{
						vec.push_back(vec_val);
					}
				}
				vec_array2D.push_back(vec);
			}
		}
		vec_array3D.push_back(vec_array2D);
	}

	return vec_array3D;
}


