#include <CNN.h>
#include <maths.h>
#include <time.h>
#include <iostream>

using namespace std;


// CNN网络，训练
void CNN::train(const vector<Mat> &train_x, const vector<vector<double>> &train_y)
{
	cout << "begin to train" << endl;

	if (train_x.size() != train_y.size())
	{
		cout << "train_x size is not same as train_y size!" << endl << "stop CNN trainning!" << endl;
		return;
	}

	int m = train_x.size();// 训练样本个数
	int numbatches = ceil(m / _batchsize);// "训练集整体迭代一次" 网络权值更新的次数

	// 对训练集整体迭代次数做循环
	for (int I = 0; I < _epochs; I++)
	{
		// 显示进度
		cout << "epoch " << I+1 << "/" << _epochs << endl;

		clock_t tic = clock(); //获取毫秒级数目
		
		// 打乱训练样本顺序，实现洗牌的功能
		const vector<int> kk = randperm_vector(m);

		// *********************************************************************************************** //
		// 对"训练集整体迭代一次"网络权值更新的次数做循环

		double mse = 0;// 当次训练集整体迭代时的均方误差

		// 整体训练一次网络更新的次数,
		// 即：假设训练样本数为1000个，批处理数为10个，那整体训练一次就得1000/10=100次
		for (int L = 0; L < numbatches; L++)
		{
			// 取出打乱顺序后的batchsize个样本和对应的标签
			vector<Mat> batch_train_x;
			vector<vector<double>> batch_train_y;

			for (int i = L*_batchsize; i < min((L + 1)*_batchsize, m); i++)
			{
				batch_train_x.push_back(train_x.at(kk.at(i)));
				batch_train_y.push_back(train_y.at(kk.at(i)));
			}
			// 显示当前正在处理的批量图片（只显示10张，当批处理图片的数量小于10时会报错，这时需要修改显示的张数）
			//string window_title = "Images from " + to_string(L*_batchsize) + " to " + to_string(min((L + 1)*_batchsize, m));
			//multi_images_64FC1_show_one_window(window_title, batch_train_x, CvSize(5, 2), CvSize(32, 32), 150);

			// 在当前的网络权值和网络输入下计算网络的输出(正向计算)
			feed_forward(batch_train_x);
		
			// 得到上面的网络输出后，通过对应的样本标签用bp算法来得到误差对网络权值(反向传播) 的导数？
			back_propagation(batch_train_y);

			// 得到误差对权值的导数后，就通过权值更新方法去更新权值
			update();

			mse += _err;
		}

		mse /= numbatches;
		_ERR.at(I) = mse; // 记录第I次训练集整体迭代时的均方误差

		// *********************************************************************************************** //

		clock_t toc = clock(); //获取毫秒级数目
		cout << "time has elapsed: " << (double)(toc - tic) / 1000 << " seconds" << endl;
	}

	cout << "train has finished!" << endl;
}


// CNN网络，测试，返回错误率
double CNN::test(const vector<Mat> &test_x, const vector<vector<double>> &test_y)
{
	cout << "begin to test" << endl;
	return 0;
}


// 依据网络结构设置CNN.layers, 初始化一个CNN网络
void CNN::init()
{
	// CNN网络层数
	int n = _layers.size();

	cout << "input layer " << 1 << " has initialised!" << endl;

	// 对CNN网络层数做循环
	for (int L = 1; L < n; L++)// 注意:这里实际上可以从第2层开始，所以L初始值是1不是0
	{
		// ======================================================================
		// 以下代码仅对第2, 4层(卷积层)有效

		if (_layers.at(L).type == 'c')
		{
			// 由前一层图像行列数,和本层卷积核尺度,计算本层图像行列数,二维向量
			_layers.at(L).iSizePic[0] = _layers.at(L - 1).iSizePic[0] - _layers.at(L).iSizeKer + 1;
			_layers.at(L).iSizePic[1] = _layers.at(L - 1).iSizePic[1] - _layers.at(L).iSizeKer + 1;

			// "前一层任意一个通道", 对应"本层所有通道"卷积核权值W(可训练参数)个数, 不包括加性偏置
			int fan_out = _layers.at(L).iChannel * pow(_layers.at(L).iSizeKer, 2);

			// "前一层所有通道", 对应"本层任意一个通道"卷积核权值W(可训练参数)个数, 不包括加性偏置
			int fan_in = _layers.at(L-1).iChannel * pow(_layers.at(L).iSizeKer, 2);

			// 对当前层的Ker和Ker_delta初始化维数，并赋初值
			_layers.at(L).Ker.clear();
			_layers.at(L).Ker.resize(_layers.at(L - 1).iChannel);

			_layers.at(L).Ker_delta.clear();
			_layers.at(L).Ker_delta.resize(_layers.at(L - 1).iChannel);

			// 对本层输入通道数做循环
			for (int I = 0; I < _layers.at(L - 1).iChannel; I++)
			{
				_layers.at(L).Ker.at(I).resize(_layers.at(L).iChannel);
				_layers.at(L).Ker_delta.at(I).resize(_layers.at(L).iChannel);

				// 对本层输出通道数做循环
				for (int J = 0; J < _layers.at(L).iChannel; J++)
				{
					double maximum = (double)sqrt(6.0f / (fan_in + fan_out));

					// "前一层所有通道",对"本层所有通道",层对层的全连接,卷积核权值W,进行均匀分布初始化,范围为:[-1,1]*sqrt(6/(fan_in+fan_out))
					_layers.at(L).Ker[I][J] = rand_array_double(_layers.at(L).iSizeKer, _layers.at(L).iSizeKer, -maximum, maximum);
					_layers.at(L).Ker_delta[I][J] = get_zero_array_double_same_size_as(_layers.at(L).Ker[I][J]);
				}
			}

			// 对本层输出通道加性偏置进行0值初始化
			_layers.at(L).B.assign(_layers.at(L).iChannel, 0);
			_layers.at(L).B_delta.assign(_layers.at(L).iChannel, 0);

			cout << "convolutional layer " << L + 1 << " has initialised!" << endl;
		}

		// ======================================================================
		// 以下代码对第3,5层(下采样层)有效

		if (_layers.at(L).type == 's')
		{
			_layers.at(L).iSizePic[0] = floor(_layers.at(L - 1).iSizePic[0] / _layers.at(L).iSample);
			_layers.at(L).iSizePic[1] = floor(_layers.at(L - 1).iSizePic[1] / _layers.at(L).iSample);
			_layers.at(L).iChannel = _layers.at(L - 1).iChannel;

			// 以下代码用于下采样层的计算

			// 对本层输出通道乘性偏置进行1值初始化
			_layers.at(L).Beta.assign(_layers.at(L).iChannel, 1);
			_layers.at(L).Beta_delta.assign(_layers.at(L).iChannel, 0);

			// 对本层输出通道加性偏置进行0值初始化
			_layers.at(L).B.assign(_layers.at(L).iChannel, 0);
			_layers.at(L).B_delta.assign(_layers.at(L).iChannel, 0);

			cout << "subsampling layer " << L + 1 << " has initialised!" << endl;
		}

		// ======================================================================
		// 本层是全连接层的前提下，三种情况：前一层是下采样层，前一层是卷积层，前一层是输入层

		if (_layers.at(L).type == 'f')
		{
			if (_layers.at(L - 1).type == 's' || _layers.at(L - 1).type == 'c' || _layers.at(L - 1).type == 'i')
			{
				// ------------------------------------------------------------------
				// 以下代码对第6层(过渡全连接层)有效

				// 上一层每个通道的像素个数 * 上一层输入通道数 = 当前层全连接输入个数
				int fvnum = _layers.at(L - 1).iSizePic[0] * _layers.at(L - 1).iSizePic[1] * _layers.at(L - 1).iChannel;
				// 当前输出层类别个数
				int onum = _layers.at(L).iChannel;

				double maximum = (double)sqrt(6.0f / (onum + fvnum));
				// 初始化当前层与上一层的连接权值
				_layers.at(L).W = rand_array_double(fvnum, onum, -maximum, maximum);// 注意是W[I][J],I为上一层的数目，J为当前层数目
				_layers.at(L).W_delta = get_zero_array_double_same_size_as(_layers.at(L).W);

				// 对本层输出通道加性偏置进行0值初始化
				_layers.at(L).B.assign(onum, 0);
				_layers.at(L).B_delta.assign(onum, 0);
			}
			else if (_layers.at(L - 1).type == 'f')
			{
				// ------------------------------------------------------------------
				// 以下代码对第7层(全连接层)有效。 对第8层也有效吧？

				// 上一层输入通道数 = 当前层全连接输入个数
				int fvnum = _layers.at(L - 1).iChannel;
				// 当前输出层类别个数
				int onum = _layers.at(L - 1).iChannel;

				double maximum = (double)sqrt(6.0f / (onum + fvnum));
				// 初始化当前层与上一层的连接权值
				_layers.at(L).W = rand_array_double(fvnum, onum, -maximum, maximum);// 注意是W[I][J],I为上一层的数目，J为当前层数目
				_layers.at(L).W_delta = get_zero_array_double_same_size_as(_layers.at(L).W);

				// 对本层输出通道加性偏置进行0值初始化
				_layers.at(L).B.assign(onum, 0);
				_layers.at(L).B_delta.assign(onum, 0);
			}

			cout << "fully connected layer " << L + 1 << " has initialised!" << endl;
		}
	}
}


// CNN网络,正向计算(批处理算法,核心是convn用法,和输出层批量映射)
void CNN::feed_forward(const vector<Mat> &train_x)
{
	;
}


// CNN网络,反向传播(批处理算法)
void CNN::back_propagation(const vector<vector<double>> &train_y)
{
	;
}


// CNN网络,卷积层和输出层的权值更新(附加惯性项)
void CNN::update(void)
{
	;
}
