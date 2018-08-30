// This is Math.cpp

#include <maths.h>
#include <stdlib.h>
#include <time.h>


// 类似matlab中的randperm函数，即得到0~num-1之间的打乱顺序后的数组
void randperm_array(int serial_num[], int num)
{
	for (int i = 0; i < num; i++)
	{
		serial_num[i] = i;
	}

	int j, temp;

	srand((unsigned)time(NULL));//srand()函数产生一个以当前时间开始的随机种子
	for (int i = num; i > 1; i--)
	{

		j = rand() % i;
		temp = serial_num[i - 1];
		serial_num[i - 1] = serial_num[j];
		serial_num[j] = temp;
	}
}

