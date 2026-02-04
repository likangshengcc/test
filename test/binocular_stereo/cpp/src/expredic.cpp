
#include "expredict.h"

using namespace std;
void predict(double* y, int num, double* predict_val, double* next, float a, int init_num = 3)
{
	for (int i = 0; i < num; i++)
	{
		double s0 = 0.0f;
		double s1 = 0.0f;
		if (num < init_num) s0 = y[i];
		else
		{
			for (int j = 0; j < init_num; j++)
			{
				s0 += y[j];
			}
			s0 /= init_num;
		}
		if (i == 0)
		{
			s1 = (1 - a) *s0 + a*y[i];
		}
		else
		{
			s1 = a * y[i] + (1 - a)*predict_val[i - 1];
		}
		predict_val[i] = s1;
	}
	*next = a * y[num - 1] + (1 - a)*predict_val[num - 1];
}

void predict_one_entire(vector<double> input, vector<double>& output, float a, int init_num, int insert_num)
{
	int num = input.size();
	
	vector<double> predict;
	for (int i = 0; i < num; i++)
	{
		double s0 = 0.0f, s1 = 0.0f;
		if (num < init_num) s0 = input[i];
		else
		{
			for (int j = 0; j < init_num; j++)
			{
				s0 += input[j];
			}
			s0 /= init_num;
		}
		if (i == 0)
			s1 = (1 - a)*s0 + a*input[i];
		else
			s1 = a*input[i] + (1 - a)*predict[i - 1];
		predict.push_back(s1);
		double next = a * input[i] + (1 - a)*predict[i];
		if (i < 1) continue;
		for (int j = 0; j <= insert_num; j++)//插入3个值
		{
			double insertv = (next - input[i - 1]) * (j + 1) / (insert_num + 1) + input[i - 1];
			output.push_back(insertv);
		}
	}

}


void predict_first(vector<double> input, vector<double>& output, vector<double>& output_v, vector<double>& output_a, float a, int init_num, int insert_num)
{
	int num = input.size();
	double last_predict = 0.0f;
	if (output.empty()) last_predict = input.back();
	else last_predict = output.back();
	vector<double> predict;
	for (int i = 0; i < num; i++)
	{
		double s0 = 0.0f, s1 = 0.0f;
		if (num < init_num) s0 = input[i];
		else
		{
			for (int j = 0; j < init_num; j++)
			{
				s0 += input[j];
			}
			s0 /= init_num;
		}
		if (i == 0)
			s1 = (1 - a)*s0 + a*input[i];
		else
			s1 = a*input[i] + (1 - a)*predict[i - 1];
		predict.push_back(s1);
		
	}
	if (num < 2) return;
	double next = a * input[num - 1] + (1 - a)*predict[num - 1];
	
	for (int j = 0; j <= insert_num; j++)//插入3个值
	{
		double insertv = (next - last_predict) * (j + 1) / (insert_num + 1) + last_predict;
		output.push_back(insertv);
	}

	double v = (next - last_predict) / (insert_num + 1);
	output_v.insert(output_v.end(), insert_num + 1, v);
	
	if (output_v.size() > 4)
	{
		double a = output_v.back() - *(output_v.end() - insert_num - 2);
		output_a.insert(output_a.end(), 1, a);
		for (int i = 0; i < insert_num; i++)
		{
			output_a.push_back(0);
		}
	}
	else
	{
		output_a.insert(output_a.end(), insert_num + 1, 0);
	}
}


void predict_second(vector<double> input, vector<double>& output, vector<double>& output_v, vector<double>& output_a, float a, int init_num, int insert_num, double timeuse)
{
	int num = input.size();
	double last_predict = 0.0f;
	if (output.empty()) last_predict = input.back();
	else last_predict = output.back();
	vector<double> predict1,predict2;
	for (int i = 0; i < num; i++)
	{
		double s0 = 0.0f, s1 = 0.0f;
		if (num < init_num) s0 = input[i];
		else
		{
			for (int j = 0; j < init_num; j++)
			{
				s0 += input[j];
			}
			s0 /= init_num;
		}
		if (i == 0)
			s1 = (1 - a)*s0 + a*input[i];
		else
			s1 = a*input[i] + (1 - a)*predict1[i - 1];
		predict1.push_back(s1);

	}
	for (int i = 0; i < num; i++)
	{
		double s = 0.0f;
		if (i == 0) predict2.push_back(predict1[0]);
		else
		{
			s = a * predict1[i] + (1 - a) * predict2[i - 1];
			predict2.push_back(s);
		}
	}

	if (num < 1) return;
	
	double a_t = 2 * predict1[num - 1] - predict2[num - 1];
	double b_t = a / (1 - a) * (predict1[num - 1] - predict2[num - 1]);

	double next = a_t + b_t * 1;

	for (int j = 0; j <= insert_num; j++)//插入3个值
	{
		double insertv = (next - last_predict) * (j + 1) / (insert_num + 1) + last_predict;
		output.push_back(insertv);
	}

	//double v = (next - last_predict) / (insert_num + 1);

	double v = (next - last_predict) * 1000 * 3.6 / timeuse;
	output_v.insert(output_v.end(), insert_num + 1, v);
	
	if (output_v.size() > 4)
	{
		//double a = output_v.back() - *(output_v.end() - insert_num - 2);
		double a = (output_v.back() - *(output_v.end() - insert_num - 2)) * 10000 / (36 * timeuse);
		output_a.insert(output_a.end(), 1, a);
		for (int i = 0; i < insert_num; i++)
		{
			output_a.push_back(0);
		}
	}
	else
	{
		output_a.insert(output_a.end(), insert_num + 1, 0);
	}
}

void predict_third(vector<double> input, vector<double>& output, vector<double>& output_v, vector<double>& output_a, float a, int init_num, int insert_num)
{
	int num = input.size();
	double last_predict = 0.0f;
	if (output.empty()) last_predict = input.back();
	else last_predict = output.back();
	vector<double> predict1, predict2, predict3;
	for (int i = 0; i < num; i++)
	{
		double s0 = 0.0f, s1 = 0.0f;
		if (num < init_num) s0 = input[i];
		else
		{
			for (int j = 0; j < init_num; j++)
			{
				s0 += input[j];
			}
			s0 /= init_num;
		}
		if (i == 0)
			s1 = (1 - a)*s0 + a*input[i];
		else
			s1 = a*input[i] + (1 - a)*predict1[i - 1];
		predict1.push_back(s1);

	}
	for (int i = 0; i < num; i++)
	{
		double s = 0.0f;
		if (i == 0) predict2.push_back(predict1[0]);
		else
		{
			s = a * predict1[i] + (1 - a) * predict2[i - 1];
			predict2.push_back(s);
		}
	}
	for (int i = 0; i < num; i++)
	{
		double s = 0.0f;
		if (i == 0) predict3.push_back(predict2[0]);
		else
		{
			s = a * predict2[i] + (1 - a) * predict3[i - 1];
			predict3.push_back(s);
		}
	}

	if (num < 1) return;

	double a_t = 3 * predict1[num - 1] - 3 * predict2[num - 1] + predict3[num - 1];
	double b_t = a / (2 * (1 - a)*(1 - a)) * ((6 - 5 * a)*predict1[num - 1] - 2 * (5 - 4 * a)*predict2[num - 1] + (4 - 3 * a)*predict3[num - 1]);
	double c_t = a / (2 * (1 - a)*(1 - a)) * (predict1[num - 1] - 2 * predict2[num - 1] + predict3[num - 1]);
	double next = a_t + b_t * 1 + c_t * 1 * 1;

	for (int j = 0; j <= insert_num; j++)//插入3个值
	{
		double insertv = (next - last_predict) * (j + 1) / (insert_num + 1) + last_predict;
		output.push_back(insertv);
	}

	double v = (next - last_predict) / (insert_num + 1);
	output_v.insert(output_v.end(), insert_num + 1, v);

	if (output_v.size() > 4)
	{
		double a = output_v.back() - *(output_v.end() - insert_num - 2);
		output_a.insert(output_a.end(), 1, a);
		for (int i = 0; i < insert_num; i++)
		{
			output_a.push_back(0);
		}
	}
	else
	{
		output_a.insert(output_a.end(), insert_num + 1, 0);
	}
}

//输出参数不同，不需要插值，输出一个预测值及相关斜率
void predict_first_k(vector<double> input, double& output, double& k, float a, int init_num)
{
	int num = input.size();
	
	vector<double> predict;
	for (int i = 0; i < num; i++)
	{
		double s0 = 0.0f, s1 = 0.0f;
		if (num < init_num) s0 = input[i];
		else
		{
			for (int j = 0; j < init_num; j++)
			{
				s0 += input[j];
			}
			s0 /= init_num;
		}
		if (i == 0)
			s1 = (1 - a)*s0 + a*input[i];
		else
			s1 = a*input[i] + (1 - a)*predict[i - 1];
		predict.push_back(s1);

	}
	if (num < 2) return;
	double next = a * input[num - 1] + (1 - a)*predict[num - 1];

	output = next;
	k = next - input.back();
	
}


void predict_second_k(vector<double> input, double& output, double& k, float a, int init_num)
{
	int num = input.size();
	
	vector<double> predict1, predict2;
	for (int i = 0; i < num; i++)
	{
		double s0 = 0.0f, s1 = 0.0f;
		if (num < init_num) s0 = input[i];
		else
		{
			for (int j = 0; j < init_num; j++)
			{
				s0 += input[j];
			}
			s0 /= init_num;
		}
		if (i == 0)
			s1 = (1 - a)*s0 + a*input[i];
		else
			s1 = a*input[i] + (1 - a)*predict1[i - 1];
		predict1.push_back(s1);

	}
	for (int i = 0; i < num; i++)
	{
		double s = 0.0f;
		if (i == 0) predict2.push_back(predict1[0]);
		else
		{
			s = a * predict1[i] + (1 - a) * predict2[i - 1];
			predict2.push_back(s);
		}
	}

	if (num < 1) return;

	double a_t = 2 * predict1[num - 1] - predict2[num - 1];
	double b_t = a / (1 - a) * (predict1[num - 1] - predict2[num - 1]);

	double next = a_t + b_t * 1;

	output = next;
	k = next - input.back();
}

void predict_third_k(vector<double> input, double& output, double& k, float a, int init_num)
{
	int num = input.size();
	
	vector<double> predict1, predict2, predict3;
	for (int i = 0; i < num; i++)
	{
		double s0 = 0.0f, s1 = 0.0f;
		if (num < init_num) s0 = input[i];
		else
		{
			for (int j = 0; j < init_num; j++)
			{
				s0 += input[j];
			}
			s0 /= init_num;
		}
		if (i == 0)
			s1 = (1 - a)*s0 + a*input[i];
		else
			s1 = a*input[i] + (1 - a)*predict1[i - 1];
		predict1.push_back(s1);

	}
	for (int i = 0; i < num; i++)
	{
		double s = 0.0f;
		if (i == 0) predict2.push_back(predict1[0]);
		else
		{
			s = a * predict1[i] + (1 - a) * predict2[i - 1];
			predict2.push_back(s);
		}
	}
	for (int i = 0; i < num; i++)
	{
		double s = 0.0f;
		if (i == 0) predict3.push_back(predict2[0]);
		else
		{
			s = a * predict2[i] + (1 - a) * predict3[i - 1];
			predict3.push_back(s);
		}
	}

	if (num < 1) return;

	double a_t = 3 * predict1[num - 1] - 3 * predict2[num - 1] + predict3[num - 1];
	double b_t = a / (2 * (1 - a)*(1 - a)) * ((6 - 5 * a)*predict1[num - 1] - 2 * (5 - 4 * a)*predict2[num - 1] + (4 - 3 * a)*predict3[num - 1]);
	double c_t = a / (2 * (1 - a)*(1 - a)) * (predict1[num - 1] - 2 * predict2[num - 1] + predict3[num - 1]);
	double next = a_t + b_t * 1 + c_t * 1 * 1;

	output = next;
	k = next - input.back();
}
