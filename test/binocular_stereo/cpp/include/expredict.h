#ifndef _EXPREDICT_H
#define _EXPREDICT_h
#include "stdlib.h"
#include "stdio.h"
#include <vector>
using namespace std;





extern void predict_first(vector<double> input, vector<double>& output, vector<double>& output_v, vector<double>& output_a, float a = 0.8, int init_num = 3, int insert_num = 3);

extern void predict_second(vector<double> input, vector<double>& output, vector<double>& output_v, vector<double>& output_a, float a = 0.8, int init_num = 3, int insert_num = 3, double timeuse = 1.0);

extern void predict_third(vector<double> input, vector<double>& output, vector<double>& output_v, vector<double>& output_a, float a = 0.8, int init_num = 3, int insert_num = 3);




extern void predict_first_k(vector<double> input, double& output, double& k, float a = 0.8, int init_num = 3);

extern void predict_second_k(vector<double> input, double& output, double& k, float a = 0.8, int init_num = 3);

extern void predict_third_k(vector<double> input, double& output, double& k, float a = 0.8, int init_num = 3);

#endif
