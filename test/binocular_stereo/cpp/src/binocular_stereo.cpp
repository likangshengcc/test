
 // ZED includes
#include <sl/Camera.hpp>
#include "cvui.h"
// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/cvconfig.h>
// Sample includes
#include <SaveDepth.hpp>
#include "bmpHelper.h"
#include "expredict.h"

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#include <math.h>
#include <numeric>

typedef struct udp_data{
	float ldis;
	float loffset;
	float lspeed;
	float fdis;
	float foffset;
	float fspeed;
	float rdis;
	float roffset;
	float rspeed;
	float stopdis;
	char  fn[30];
} UDP_DATA;


typedef struct sdl_cloud_data{
	int x;
	int y;
	sl::float4 sdl_point_value;
} SDL_CLOUD_DATA;


using namespace sl;

float speed_left[30] = {18.646,17.567,16.507,15.434,14.374,13.301,12.222,11.143,10.07,9.01,7.95,6.871,5.798,4.719,3.659,2.586,1.507,0.447};
float speed_front[30] = {16.562,16.104,15.612,15.0895,14.5395,13.9525,13.3355,12.6905,12.0125,11.3005,10.558,9.782,8.978,8.141,7.273,6.378,5.446,4.48,3.487,2.46,1.406,0.324};
float speed_right[30] = {18.596,16.133,13.829,11.6565,9.6585,7.8125,6.1135,4.5765,3.1955,1.958,0.8855};

float img[6400][3600], X[6400][3600], Y[6400][3600], Z[6400][3600];
float Z1[6400][3600],Z0[6400][3600];

std::vector<double>leftdis,frontdis,rightdis,leftoutput,frontoutput,rightoutput,left_output_v,front_output_v,right_output_v,left_output_a,front_output_a,right_output_a;

cv::Mat slMat2cvMat(Mat& input);

#define DEST_PORT 8282
#define DEST_IP_ADDRESS "192.168.0.200"

#define IMG_WIDTH	960
#define IMG_HEIGHT	540

unsigned char imgdata[IMG_WIDTH * IMG_HEIGHT * 3];
// 将一阵图像分为四个部分，返回4部分的xy偏移量

std::vector<cv::Point> splitimg(cv::Mat& depth_image_ocv, Resolution& new_image_size,
	std::string fnamestr, std::vector<std::string>& rvecfnames)
{

	std::vector<cv::Rect> vecrc;
	vecrc.push_back(cv::Rect(0, 0, new_image_size.width, new_image_size.height));
	//vecrc.push_back(cv::Rect(new_image_size.width / 4, new_image_size.height / 3, new_image_size.width / 2, 2*new_image_size.height / 3));
	vecrc.push_back(cv::Rect(7*new_image_size.width / 16, 0, new_image_size.width / 2, new_image_size.height));
	vecrc.push_back(cv::Rect(new_image_size.width / 16, 0, new_image_size.width / 2, new_image_size.height));

	int 	index = 0;
	char 	fname[512] = {};

	std::vector<cv::Point> result;
	rvecfnames.clear();
	for (auto it = vecrc.begin(); it != vecrc.end(); ++it)
	{
		sprintf(fname, fnamestr.c_str(), index++);
		imwrite(fname, depth_image_ocv(*it));
		result.push_back(cv::Point(it->x, it->y));
		rvecfnames.push_back(fname);
	}
	return result;
}

struct ITEMTRAJ
{
	float alpha;
};

// 障碍物
struct ITEMDATA 
{
	cv::Point pt1; 	// 左上角
	cv::Point pt2; 	// 右下角
	float dis;	// 与camera的距离
	float lastdis;
	float spd;
	float zuoyou;	//新添加的距离车的左右位置
	float deep;		//deep
	float alpha;	
	int n; 		// 索引
	char dirc[10];	//fangxiang
};

// 便于一一对应障碍物与SingleData
struct ItemSingleData : SingleData {
	int n;		// 索引
};

// 排序的算子
bool comp(const ITEMDATA& a, const ITEMDATA& b)
{
	//return a.dis < b.dis;
	return a.n < b.n;
}

// 排序
void SortItem(std::vector<ITEMDATA>& arr)
{
	std::sort(arr.begin(), arr.end(), comp);
}


// 矩形相交检测
bool IsRectOverlap(cv::Rect& rc1, cv::Rect& rc2)
{
	cv::Rect r = rc1 & rc2;
	if (r.width)
	{
		return true;
	}
	return false;
}
// 合并矩形
cv::Rect MergeRect(cv::Rect& rc1, cv::Rect& rc2) 
{
	cv::Rect result;
	result = rc1 | rc2;
	return result;
}

// 合并障碍物，3个分别与另外2个比较，如果有相交就合并
std::vector<ITEMDATA> MergeItem(std::vector<ITEMDATA>& arr)
{
	std::vector<ITEMDATA> result;
	
	if(arr.size() == 3)
	{
		cv::Rect r1(arr[0].pt1, arr[0].pt2);
		cv::Rect r2(arr[1].pt1, arr[1].pt2);
		cv::Rect r3(arr[2].pt1, arr[2].pt2);
		ITEMDATA item;
		if (IsRectOverlap(r1, r2)) 		// 先比较1和2
		{
			auto r1r2 = MergeRect(r1, r2);
			item.dis = arr[0].dis < arr[1].dis ? arr[0].dis : arr[1].dis;
			item.zuoyou = abs(arr[0].zuoyou) < abs(arr[1].zuoyou) ? arr[0].zuoyou : arr[1].zuoyou;
			item.spd = arr[0].spd < arr[1].spd ? arr[0].spd : arr[1].spd;
			item.n = 0;
			if (IsRectOverlap(r1r2, r3)) 	// 比较12与3
			{
				auto r1r2r3 = MergeRect(r1r2, r3);
				item.dis = item.dis < arr[2].dis ? item.dis : arr[2].dis;
				item.zuoyou = abs(item.zuoyou) < abs(arr[2].zuoyou) ? item.zuoyou : arr[2].zuoyou;
				item.spd = item.spd < arr[2].spd ? item.spd : arr[2].spd;
				item.pt1 = r1r2r3.tl();
				item.pt2 = r1r2r3.br();
				result.push_back(item);
			}
			else
			{
				item.pt1 = r1r2.tl();
				item.pt2 = r1r2.br();
				result.push_back(item);
				arr[2].n = 1;
				result.push_back(arr[2]);
			}
		}
		else if (IsRectOverlap(r1, r3))  	// 先比较1和3
		{
			auto r1r3 = MergeRect(r1, r3);
			item.dis = arr[0].dis < arr[2].dis ? arr[0].dis : arr[2].dis;
			item.zuoyou = abs(arr[0].zuoyou) < abs(arr[2].zuoyou) ? arr[0].zuoyou : arr[2].zuoyou;
			item.spd = arr[0].spd < arr[2].spd ? arr[0].spd : arr[2].spd;
			item.n = 0;
			if (IsRectOverlap(r1r3, r2)) 	// 比较13与2
			{
				auto r1r2r3 = MergeRect(r1r3, r2);
				item.dis = item.dis < arr[1].dis ? item.dis : arr[1].dis;
				item.zuoyou = abs(item.zuoyou) < abs(arr[1].zuoyou) ? item.zuoyou : arr[1].zuoyou;
				item.spd = item.spd < arr[1].spd ? item.spd : arr[1].spd;
				item.pt1 = r1r2r3.tl();
				item.pt2 = r1r2r3.br();
				result.push_back(item);
			}
			else
			{
				item.pt1 = r1r3.tl();
				item.pt2 = r1r3.br();
				result.push_back(item);
				arr[1].n = 1;
				result.push_back(arr[1]);
			}
		}
		else if (IsRectOverlap(r2, r3)) 	// 先比较2和3
		{
			auto r2r3 = MergeRect(r2, r3);
			item.dis = arr[1].dis < arr[2].dis ? arr[1].dis : arr[2].dis;
			item.zuoyou = abs(arr[1].zuoyou) < abs(arr[2].zuoyou) ? arr[1].zuoyou : arr[2].zuoyou;
			item.spd = arr[1].spd < arr[2].spd ? arr[1].spd : arr[2].spd;
			item.n = 0;
			if (IsRectOverlap(r2r3, r1))	// 比较23与1
			{
				auto r1r2r3 = MergeRect(r2r3, r1);
				item.dis = item.dis < arr[0].dis ? item.dis : arr[0].dis;
				item.zuoyou = abs(item.zuoyou) < abs(arr[0].zuoyou) ? item.zuoyou : arr[0].zuoyou;
				item.spd = item.spd < arr[0].spd ? item.spd : arr[0].spd;
				item.pt1 = r1r2r3.tl();
				item.pt2 = r1r2r3.br();
				result.push_back(item);
			}
			else
			{
				item.pt1 = r2r3.tl();
				item.pt2 = r2r3.br();
				result.push_back(item);
				arr[0].n = 1;
				result.push_back(arr[0]);
			}
		}
		else
		{
			result.push_back(arr[0]);
			result.push_back(arr[1]);
			result.push_back(arr[2]);
		}
	}
	else if(arr.size() == 2)
	{
		cv::Rect r1(arr[0].pt1, arr[0].pt2);
		cv::Rect r2(arr[1].pt1, arr[1].pt2);

		ITEMDATA item;
		if (IsRectOverlap(r1, r2)) 		// 先比较1和2
		{
			auto r1r2 = MergeRect(r1, r2);
			item.dis = arr[0].dis < arr[1].dis ? arr[0].dis : arr[1].dis;
			item.zuoyou = abs(arr[0].zuoyou) < abs(arr[1].zuoyou) ? arr[0].zuoyou : arr[1].zuoyou;
			item.spd = arr[0].spd < arr[1].spd ? arr[0].spd : arr[1].spd;
			item.n = 0;
			item.pt1 = r1r2.tl();
			item.pt2 = r1r2.br();
			result.push_back(item);
		}
		else
		{
			result.push_back(arr[0]);
			result.push_back(arr[1]);
		}
	}
	else if(arr.size() == 1)
	{
		result.push_back(arr[0]);
	}	

	return result;
}

struct SDLDATA // sdl一帧的迭代过程所需的数据，用于展示迭代过程
{
	std::vector<std::vector<ItemSingleData>> vecItem; // 需要迭代的障碍物
	void reset()
	{
		vecItem.clear();
	}
};

// 将sdl返回的数据SingleData转为矩形的2个点
void SingleData2xy(SingleData& data, sl::Resolution& size, int& startx, int& starty, int& endx, int& endy) {
	startx = data.x - (int)data.r;
	if (startx < 0)
	{
		startx = 0;
	}
	endx = data.x + (int)data.r;
	if (endx > size.width)
	{
		endx = size.width;
	}
	starty = data.y - (int)data.r;
	if (starty < 0)
	{
		starty = 0;
	}
	endy = data.y + (int)data.r;
	if (endy > size.height)
	{
		endy = size.height;
	}
}

// 对矩形进行缩放
// 对矩形进行缩放
void ScaleRect(cv::Point& pt1, cv::Point& pt2, float scale = 0.1f) {
	auto sx = fabs(pt1.x - pt2.x) * scale;
	pt1.x -= sx;
	pt2.x += sx;
	auto sy = fabs(pt1.y - pt2.y) * scale;
	pt1.y -= sy;
	pt2.y += sy;
}

void stopline(cv::Mat &binary, int &up, int &down)
{
	int stopline_low = 0;
	int stopline_temp;
	cv::Mat bgr;
	cv::blur(binary, binary, cv::Size(3, 3));
	cv::Canny(binary, bgr, 2, 5);

	//cv::imshow("00", bgr);

	//waitKey();

	vector<cv::Vec4f> plines;
	cv::HoughLinesP(bgr, plines, 1, CV_PI / 180.0, 70, 40, 25);
	cv::Scalar color = cv::Scalar(0, 255, 255);
	up = 0;
	int udpro = 0;
	down = 0;
	int udnum = 0;
	for (size_t i = 0; i < plines.size(); i++) {
		cv::Vec4f hline = plines[i];
		if (abs(int(hline[1] - hline[3])) < 3 && abs(int(hline[0] - hline[2])) > 80) {
			udnum++;
			stopline_temp = hline[1];
			if (stopline_temp > stopline_low)
				stopline_low = hline[1];
			//udpro += hline[1] + 310;
			//cout << "find line!" << endl;
		}
	}
	if (udnum)
	{
		up = stopline_low + 310;// udpro / udnum;
		down = up + 10;// udpro / udnum + 30;
	}
}

// 排序的算子
bool compx(const float& a, const float& b)
{
	//return a.dis < b.dis;
	return a < b;
}

// 排序
void SortX(std::vector<float>& arr)
{
	std::sort(arr.begin(), arr.end(), compx);
}

bool create_dir(char *path,int depth,char *new_path)
{
	struct dirent **name_list;
	bool dir_existence = false;
	int x_n,dir_n = 0;
	bool ret = false;
	int n = scandir(path,&name_list,0,alphasort);
	if(n < 0){
		printf("\nscandir return %d \n",n);
	}
	else
	{
		int index = 0;
		while(index < n){
			if(name_list[index]->d_type == 4){
				//printf("\nname: %s,%d\n",name_list[index]->d_name,name_list[index]->d_type);
				if(strncmp(name_list[index]->d_name,"video_",6)==0){
				//	printf("\nok:%s\n",name_list[index]->d_name);
					char* s_x;
					char string_x[50];
					int i = 0;
					s_x = name_list[index]->d_name;
					while((*s_x++) != '_');
					while((*s_x) != '\0'){
						string_x[i++] = (*s_x++);
					}
					string_x[i] = '\0';
					x_n = atoi(string_x);
					if(x_n > dir_n)
						dir_n = x_n;
					dir_existence = true;
				}
			}
			free(name_list[index++]);
		}
		if(!dir_existence){
			char *path = "./video_0";
			if(mkdir(path,0766)==0)
			{
				//printf("\ncreated the directory %s.\n",path);
				strcpy(new_path , path);
				ret = true;
			}
			else
			{
				printf("can't create the directory %s.\n",path);
				printf("errno:%d\n",errno);
				printf("ERR:%s\n",strerror(errno));
				ret = false;
			}
		}	
		else{
				char string_x[50];
				
				dir_n++;
				printf("\nadd dir video_%d\n",dir_n);
				char dir_name[50];
				strcpy(dir_name, "./video_");
				sprintf(string_x,"%d",dir_n);
				strcat(dir_name,string_x);
			
				char *path = dir_name;
				if(mkdir(path,0766)==0)
				{
					//printf("\ncreated the directory %s.\n",path);
					strcpy(new_path , path);
					ret = true;
				}
				else
				{
					printf("can't create the directory %s.\n",path);
					printf("errno:%d\n",errno);
					printf("ERR:%s\n",strerror(errno));
					ret = false;
				}		
		}		
		free(name_list);
	}
	return ret;
}

float calcu_error(vector<float> x, vector<float> y, float k, float b)
{
	float error = 0;
	for (int i = 0; i < x.size(); i++)
	{
		float ry = k*x[i] + b;
		error += pow((y[i] - ry), 2);
	}
	return error;
}

float linear_regression(vector<float> x, vector<float> y, float& k, float& b)
{
	assert(x.size() == y.size());
	float mean_x = accumulate(x.begin(), x.end(), 1.0) / (float)(x.size());
	float mean_y = accumulate(y.begin(), y.end(), 1.0) / (float)(y.size());

	float sumx2 = 0, sumy2 = 0, sumxy = 0;
	for (int i = 0; i < x.size(); i++)
	{
		sumx2 += x[i] * x[i];
		sumy2 += y[i] * y[i];
		sumxy += x[i] * y[i];
	}

	sumx2 = sumx2 / x.size();
	sumy2 = sumy2 / y.size();
	sumxy = sumxy / x.size();

	float dx2 = sumx2 - mean_x*mean_x;
	float dy2 = sumy2 - mean_y*mean_y;
	float dxy = sumxy - mean_x*mean_y;

	float t = atan2(2 * dxy, dx2 - dy2) / 2;
	k = sin(t) / cos(t);
	b = mean_y - k*mean_x;
	return calcu_error(x, y, k, b);
}

float linear_regression_x(std::vector<SDL_CLOUD_DATA>& sdl_point_trans_set)
{
	float sum_x = 0.0;
	float mean_x= 0.0;

	for(int i; i < sdl_point_trans_set.size(); i++)
	{
		sum_x += sdl_point_trans_set[i].sdl_point_value.x;
	}
	mean_x = sum_x / (float)sdl_point_trans_set.size();
	return mean_x;
}

float linear_regression_z(std::vector<SDL_CLOUD_DATA>& sdl_point_trans_set)
{
	float sum_z = 0.0;
	float mean_z= 0.0;

	for(int i; i < sdl_point_trans_set.size(); i++)
	{
		sum_z += sdl_point_trans_set[i].sdl_point_value.z;
	}
	mean_z = sum_z / (float)sdl_point_trans_set.size();
	return mean_z;
}

float dot2line(float k, float b, float x, float y)
{
	return fabs(k * x - y + b) / sqrt(k * k + 1);
}

void calcu_sdl(std::vector<SDL_CLOUD_DATA>& sdl_point_trans_in_set, float init_std, float& center_x, std::vector<SDL_CLOUD_DATA>& sdl_point_trans_set )
{
	//assert(x.size() == y.size());
	//innerx.clear(); innery.clear();

	sdl_point_trans_set.clear();
	vector<float> distances;
	printf("sdl_point_trans_in_set.size()=%d\n",sdl_point_trans_in_set.size());
	for (int i = 0; i < sdl_point_trans_in_set.size(); i++)
	{

		float dis = fabs(sdl_point_trans_in_set[i].sdl_point_value.x - center_x);
		distances.push_back(dis);
		if (dis <= init_std)
		{
			sdl_point_trans_set.push_back(sdl_point_trans_in_set[i]);
		}
	}
	center_x = linear_regression_x(sdl_point_trans_set);	
}

void calcu_sdl_z(std::vector<SDL_CLOUD_DATA>& sdl_point_trans_in_set, float init_std_z, float& center_z, std::vector<SDL_CLOUD_DATA>& sdl_point_trans_set )
{
	//assert(x.size() == y.size());
	//innerx.clear(); innery.clear();

	sdl_point_trans_set.clear();
	vector<float> distances;
	printf("sdl_point_trans_in_set.size()=%d\n",sdl_point_trans_in_set.size());
	for (int i = 0; i < sdl_point_trans_in_set.size(); i++)
	{

		float dis = fabs(sdl_point_trans_in_set[i].sdl_point_value.z - center_z);
		distances.push_back(dis);
		if (dis <= init_std_z)
		{
			sdl_point_trans_set.push_back(sdl_point_trans_in_set[i]);
		}
	}
	center_z = linear_regression_z(sdl_point_trans_set);	
}

		
char saveimg_dir[50];

int main(int argc, char **argv) {

// initial udp   
	int sock_fd;
	int send_num;
	int recv_num;
	socklen_t dest_len;
	char send_buf[400] = {"hello "};
	char recv_buf[200];
	struct sockaddr_in addr_serv;

	sock_fd = socket(AF_INET,SOCK_DGRAM,0);
	
	memset(&addr_serv,0,sizeof(addr_serv));
	addr_serv.sin_family = AF_INET;
	addr_serv.sin_addr.s_addr = inet_addr(DEST_IP_ADDRESS);
	addr_serv.sin_port = htons(DEST_PORT);

	dest_len = sizeof(struct sockaddr_in);
	printf("begin send:\n");

	char *now_dir,pwd[2]=".";
	
	bool creat_dir = false;

	now_dir = pwd;

	printf("\nDirectory scan of %s\n",now_dir);
	creat_dir = create_dir(now_dir,0,saveimg_dir);
	if(creat_dir)
		printf("\nnew_dir : %s\n",saveimg_dir);
	else
		printf("\n cann't create new dir.\n");
	



	printf("\nFinish.\n");

// 下面的初始化代码都是案例里自带的
// Create a ZED camera object
    Camera zed;

	static struct timeval start,end,start1,end1;
	double run_timeuse = 0;
	double run_timeuse1 = 0;

// Set configuration parameters
    InitParameters init_params;
	init_params.camera_fps = 30;
	init_params.camera_resolution = RESOLUTION::HD720;
	init_params.depth_mode = DEPTH_MODE::ULTRA;
	init_params.coordinate_units = UNIT::METER;
   //	if (argc > 1) 
	//	init_params.input.setFromSVOFile(argv[1]);
        
// Open the camera
    ERROR_CODE err = zed.open(init_params); 
   	if (err != ERROR_CODE::SUCCESS) 
	{
    	printf("%s\n", toString(err).c_str());
       	zed.close();
    	return 1; // Quit if an error occurred
    }

  
// Set runtime parameters after opening the camera
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE::STANDARD;

// Prepare new image size to retrieve half-resolution images
	Resolution image_size = zed.getCameraInformation().camera_resolution;

    int new_width 	= image_size.width / 2;
    int new_height 	= image_size.height / 2;

    Resolution new_image_size(new_width, new_height);

// To share data between sl::Mat and cv::Mat, use slMat2cvMat()
// Only the headers and pointer to the sl::Mat are copied, not the data itself
	sl::Mat image_zed(new_width, new_height, MAT_TYPE::U8_C4);
	cv::Mat image_ocv = slMat2cvMat(image_zed);
	sl::Mat depth_image_zed(new_width, new_height, MAT_TYPE::U8_C4);
	cv::Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
    Mat point_cloud;
	cv::Mat image_ocv_copy;

	std::vector<SDL_CLOUD_DATA> sdl_point_set;
	std::vector<SDL_CLOUD_DATA> sdl_point_nogrand_set;

//所有涉及使用new生成的指针、矩阵等需要占用堆栈的变量均要做销毁处理。（未完全优化）
	int n = 5; 	// GaussianFunc参数
	int v = 30;	// GaussianFunc参数
	int f = 0;	// GaussianFunc参数
	
	std::vector<ItemSingleData> vecnoit; // 没有迭代过程的最终障碍物数组
	 
	
	int fnameindex = 0; // 保存点云文件名的索引增量
    char key = ' ';

	int maxitemnum = 3; // 最多障碍物数量，默认1，最大3
	int rectsize = 5; // GaussianFunc搜索框的size，数越大框越小，默认2框最大，递增1则越来越小

	
	bool bsavepointcloud = false;//保存点云数据的开关
	bool bdrawmerged = true;// 绘制合并后的障碍物的开关
	bool breplay = false;// 显示迭代过程的开关

	bool stop_flag = true;//stop_line on/off

	

	bool first_frame = true; // the first frame


	SDLDATA itdata; // 用于迭代过程的数据
	float mindislast[4];
	

	char msg[128]  = {};
	char msg1[128] = {};
	char msg2[128] = {};
	char msg3[128] = {};

	const char save_dir[256] = "../video/video_";
	long int save_count = 0;

	char savejpg_name[100];
	char string[100];


   
	while (key != 'q') 
	{
		gettimeofday(&start1,NULL);	

		
	   	if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {

			vecnoit.clear();
		
        	zed.retrieveImage(image_zed, VIEW::LEFT, MEM::CPU, new_image_size);
		
       		zed.retrieveImage(depth_image_zed, VIEW::DEPTH, MEM::CPU, new_image_size);
		
			zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA, MEM::CPU, new_image_size);

			if(!first_frame)
			{
				gettimeofday(&end,NULL);
				run_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
				run_timeuse /= 1000;
				//printf("\nrun_timeuse = %fms",run_timeuse);
				
			}
			gettimeofday(&start,NULL);

			cv::Mat matreplay;
			cv::Mat depthim;
			cv::Mat sdl_image = cv::Mat(new_height,new_width,CV_8UC1,cv::Scalar(255));

			//cv::imshow("sdl_image", sdl_image);
			//cv::imshow("depth_image_zed", depth_image_ocv);

			

		//	cv::cvtColor(depth_image_ocv, depthim, CV_RGBA2GRAY);
			
			//cv::cvtColor(image_ocv, image_ocv_copy, 1);

			int x, y, z;

			sl::float4 point_cloud_value;

			SDL_CLOUD_DATA sdl_point;

			sdl_point_set.clear();

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					x = i;
					y = j;

					point_cloud.getValue(x, y, &point_cloud_value);

					X[i][j] = point_cloud_value.x;

					Y[i][j] = point_cloud_value.y;

					if(std::isfinite(point_cloud_value.z))
					{
						//Z1[i][j] = sqrt(point_cloud_value.x * point_cloud_value.x + point_cloud_value.y * point_cloud_value.y + point_cloud_value.z * point_cloud_value.z);
						//Z0[i][j] = point_cloud_value.z;//Z1[i][j];						
						//Z[i][j] = point_cloud_value.z;

						sdl_point.x = i;
						sdl_point.y = j;
						sdl_point.sdl_point_value = point_cloud_value;
						sdl_point_set.push_back(sdl_point);
					

					/*	if (Z1[i][j] < 39)//除去无效障碍，只取相对车来说的近景障碍
						{
							depthim.at<uchar>(j, i) = 0;
						}
						else
						{
							depthim.at<uchar>(j, i) = 255;
						}*/
					}
					else
					{
						//depthim.at<uchar>(j, i) = 255;
						//Z1[i][j] = 40;
						//img[i][j] = 40;
						//Z[i][j] = 40;
						//Z0[i][j] = 40;
					}					

				}
			}

			printf("sdl_point_set.size()=%d\n",sdl_point_set.size());

		/*	gettimeofday(&end1,NULL);
			run_timeuse = 1000000 * (end1.tv_sec - start1.tv_sec) + end1.tv_usec - start1.tv_usec;
			run_timeuse /= 1000;
			printf("\nrun_timeuse = %fms\n",run_timeuse);*/

			for (int i = 0; i < sdl_point_set.size(); i++)
			{
				sdl_image.at<uchar>(sdl_point_set[i].y, sdl_point_set[i].x) = 0;
			}

			cv::imshow("sdl_point_set", sdl_image);
			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/sdl_point_set_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);
			//key = cv::waitKey();

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}

			float grid_size = 0.5;
			float height_thresh = 0.08;
    		std::map<std::pair<int, int>, float> grid_min_z;

			int minn_y = std::numeric_limits<int>::max();
    		int maxx_y = std::numeric_limits<int>::lowest();
    		for (int i = 0; i < sdl_point_set.size(); i++ )
    		{
        		int grid_x = static_cast<int>(sdl_point_set[i].sdl_point_value.x / grid_size);
        		int grid_y = static_cast<int>(sdl_point_set[i].sdl_point_value.z / grid_size);
        		auto key = std::make_pair(grid_x, grid_y);


        		if (grid_min_z.find(key) == grid_min_z.end() || sdl_point_set[i].sdl_point_value.y > grid_min_z[key])
        		{
            		grid_min_z[key] = sdl_point_set[i].sdl_point_value.y;
					//printf("\ngrid_min_z[%d,%d] = %f\n",key.first,key.second,grid_min_z[key]);
        		}
    		}
			
			//for(int i = minn_y; i <= maxx_y; i++)
				//printf("\ngrid_min_z[%d] = %f\n",i,grid_min_z[i]);

			sdl_point_nogrand_set.clear();

    		// 提取地面点（高度差小于阈值）
    		for (int i = 0; i < sdl_point_set.size(); i++)
    		{
        		int grid_x = static_cast<int>(sdl_point_set[i].sdl_point_value.x / grid_size);
        		int grid_y = static_cast<int>(sdl_point_set[i].sdl_point_value.z / grid_size);
        		float min_z = grid_min_z[std::make_pair(grid_x, grid_y)];
				
				auto first_element = grid_min_z.begin();

				//printf("min_z[%d,%d] = %f,sdl_point_set[%d].sdl_point_value.y= %f\n",first_element->first.first,first_element->first.second,min_z,i,sdl_point_set[i].sdl_point_value.y);

				float height = fabs(sdl_point_set[i].sdl_point_value.y - min_z);
				//printf("height = %f\n",height);

        		if (height > height_thresh)
        		{
					//printf("min_z[%d,%d]ok\n",first_element->first.first,first_element->first.second);
            		sdl_point_nogrand_set.push_back(sdl_point_set[i]);
        		}
    		}

			printf("sdl_point_nogrand_set.size()=%d\n",sdl_point_nogrand_set.size());
			for (int i = 0; i < sdl_point_nogrand_set.size(); i++)
			{
				sdl_image.at<uchar>(sdl_point_nogrand_set[i].y, sdl_point_nogrand_set[i].x) = 0;
			}

			cv::imshow("sdl_point_nogrand_set", sdl_image);
			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/sdl_point_nogrand_set_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);

			//key = cv::waitKey();


			//	cv::imshow("Image1", depthim);
			//SDL点云处理
#if 1
			//ROI范围
			std::vector<SDL_CLOUD_DATA> sdl_point_roi_set;
    		for (int i = 0; i < sdl_point_nogrand_set.size(); i++)
    		{
				//if((sdl_point_nogrand_set[i].sdl_point_value.x < 2.5)&&(sdl_point_nogrand_set[i].sdl_point_value.x > -3.5)&&(sdl_point_nogrand_set[i].sdl_point_value.y < 	
		                   //  0.8)&&(sdl_point_nogrand_set[i].sdl_point_value.y > -1.5))

				if((sdl_point_nogrand_set[i].sdl_point_value.x < 1.0)&&(sdl_point_nogrand_set[i].sdl_point_value.x > -0.4)&&(sdl_point_nogrand_set[i].sdl_point_value.y < 	
		        	0.8)&&(sdl_point_nogrand_set[i].sdl_point_value.y > -0.8))
					sdl_point_roi_set.push_back(sdl_point_nogrand_set[i]);
    		}	
			
			std::vector<SDL_CLOUD_DATA> sdl_cloud;
			for (int i = 0; i < sdl_point_roi_set.size(); i++)
    		{
        		sdl_cloud.push_back(sdl_point_roi_set[i]);
    		}

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}

			printf("sdl_cloud.size()=%d\n",sdl_cloud.size());
			for (int i = 0; i < sdl_cloud.size(); i++)
			{
				sdl_image.at<uchar>(sdl_cloud[i].y, sdl_cloud[i].x) = 0;
			}

			cv::imshow("sdl_cloud", sdl_image);

			cv::imshow("Image", image_ocv);

			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/video_sdl_cloud_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);

			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/video_image_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			//cv::imwrite(savejpg_name,image_ocv);

			/*vector<float> innerx;
			for (int i = 0; i < sdl_cloud.size(); i++)
			{
				innerx.push_back(sdl_cloud[i].sdl_point_value.x);
			}*/

			int fd;
			char str[100];
			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			sprintf(str,"/sdl_%ld.csv",save_count);
			strcat(savejpg_name,str);
			fd = open(savejpg_name,O_RDWR|O_CREAT|O_TRUNC,S_IRWXU);
			if(fd < 0){
				printf("\nFail to open File\n");
				return 0;
			}
			/*write(fd,"x=[",strlen("x=["));
			for (int i = 0; i < sdl_cloud.size(); i++)
			{
				sprintf(str,"%f,",sdl_cloud[i].sdl_point_value.x);
				write(fd,str,strlen(str));
				if(i%10 == 0)
					write(fd,"\n",strlen("\n"));
			}

			write(fd,"]\n",strlen("]\n"));
			write(fd,"y=[",strlen("y=["));
			for (int i = 0; i < sdl_cloud.size(); i++)
			{
				sprintf(str,"%f,",sdl_cloud[i].sdl_point_value.y);
				write(fd,str,strlen(str));
				if(i%10 == 0)
					write(fd,"\n",strlen("\n"));
			}

			write(fd,"]\n",strlen("]\n"));*/

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}


			for (int i = 0; i < sdl_cloud.size(); i++)
			{
				sprintf(str,"%d,",sdl_cloud[i].x);
				write(fd,str,strlen(str));
				sprintf(str,"%d,",sdl_cloud[i].y);
				write(fd,str,strlen(str));
				sprintf(str,"%f,",sdl_cloud[i].sdl_point_value.x);
				write(fd,str,strlen(str));
				sprintf(str,"%f,",(-1)*sdl_cloud[i].sdl_point_value.y);
				write(fd,str,strlen(str));
				sprintf(str,"%f\n",sdl_cloud[i].sdl_point_value.z);
				write(fd,str,strlen(str));

				sdl_image.at<uchar>(sdl_cloud[i].y, sdl_cloud[i].x) = 0;


				//printf("\nx = %d,y = %d,point_x = %f,point_y = %f\n",sdl_cloud[i].x,sdl_cloud[i].y,sdl_cloud[i].sdl_point_value.x,sdl_cloud[i].sdl_point_value.y);
				//cv::imshow("sdl_cloud_point", sdl_image);
				//key = cv::waitKey(); 
			}

			close(fd);

			float init_std = 0.7;
			float center_x = 0.0;

			float sum_init_x = 0.0;
			std::vector<SDL_CLOUD_DATA> sdl_init_x_set;
			vector<float> distances;
			for (int i = 0; i < sdl_cloud.size(); i++)
			{
				float dis = fabs(sdl_cloud[i].sdl_point_value.x - center_x);
				distances.push_back(dis);
				if (dis <= init_std)
				{
					sdl_init_x_set.push_back(sdl_cloud[i]);
				}
			}
			center_x = linear_regression_x(sdl_init_x_set);	
			
    		int transfer_cnt = 10, restrain_cnt = 3;
    		float transfer_limit = 0.1, restrain_limit = 0.1, last_val = center_x;
			int mean_center_x;

			cv::Mat breplay =  image_ocv.clone();

			std::vector<SDL_CLOUD_DATA> sdl_point_trans_set;

			int replay_count = 0;
			
			for(int j = 0; j < transfer_cnt; j++)
			{			
				calcu_sdl(sdl_cloud, init_std, center_x, sdl_point_trans_set);


			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);			
			sprintf(str,"/transfer_x_%d_%ld.csv",j,save_count);
			strcat(savejpg_name,str);
			fd = open(savejpg_name,O_RDWR|O_CREAT|O_TRUNC,S_IRWXU);
			if(fd < 0){
				printf("\nFail to open File\n");
				return 0;
			}


			for (int i = 0; i < sdl_point_trans_set.size(); i++)
			{
				sprintf(str,"%d,",sdl_point_trans_set[i].x);
				write(fd,str,strlen(str));
				sprintf(str,"%d,",sdl_point_trans_set[i].y);
				write(fd,str,strlen(str));
				sprintf(str,"%f,",sdl_point_trans_set[i].sdl_point_value.x);
				write(fd,str,strlen(str));
				sprintf(str,"%f,",(-1)*sdl_point_trans_set[i].sdl_point_value.y);
				write(fd,str,strlen(str));
				sprintf(str,"%f\n",sdl_point_trans_set[i].sdl_point_value.z);
				write(fd,str,strlen(str));
			}

			close(fd);

				//printf("\ncenter_x = %f\n",center_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				printf("%d:sdl_point_trans_set.size()=%d,center_x=%f\n",j,sdl_point_trans_set.size(),center_x);
				for (int i = 0; i < sdl_point_trans_set.size(); i++)
				{
					sdl_image.at<uchar>(sdl_point_trans_set[i].y, sdl_point_trans_set[i].x) = 0;
				}

				cv::imshow("sdl_point_trans_set", sdl_image);

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_point_trans_set_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				//cv::imwrite(savejpg_name,sdl_image);

				//key = cv::waitKey();

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_point_trans_set.size(); i++)
        		{
					if(sdl_point_trans_set[i].y < min_y)
						min_y = sdl_point_trans_set[i].y;
					if(sdl_point_trans_set[i].y > max_y)
						max_y = sdl_point_trans_set[i].y;
					if(sdl_point_trans_set[i].x < min_x)
						min_x = sdl_point_trans_set[i].x;
					if(sdl_point_trans_set[i].x > max_x)
						max_x = sdl_point_trans_set[i].x;
				}

				printf("\nsdl_transfer_%d:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",j,min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				//cv::imshow("sdl_image_white", sdl_image);  

				//key = cv::waitKey(); 

				for (int i = 0; i < sdl_cloud.size(); i++)
				{
					//sdl_image.at<uchar>(sdl_maxz_trans_cloud[i].y, sdl_maxz_trans_cloud[i].x) = 0;
					sdl_image.at<uchar>(sdl_cloud[i].y, sdl_cloud[i].x) = 0;
				}



				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

				cv::imshow("sdl_transfer_set", sdl_image);

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_trans_set_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);


				replay_count++;

				//float i = (float)sdl_point_trans_set.size()/sdl_cloud.size();
				float i = (float)replay_count*2/33;
				cv::Scalar c(255.f * i,255.f * i,255);

				cv::rectangle(breplay, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), c, 2);
				cv::imshow("breplay", breplay); 

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_breplay_x_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,breplay);

			
				std::vector<SDL_CLOUD_DATA> sdl_point_line_set;

    			for (int i = 0; i < sdl_cloud.size(); i++)
    			{
					if((fabs(sdl_cloud[i].sdl_point_value.x - center_x)) < 0.02)
						sdl_point_line_set.push_back(sdl_cloud[i]);
				}

				min_y = 1080;
    			max_y = 0;
				min_x = 1920;
    			max_x = 0;

				int sum_sdl_center_x = 0;

        		for (int i = 0; i < sdl_point_line_set.size(); i++)
        		{
					if(sdl_point_line_set[i].y < min_y)
						min_y = sdl_point_line_set[i].y;
					if(sdl_point_line_set[i].y > max_y)
						max_y = sdl_point_line_set[i].y;
					if(sdl_point_line_set[i].x < min_x)
						min_x = sdl_point_line_set[i].x;
					if(sdl_point_line_set[i].x > max_x)
						max_x = sdl_point_line_set[i].x;
					sum_sdl_center_x += sdl_point_line_set[i].x;
				}

				if(sdl_point_line_set.size() > 0)
					mean_center_x = sum_sdl_center_x / sdl_point_line_set.size();

				//printf("\nsdl_transfer_%d:min_y = %d,max_y = %d,min_x = %d,max_x = %d,mean_center_x = %d\n",j,min_y,max_y,min_x,max_x,mean_center_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				//cv::imshow("sdl_image_white", sdl_image);  


				printf("sdl_point_line_set.size()=%d\n",sdl_point_line_set.size());
				for (int i = 0; i < sdl_point_line_set.size(); i++)
				{
					sdl_image.at<uchar>(sdl_point_line_set[i].y, sdl_point_line_set[i].x) = 0;
				}

				if(sdl_point_line_set.size() > 0)
				{
					cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

					cv::line(sdl_image, cv::Point(mean_center_x, 5), cv::Point(mean_center_x, sdl_image.rows - 1), cv::Scalar(0), 2);
				}

				printf("\nj = %d \n",j);


				cv::imshow("sdl_line_cloud", sdl_image);

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_line_cloud_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);

				

				if (j > 0 && (last_val - center_x)*(last_val - center_x) < transfer_limit * init_std) 
            		break;

				last_val = center_x;  

				//key = cv::waitKey();
			}

			
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			float init_std_z = 3.0;
			float center_z = 0.0;

			float sum_init_z = 0.0;
			printf("Z:sdl_point_trans_set.size()=%d\n",sdl_point_trans_set.size());
			for (int i = 0; i < sdl_point_trans_set.size(); i++)
			{
				sum_init_z += sdl_point_trans_set[i].sdl_point_value.z;
			}
			center_z = sum_init_z /(float)sdl_point_trans_set.size();
			
    		transfer_cnt = 10;
    		transfer_limit = 0.1; 
			last_val = center_z;
			std::vector<SDL_CLOUD_DATA> sdl_point_trans_z_set;
			for(int j = 0; j < transfer_cnt; j++)
			{			
				calcu_sdl_z(sdl_point_trans_set, init_std_z, center_z, sdl_point_trans_z_set);


				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);			
				sprintf(str,"/transfer_z_%d_%ld.csv",j,save_count);
				strcat(savejpg_name,str);
				fd = open(savejpg_name,O_RDWR|O_CREAT|O_TRUNC,S_IRWXU);
				if(fd < 0){
					printf("\nFail to open File\n");
					return 0;
				}


				for (int i = 0; i < sdl_point_trans_z_set.size(); i++)
				{
					sprintf(str,"%d,",sdl_point_trans_z_set[i].x);
					write(fd,str,strlen(str));
					sprintf(str,"%d,",sdl_point_trans_z_set[i].y);
					write(fd,str,strlen(str));
					sprintf(str,"%f,",sdl_point_trans_z_set[i].sdl_point_value.x);
					write(fd,str,strlen(str));
					sprintf(str,"%f,",(-1)*sdl_point_trans_z_set[i].sdl_point_value.y);
					write(fd,str,strlen(str));
					sprintf(str,"%f\n",sdl_point_trans_z_set[i].sdl_point_value.z);
					write(fd,str,strlen(str));
				}

				close(fd);

				//printf("\ncenter_x = %f\n",center_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				printf("%d:sdl_point_trans_z_set.size()=%d,center_z=%f\n",j,sdl_point_trans_z_set.size(),center_z);
				for (int i = 0; i < sdl_point_trans_z_set.size(); i++)
				{
					sdl_image.at<uchar>(sdl_point_trans_z_set[i].y, sdl_point_trans_z_set[i].x) = 0;
				}

				cv::imshow("sdl_point_trans_z_set", sdl_image);

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_point_trans_z_set_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				//cv::imwrite(savejpg_name,sdl_image);

				//key = cv::waitKey();

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_point_trans_z_set.size(); i++)
        		{
					if(sdl_point_trans_z_set[i].y < min_y)
						min_y = sdl_point_trans_z_set[i].y;
					if(sdl_point_trans_z_set[i].y > max_y)
						max_y = sdl_point_trans_z_set[i].y;
					if(sdl_point_trans_z_set[i].x < min_x)
						min_x = sdl_point_trans_z_set[i].x;
					if(sdl_point_trans_z_set[i].x > max_x)
						max_x = sdl_point_trans_z_set[i].x;
				}

				printf("\nsdl_transfer_z_%d:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",j,min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				//cv::imshow("sdl_image_white", sdl_image);  

				//key = cv::waitKey(); 

				for (int i = 0; i < sdl_cloud.size(); i++)
				{
					//sdl_image.at<uchar>(sdl_maxz_trans_cloud[i].y, sdl_maxz_trans_cloud[i].x) = 0;
					sdl_image.at<uchar>(sdl_cloud[i].y, sdl_cloud[i].x) = 0;
				}



				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

				cv::imshow("sdl_transfer_z_set", sdl_image);

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_trans_z_set_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);

				//float i = (float)sdl_point_trans_z_set.size()/sdl_cloud.size();
				replay_count++;


				float i = (float)replay_count*2/33;
				cv::Scalar c(255.f * i,255.f * i,255);

				cv::rectangle(breplay, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), c, 2);
				cv::imshow("breplay", breplay); 

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_breplay_z_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,breplay);			
		

				printf("\nz_j = %d \n",j);



				if (j > 0 && (last_val - center_z)*(last_val - center_z) < transfer_limit * init_std_z) 
            		break;

				last_val = center_z;  

				//key = cv::waitKey();
			}







//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			float std_x = init_std;
			last_val = std_x;
			float mean_x = 0.0;

			std::vector<SDL_CLOUD_DATA> sd_res_set;			
			std::vector<SDL_CLOUD_DATA> sdl_trans_cloud;
			//sdl_point_trans_set.clear();
			for (int i = 0; i < sdl_cloud.size(); i++)
        	{
				sdl_trans_cloud.push_back(sdl_cloud[i]);
			//	sdl_point_trans_set.push_back(sdl_cloud[i]);
			}
    		for (int i = 0; i < restrain_cnt; i++)
   			{
		        sd_res_set.clear();
				        		
				printf("%d:sdl_point_trans_set.size()=%d,std_x=%f\n",i,sdl_point_trans_z_set.size(),std_x); 
				for (int i = 0; i < sdl_point_trans_z_set.size(); i++)
        		{            
            		sd_res_set.push_back(sdl_point_trans_z_set[i]);
        		}

        		float sum_x = 0.0;
        		float x_count = 0.0;
        		
				for (int i = 0; i < sd_res_set.size(); i++)
        		{
            		sum_x += sd_res_set[i].sdl_point_value.x;
            		x_count += 1.0;
        		}
        		if(x_count > 0)
            		mean_x = sum_x / x_count;

        		float sum_std_x = 0.0;
        		float std_x_count = 0.0;
    
				for (int i = 0; i < sd_res_set.size(); i++)
        		{
            		sum_std_x += (sd_res_set[i].sdl_point_value.x - mean_x) * (sd_res_set[i].sdl_point_value.x - mean_x);
            		std_x_count += 1.0;
        		}
        		if(std_x_count > 0)
            		std_x = std::sqrt(sum_std_x / (std_x_count - 1));  

					//std_x /= 0.8;   

				printf("%d:sd_res_set.size()=%d,std_x=%f,center_x=%f,mean_x=%f\n",i,sd_res_set.size(),std_x,center_x,mean_x); 
				sdl_point_trans_z_set.clear();
				for (int i = 0; i < sd_res_set.size(); i++)
        		{            		
            		if (fabs(sd_res_set[i].sdl_point_value.x - center_x) <= std_x) 
                		sdl_point_trans_z_set.push_back(sd_res_set[i]);
        		}   

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_point_trans_z_set.size(); i++)
        		{
					if(sdl_point_trans_z_set[i].y < min_y)
						min_y = sdl_point_trans_z_set[i].y;
					if(sdl_point_trans_z_set[i].y > max_y)
						max_y = sdl_point_trans_z_set[i].y;
					if(sdl_point_trans_z_set[i].x < min_x)
						min_x = sdl_point_trans_z_set[i].x;
					if(sdl_point_trans_z_set[i].x > max_x)
						max_x = sdl_point_trans_z_set[i].x;
				}

				//printf("\nsdl_restrain:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				//cv::imshow("sdl_image_white", sdl_image);  

				//key = cv::waitKey(); 

				for (int i = 0; i < sdl_point_trans_z_set.size(); i++)
				{
					//sdl_image.at<uchar>(sdl_maxz_trans_cloud[i].y, sdl_maxz_trans_cloud[i].x) = 0;
					//sdl_image.at<uchar>(sdl_trans_cloud[i].y, sdl_trans_cloud[i].x) = 0;
					sdl_image.at<uchar>(sdl_point_trans_z_set[i].y, sdl_point_trans_z_set[i].x) = 0;
				}

			//sprintf(str,"restrain_%d.csv",i);
			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			sprintf(str,"/restrain_%d_%ld.csv",i,save_count);
			strcat(savejpg_name,str);
			fd = open(savejpg_name,O_RDWR|O_CREAT|O_TRUNC,S_IRWXU);
			if(fd < 0){
				printf("\nFail to open File\n");
				return 0;
			}


			for (int i = 0; i < sdl_point_trans_z_set.size(); i++)
			{
				sprintf(str,"%d,",sdl_point_trans_z_set[i].x);
				write(fd,str,strlen(str));
				sprintf(str,"%d,",sdl_point_trans_z_set[i].y);
				write(fd,str,strlen(str));
				sprintf(str,"%f,",sdl_point_trans_z_set[i].sdl_point_value.x);
				write(fd,str,strlen(str));
				sprintf(str,"%f,",(-1)*sdl_point_trans_z_set[i].sdl_point_value.y);
				write(fd,str,strlen(str));
				sprintf(str,"%f\n",sdl_point_trans_z_set[i].sdl_point_value.z);
				write(fd,str,strlen(str));
			}

			close(fd);


				

				cv::Mat med_img_res;

				cv::medianBlur(sdl_image,med_img_res,5);
				
				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);		
				
				min_y = 1080;
    			max_y = 0;
				min_x = 1920;
    			max_x = 0;

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{	
						if(med_img_res.at<uchar>(j, i) < 150)
						{
							if(j < min_y)
								min_y = j;
							if(j > max_y)
								max_y = j;
							if(i < min_x)
								min_x = i;
							if(i > max_x)
								max_x = i;	
						}
					}
				}								

				//printf("\nmed_img_res:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",min_y,max_y,min_x,max_x);

				cv::rectangle(med_img_res, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);
				//cv::imshow("medianBlur", med_img_res); 

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_cloud_med_");
				sprintf(string,"%d_",i);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				//cv::imwrite(savejpg_name,med_img_res);

				
				cv::Mat results =  image_ocv.clone();
				cv::rectangle(results, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0,255,0), 2);
				cv::imshow("results", results); 

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_results_");
				sprintf(string,"%d_",i);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,results);


				replay_count++;


				float ii = (float)i / 3;
				cv::Scalar c(255.f * ii,255,255.f * ii);

				cv::rectangle(breplay, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), c, 2);
				cv::imshow("breplay", breplay); 

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_breplay_r_");
				sprintf(string,"%d_",i);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,breplay);	

				//printf("\ni = %d \n",i);


				char restrain_n[100];
				sprintf(restrain_n,"sdl_cloud_res%d",i);
				cv::imshow(restrain_n, sdl_image);    

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_cloud_restr_");
				sprintf(string,"%d_",i);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);

				//key = cv::waitKey();                

        		if (i > 0 && (last_val - std_x)*(last_val - std_x) < restrain_limit * std_x) 
            		break;
        		last_val = std_x;       
				//printf("%d:sdl_maxz_in_cloud.size()=%d,std_max_x=%f\n",i,sdl_maxz_in_cloud.size(),std_max_x);      
    		}


		/*	std::vector<SDL_CLOUD_DATA> sdl_point_line_set;
    		for (int i = 0; i < sdl_point_nogrand_set.size(); i++)
    		{
				if((sdl_point_nogrand_set[i].sdl_point_value.x < 0.05)&&(sdl_point_nogrand_set[i].sdl_point_value.x > -0.05)&&(sdl_point_nogrand_set[i].sdl_point_value.y < 	
		                     0.05)&&(sdl_point_nogrand_set[i].sdl_point_value.y > -0.32))
					sdl_point_line_set.push_back(sdl_point_nogrand_set[i]);
    		}

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}

			printf("sdl_point_line_set.size()=%d\n",sdl_point_line_set.size());
			for (int i = 0; i < sdl_point_line_set.size(); i++)
			{
				sdl_image.at<uchar>(sdl_point_line_set[i].y, sdl_point_line_set[i].x) = 0;
			}

			cv::imshow("sdl_line_cloud", sdl_image);

			vector<float> innerx;
 			vector<float> innery;
			for (int i = 0; i < sdl_point_line_set.size(); i++)
			{
				innerx.push_back(sdl_point_line_set[i].sdl_point_value.x);
				innery.push_back(sdl_point_line_set[i].sdl_point_value.y);
			}			

			float k,b;
    		linear_regression(innerx, innery, k, b);

			printf("\nk = %f,b = %f\n",k,b);

			std::vector<SDL_CLOUD_DATA> sdl_point_trans_in_set;

    		for (int i = 0; i < sdl_point_nogrand_set.size(); i++)
    		{										
				sdl_point_trans_in_set.push_back(sdl_point_nogrand_set[i]);
			}
			
			float init_std = 0.5;
			std::vector<SDL_CLOUD_DATA> sdl_point_trans_set;
			
			calcu_sdl(sdl_point_trans_in_set, init_std, k, b, sdl_point_trans_set);

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}

			printf("sdl_point_trans_set.size()=%d\n",sdl_point_trans_set.size());
			for (int i = 0; i < sdl_point_trans_set.size(); i++)
			{
				sdl_image.at<uchar>(sdl_point_trans_set[i].y, sdl_point_trans_set[i].x) = 0;
			}

			cv::imshow("sdl_trans_cloud", sdl_image);

			for (int i = 0; i < sdl_point_trans_set.size(); i++)
			{
				sdl_image.at<uchar>(sdl_point_trans_set[i].y, sdl_point_trans_set[i].x) = 0;
			}*/
			
		/*	int min_y = 5;

			cv::line(sdl_image, cv::Point((min_y - b) / k, min_y), cv::Point((sdl_image.rows - 1 - b) / k, sdl_image.rows - 1), cv::Scalar(0), 2);

			cv::imshow("sdl_cloud1", sdl_image);*/

			//key = cv::waitKey();

			save_count++;

#endif

		//	cv::imshow("Image1", depthim);
	
		//	cv::imshow("Image", image_ocv);

			
			
			gettimeofday(&end1,NULL);
			run_timeuse = 1000000 * (end1.tv_sec - start1.tv_sec) + end1.tv_usec - start1.tv_usec;
			run_timeuse /= 1000;
			printf("\nrun_timeuse = %fms\n",run_timeuse);

			key = cv::waitKey(10);
			printf("\n%d:finish\n",save_count-1);

			//key = cv::waitKey();
		}
	}

	close(sock_fd);
    zed.close();
    return 0;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
	int cv_type = -1;
	switch (input.getDataType())
	{
		case (MAT_TYPE::F32_C1): cv_type = CV_32FC1; break;
		case (MAT_TYPE::F32_C2): cv_type = CV_32FC2; break;
		case (MAT_TYPE::F32_C3): cv_type = CV_32FC3; break;
		case (MAT_TYPE::F32_C4): cv_type = CV_32FC4; break;
		case (MAT_TYPE::U8_C1): cv_type = CV_8UC1; break;
		case (MAT_TYPE::U8_C2): cv_type = CV_8UC2; break;
		case (MAT_TYPE::U8_C3): cv_type = CV_8UC3; break;
		case (MAT_TYPE::U8_C4): cv_type = CV_8UC4; break;
		default: break;
	}

	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}
