
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
	init_params.camera_resolution = RESOLUTION::HD1080;
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

    int new_width 	= image_size.width ;// 2;
    int new_height 	= image_size.height ;// 2;

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
	char string[50];


   
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

			

			cv::cvtColor(depth_image_ocv, depthim, CV_RGBA2GRAY);
			
			cv::cvtColor(image_ocv, image_ocv_copy, 1);

			int x, y, z;

			sl::float4 point_cloud_value;

			SDL_CLOUD_DATA sdl_point;

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
						Z1[i][j] = sqrt(point_cloud_value.x * point_cloud_value.x + point_cloud_value.y * point_cloud_value.y + point_cloud_value.z * point_cloud_value.z);
						Z0[i][j] = point_cloud_value.z;//Z1[i][j];						
						Z[i][j] = point_cloud_value.z;

						sdl_point.x = i;
						sdl_point.y = j;
						sdl_point.sdl_point_value = point_cloud_value;
						sdl_point_set.push_back(sdl_point);
					

						if (Z1[i][j] < 39)//除去无效障碍，只取相对车来说的近景障碍
						{
							depthim.at<uchar>(j, i) = 0;
						}
						else
						{
							depthim.at<uchar>(j, i) = 255;
						}
					}
					else
					{
						depthim.at<uchar>(j, i) = 255;
						Z1[i][j] = 40;
						img[i][j] = 40;
						Z[i][j] = 40;
						Z0[i][j] = 40;
					}					

				}
			}

			printf("sdl_point_set.size()=%d\n",sdl_point_set.size());

			for (int i = 0; i < sdl_point_set.size(); i++)
			{
				sdl_image.at<uchar>(sdl_point_set[i].y, sdl_point_set[i].x) = 0;
			}

			cv::imshow("sdl_point_set", sdl_image);
			key = cv::waitKey();

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}

			float grid_size = 0.5;
			float height_thresh = 0.3;
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

			key = cv::waitKey();


			//	cv::imshow("Image1", depthim);
			//SDL点云处理
#if 1
			//ROI范围
			std::vector<SDL_CLOUD_DATA> sdl_point_roi_set;
    		for (int i = 0; i < sdl_point_nogrand_set.size(); i++)
    		{
				if((sdl_point_nogrand_set[i].sdl_point_value.x < 1.5)&&(sdl_point_nogrand_set[i].sdl_point_value.x > -1.0)&&(sdl_point_nogrand_set[i].sdl_point_value.y < 	
		                     1.0)&&(sdl_point_nogrand_set[i].sdl_point_value.y > -1.0))
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

			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/video_sdl_cloud_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);
			


    		float min_y = std::numeric_limits<float>::max();
    		float max_y = std::numeric_limits<float>::lowest();
    		float in_y;

			

    		for (int i = 0; i < sdl_cloud.size(); i++)
    		{
		        min_y = std::min(min_y, sdl_point_roi_set[i].sdl_point_value.y);
        		max_y = std::max(max_y, sdl_point_roi_set[i].sdl_point_value.y);
    		}

			
			min_y += 0.5;
    		in_y = (min_y + max_y) / 2;  

			printf("min_y=%f,max_y=%f,in_y=%f\n",min_y,max_y,in_y);
    		float y_interval = 0.1;
    		float mean_max_x = 0.0;
    		float mean_min_x = 0.0;
    		float mean_in_x = 0.0;      
    		float std_max_x = 2.5;
    		float std_min_x = 1.5;
    		float std_in_x = 1.5;
    		float sum_max_x = 0.0;
    		float sum_min_x = 0.0;
    		float sum_in_x = 0.0;
    		float max_count = 0.0;
    		float min_count = 0.0;
    		float in_count = 0.0;

    		int transfer_cnt = 10, restrain_cnt = 10;
    		float transfer_limit = 0.1, restrain_limit = 0.1, last_val = 0;
    		std::vector<SDL_CLOUD_DATA> sdl_maxz_cloud;
    		std::vector<SDL_CLOUD_DATA> sdl_minz_cloud;
    		std::vector<SDL_CLOUD_DATA> sdl_inz_cloud;

    		for (int i = 0; i < sdl_cloud.size(); i++)
    		{
        		if(sdl_cloud[i].sdl_point_value.y < max_y && sdl_cloud[i].sdl_point_value.y > ( max_y - y_interval))
        		{
            		sum_max_x += sdl_cloud[i].sdl_point_value.x;
            		max_count += 1.0;
            		sdl_maxz_cloud.push_back(sdl_cloud[i]);
        		}
        		else if(sdl_cloud[i].sdl_point_value.y > min_y && sdl_cloud[i].sdl_point_value.y < ( min_y + y_interval))
        		{
            		sum_min_x += sdl_cloud[i].sdl_point_value.x;
            		min_count += 1.0;
            		sdl_minz_cloud.push_back(sdl_cloud[i]);
        		}
        		else if(sdl_cloud[i].sdl_point_value.y > (in_y - y_interval / 2) && sdl_cloud[i].sdl_point_value.y < ( in_y + y_interval / 2))
        		{
            		sum_in_x += sdl_cloud[i].sdl_point_value.x;;
            		in_count += 1.0;
            		sdl_inz_cloud.push_back(sdl_cloud[i]);
        		}
    		}

    		if(max_count > 0)
        		mean_max_x = sum_max_x / max_count;
    		if(min_count > 0)
        		mean_min_x = sum_min_x / min_count;
    		if(in_count > 0)
        		mean_in_x =  sum_in_x / in_count;		

        	float sum_std_max_x = 0.0;
        	float std_max_count = 0.0;
    
			for (int i = 0; i < sdl_maxz_cloud.size(); i++)
        	{
            	sum_std_max_x += (sdl_maxz_cloud[i].sdl_point_value.x - mean_max_x) * (sdl_maxz_cloud[i].sdl_point_value.x - mean_max_x);
            	std_max_count += 1.0;
        	}
        	if(std_max_count > 0)
            	std_max_x = std::sqrt(sum_std_max_x / (std_max_count - 1));     

			printf("sdl_maxz_cloud.size()=%d,std_max_x=%f\n",sdl_maxz_cloud.size(),std_max_x); 



			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}

			printf("sdl_maxz_cloud.size()=%d,mean_max_x=%f\n",sdl_maxz_cloud.size(),mean_max_x);
			for (int i = 0; i < sdl_maxz_cloud.size(); i++)
			{
				sdl_image.at<uchar>(sdl_maxz_cloud[i].y, sdl_maxz_cloud[i].x) = 0;
			}

			cv::imshow("sdl_maxz_cloud", sdl_image);	

			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/video_sdl_maxz_cloud_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);

        	float sum_std_min_x = 0.0;
        	float std_min_count = 0.0;
    
			for (int i = 0; i < sdl_minz_cloud.size(); i++)
        	{
            	sum_std_min_x += (sdl_minz_cloud[i].sdl_point_value.x - mean_min_x) * (sdl_minz_cloud[i].sdl_point_value.x - mean_min_x);
            	std_min_count += 1.0;
        	}
        	if(std_min_count > 0)
            	std_min_x = std::sqrt(sum_std_min_x / (std_min_count - 1));     

			printf("sdl_minz_cloud.size()=%d,std_min_x=%f\n",sdl_minz_cloud.size(),std_min_x); 


			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}

			printf("sdl_minz_cloud.size()=%d,mean_min_x=%f\n",sdl_minz_cloud.size(),mean_min_x);
			for (int i = 0; i < sdl_minz_cloud.size(); i++)
			{
				sdl_image.at<uchar>(sdl_minz_cloud[i].y, sdl_minz_cloud[i].x) = 0;
			}

			cv::imshow("sdl_minz_cloud", sdl_image);

			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/sdl_minz_cloud_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);

        	float sum_std_in_x = 0.0;
        	float std_in_count = 0.0;
    
			for (int i = 0; i < sdl_inz_cloud.size(); i++)
        	{
            	sum_std_in_x += (sdl_inz_cloud[i].sdl_point_value.x - mean_in_x) * (sdl_inz_cloud[i].sdl_point_value.x - mean_in_x);
            	std_in_count += 1.0;
        	}
        	if(std_in_count > 0)
            	std_in_x = std::sqrt(sum_std_in_x / (std_in_count - 1));     

			printf("sdl_inz_cloud.size()=%d,std_in_x=%f\n",sdl_inz_cloud.size(),std_in_x); 


			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}
			printf("sdl_inz_cloud.size()=%d,mean_in_x=%f\n",sdl_inz_cloud.size(),mean_in_x);
			for (int i = 0; i < sdl_inz_cloud.size(); i++)
			{
				sdl_image.at<uchar>(sdl_inz_cloud[i].y, sdl_inz_cloud[i].x) = 0;
			}

			cv::imshow("sdl_inz_cloud", sdl_image);

			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/sdl_inz_cloud_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);

			// 正面最高处横向最大概率迭代
    		std::vector<SDL_CLOUD_DATA> sdl_maxz_in_cloud;
    		last_val = mean_max_x;
   	 		for (int j = 0; j < transfer_cnt; j++)
    		{
				printf("sdl_maxz_cloud.size()=%d,std_max_x=%f\n",sdl_maxz_cloud.size(),std_max_x);
				sdl_maxz_in_cloud.clear();
        		for (int i = 0; i < sdl_maxz_cloud.size(); i++)
        		{
            		
					float max_in = fabs(sdl_maxz_cloud[i].sdl_point_value.x - mean_max_x);
					
					//printf("%d:max_in=%f,std_max_x=%f\n",i,max_in,std_max_x);
            		if (fabs(sdl_maxz_cloud[i].sdl_point_value.x - mean_max_x) <= std_max_x) 
                		sdl_maxz_in_cloud.push_back(sdl_maxz_cloud[i]);
					//printf("sdl_maxz_in_cloud.size()=%d\n",sdl_maxz_in_cloud.size());
        		}

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_maxz_in_cloud.size(); i++)
        		{
					if(sdl_maxz_in_cloud[i].y < min_y)
						min_y = sdl_maxz_in_cloud[i].y;
					if(sdl_maxz_in_cloud[i].y > max_y)
						max_y = sdl_maxz_in_cloud[i].y;
					if(sdl_maxz_in_cloud[i].x < min_x)
						min_x = sdl_maxz_in_cloud[i].x;
					if(sdl_maxz_in_cloud[i].x > max_x)
						max_x = sdl_maxz_in_cloud[i].x;
		        	//min_y = std::min(min_y, sdl_maxz_in_cloud[i].y);
        		//	max_y = std::max(max_y, sdl_maxz_in_cloud[i].y);					
		        //	min_x = std::min(min_x, sdl_maxz_in_cloud[i].x);
        		//	max_x = std::max(max_x, sdl_maxz_in_cloud[i].x);					
				}

				printf("\nsdl_maxz_in_cloud_transfer:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				cv::imshow("sdl_image_white", sdl_image);  

				key = cv::waitKey();

				for (int i = 0; i < sdl_maxz_in_cloud.size(); i++)
				{
					sdl_image.at<uchar>(sdl_maxz_in_cloud[i].y, sdl_maxz_in_cloud[i].x) = 0;
				}



				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

				printf("\nj = %d \n",j);

				char transfer_n[100];
				sprintf(transfer_n,"sdl_maxz_cloud_trans%d",j);
				cv::imshow(transfer_n, sdl_image);    

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_maxz_cloud_tran_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);

				key = cv::waitKey();

        		sum_max_x = 0.0;
       		 	max_count = 0.0;
       			
				for (int i = 0; i < sdl_maxz_in_cloud.size(); i++)
        		{
            		sum_max_x += sdl_maxz_in_cloud[i].sdl_point_value.x;
            		max_count += 1.0;
        		}
        		if(max_count > 0)
            		mean_max_x = sum_max_x / max_count;

				printf("mean_max_x=%f\n",mean_max_x);

        		if (j > 0 && (last_val - mean_max_x)*(last_val - mean_max_x) < transfer_limit * std_max_x) 
            		break;
        		last_val = mean_max_x;   



    		}

			last_val = std_max_x;

			std::vector<SDL_CLOUD_DATA> sdl_maxz_trans_cloud;
			for (int i = 0; i < sdl_maxz_cloud.size(); i++)
        	{
				sdl_maxz_trans_cloud.push_back(sdl_maxz_cloud[i]);
			}
    		for (int i = 0; i < restrain_cnt; i++)
   			{
		        sdl_maxz_cloud.clear();
				        		
				printf("%d:sdl_maxz_in_cloud.size()=%d,std_max_x=%f\n",i,sdl_maxz_in_cloud.size(),std_max_x); 
				for (int i = 0; i < sdl_maxz_in_cloud.size(); i++)
        		{            
            		sdl_maxz_cloud.push_back(sdl_maxz_in_cloud[i]);
        		}

        		sum_max_x = 0.0;
        		max_count = 0.0;
        		
				for (int i = 0; i < sdl_maxz_cloud.size(); i++)
        		{
            		sum_max_x += sdl_maxz_cloud[i].sdl_point_value.x;
            		max_count += 1.0;
        		}
        		if(max_count > 0)
            		mean_max_x = sum_max_x / max_count;

        		float sum_std_max_x = 0.0;
        		float std_max_count = 0.0;
    
				for (int i = 0; i < sdl_maxz_cloud.size(); i++)
        		{
            		sum_std_max_x += (sdl_maxz_cloud[i].sdl_point_value.x - mean_max_x) * (sdl_maxz_cloud[i].sdl_point_value.x - mean_max_x);
            		std_max_count += 1.0;
        		}
        		if(std_max_count > 0)
            		std_max_x = std::sqrt(sum_std_max_x / (std_max_count - 1));     

				printf("%d:sdl_maxz_cloud.size()=%d,std_max_x=%f\n",i,sdl_maxz_cloud.size(),std_max_x); 
				sdl_maxz_in_cloud.clear();
				for (int i = 0; i < sdl_maxz_cloud.size(); i++)
        		{            		
            		if (fabs(sdl_maxz_cloud[i].sdl_point_value.x - mean_max_x) <= std_max_x) 
                		sdl_maxz_in_cloud.push_back(sdl_maxz_cloud[i]);
        		}   

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_maxz_in_cloud.size(); i++)
        		{
					if(sdl_maxz_in_cloud[i].y < min_y)
						min_y = sdl_maxz_in_cloud[i].y;
					if(sdl_maxz_in_cloud[i].y > max_y)
						max_y = sdl_maxz_in_cloud[i].y;
					if(sdl_maxz_in_cloud[i].x < min_x)
						min_x = sdl_maxz_in_cloud[i].x;
					if(sdl_maxz_in_cloud[i].x > max_x)
						max_x = sdl_maxz_in_cloud[i].x;
		        	//min_y = std::min(min_y, sdl_maxz_in_cloud[i].y);
        		//	max_y = std::max(max_y, sdl_maxz_in_cloud[i].y);					
		        //	min_x = std::min(min_x, sdl_maxz_in_cloud[i].x);
        		//	max_x = std::max(max_x, sdl_maxz_in_cloud[i].x);					
				}

				printf("\nsdl_maxz_in_cloud_restrain:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				cv::imshow("sdl_image_white", sdl_image);  

				key = cv::waitKey(); 

				for (int i = 0; i < sdl_maxz_in_cloud.size(); i++)
				{
					//sdl_image.at<uchar>(sdl_maxz_trans_cloud[i].y, sdl_maxz_trans_cloud[i].x) = 0;
					sdl_image.at<uchar>(sdl_maxz_in_cloud[i].y, sdl_maxz_in_cloud[i].x) = 0;
				}



				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

				printf("\ni = %d \n",i);


				char restrain_n[100];
				sprintf(restrain_n,"sdl_maxz_cloud_res%d",i);
				cv::imshow(restrain_n, sdl_image);    

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_maxz_cloud_restr_");
				sprintf(string,"%d_",i);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);

				key = cv::waitKey();                

        		if (i > 0 && (last_val - std_max_x)*(last_val - std_max_x) < restrain_limit * std_max_x) 
            		break;
        		last_val = std_max_x;       
				printf("%d:sdl_maxz_in_cloud.size()=%d,std_max_x=%f\n",i,sdl_maxz_in_cloud.size(),std_max_x);      
    		}

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}	

			cv::imshow("sdl_image_white", sdl_image);  

			key = cv::waitKey(); 		

			for (int i = 0; i < sdl_maxz_in_cloud.size(); i++)
			{
				sdl_image.at<uchar>(sdl_maxz_in_cloud[i].y, sdl_maxz_in_cloud[i].x) = 0;
			}

			cv::imshow("sdl_max_image1", sdl_image);

			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/sdl_max_image1_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);

			printf("\nsdl_maxz finish!\n");

			key = cv::waitKey();  
////////////////////////////////////////////////////////////////////////////////////////////////////////////
    		
			std::vector<SDL_CLOUD_DATA> sdl_minz_in_cloud;
    		last_val = mean_min_x;
   	 		for (int j = 0; j < transfer_cnt; j++)
    		{
				printf("sdl_minz_cloud.size()=%d,std_min_x=%f\n",sdl_minz_cloud.size(),std_min_x);
				sdl_minz_in_cloud.clear();
        		for (int i = 0; i < sdl_minz_cloud.size(); i++)
        		{
            		
					float min_in = fabs(sdl_minz_cloud[i].sdl_point_value.x - mean_min_x);
					
					//printf("%d:min_in=%f,std_min_x=%f\n",i,min_in,std_min_x);
            		if (fabs(sdl_minz_cloud[i].sdl_point_value.x - mean_min_x) <= std_min_x) 
                		sdl_minz_in_cloud.push_back(sdl_minz_cloud[i]);
					//printf("sdl_minz_in_cloud.size()=%d\n",sdl_minz_in_cloud.size());
        		}

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_minz_in_cloud.size(); i++)
        		{
					if(sdl_minz_in_cloud[i].y < min_y)
						min_y = sdl_minz_in_cloud[i].y;
					if(sdl_minz_in_cloud[i].y > max_y)
						max_y = sdl_minz_in_cloud[i].y;
					if(sdl_minz_in_cloud[i].x < min_x)
						min_x = sdl_minz_in_cloud[i].x;
					if(sdl_minz_in_cloud[i].x > max_x)
						max_x = sdl_minz_in_cloud[i].x;
		        	//min_y = std::min(min_y, sdl_maxz_in_cloud[i].y);
        		//	max_y = std::max(max_y, sdl_maxz_in_cloud[i].y);					
		        //	min_x = std::min(min_x, sdl_maxz_in_cloud[i].x);
        		//	max_x = std::max(max_x, sdl_maxz_in_cloud[i].x);					
				}

				printf("\nsdl_minz_in_cloud_transfer:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}	

				cv::imshow("sdl_image_white", sdl_image);  

				key = cv::waitKey(); 	

				for (int i = 0; i < sdl_minz_in_cloud.size(); i++)
				{
					sdl_image.at<uchar>(sdl_minz_in_cloud[i].y, sdl_minz_in_cloud[i].x) = 0;
				}



				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

				printf("\nj = %d \n",j);

				char transfer_n[100];
				sprintf(transfer_n,"sdl_minz_cloud_trans%d",j);
				cv::imshow(transfer_n, sdl_image);    

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_minz_cloud_tran_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);

				key = cv::waitKey(); 

        		sum_min_x = 0.0;
       		 	min_count = 0.0;
       			
				for (int i = 0; i < sdl_minz_in_cloud.size(); i++)
        		{
            		sum_min_x += sdl_minz_in_cloud[i].sdl_point_value.x;
            		min_count += 1.0;
        		}
        		if(min_count > 0)
            		mean_min_x = sum_min_x / min_count;

				printf("mean_min_x=%f\n",mean_min_x);

        		if (j > 0 && (last_val - mean_min_x)*(last_val - mean_min_x) < transfer_limit * std_min_x) 
            		break;
        		last_val = mean_min_x;       
    		}

			last_val = std_min_x;
			std::vector<SDL_CLOUD_DATA> sdl_minz_trans_cloud;
			for (int i = 0; i < sdl_minz_cloud.size(); i++)
        	{
				sdl_minz_trans_cloud.push_back(sdl_minz_cloud[i]);
			}
    		for (int i = 0; i < restrain_cnt; i++)
   			{
		        sdl_minz_cloud.clear();
				        		
				printf("%d:sdl_minz_in_cloud.size()=%d,std_min_x=%f\n",i,sdl_minz_in_cloud.size(),std_min_x); 

				for (int i = 0; i < sdl_minz_in_cloud.size(); i++)
        		{            
            		sdl_minz_cloud.push_back(sdl_minz_in_cloud[i]);
        		}

        		sum_min_x = 0.0;
        		min_count = 0.0;
        		
				for (int i = 0; i < sdl_minz_cloud.size(); i++)
        		{
            		sum_min_x += sdl_minz_cloud[i].sdl_point_value.x;
            		min_count += 1.0;
        		}
        		if(min_count > 0)
            		mean_min_x = sum_min_x / min_count;

        		float sum_std_min_x = 0.0;
        		float std_min_count = 0.0;
    
				for (int i = 0; i < sdl_minz_cloud.size(); i++)
        		{
            		sum_std_min_x += (sdl_minz_cloud[i].sdl_point_value.x - mean_min_x) * (sdl_minz_cloud[i].sdl_point_value.x - mean_min_x);
            		std_min_count += 1.0;
        		}
        		if(std_min_count > 0)
            		std_min_x = std::sqrt(sum_std_min_x / (std_min_count - 1));     

				printf("%d:sdl_minz_cloud.size()=%d,std_min_x=%f\n",i,sdl_minz_cloud.size(),std_min_x); 
				sdl_minz_in_cloud.clear();
				for (int i = 0; i < sdl_minz_cloud.size(); i++)
        		{            		
            		if (fabs(sdl_minz_cloud[i].sdl_point_value.x - mean_min_x) <= std_min_x) 
                		sdl_minz_in_cloud.push_back(sdl_minz_cloud[i]);
        		}   

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_minz_in_cloud.size(); i++)
        		{
					if(sdl_minz_in_cloud[i].y < min_y)
						min_y = sdl_minz_in_cloud[i].y;
					if(sdl_minz_in_cloud[i].y > max_y)
						max_y = sdl_minz_in_cloud[i].y;
					if(sdl_minz_in_cloud[i].x < min_x)
						min_x = sdl_minz_in_cloud[i].x;
					if(sdl_minz_in_cloud[i].x > max_x)
						max_x = sdl_minz_in_cloud[i].x;
		        	//min_y = std::min(min_y, sdl_maxz_in_cloud[i].y);
        		//	max_y = std::max(max_y, sdl_maxz_in_cloud[i].y);					
		        //	min_x = std::min(min_x, sdl_maxz_in_cloud[i].x);
        		//	max_x = std::max(max_x, sdl_maxz_in_cloud[i].x);					
				}

				printf("\nsdl_minz_in_cloud_restrain:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				cv::imshow("sdl_image_white", sdl_image);  

				key = cv::waitKey(); 



				for (int i = 0; i < sdl_minz_in_cloud.size(); i++)
				{
					sdl_image.at<uchar>(sdl_minz_in_cloud[i].y, sdl_minz_in_cloud[i].x) = 0;
				}

				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

				printf("\ni = %d \n",i);

				char restrain_n[100];
				sprintf(restrain_n,"sdl_minz_cloud_res%d",i);
				cv::imshow(restrain_n, sdl_image);    

				key = cv::waitKey();

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_minz_cloud_restr_");
				sprintf(string,"%d_",i);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);	

				key = cv::waitKey(); 			                

        		if (i > 0 && (last_val - std_min_x)*(last_val - std_min_x) < restrain_limit * std_min_x) 
            		break;
        		last_val = std_min_x;       
				printf("%d:sdl_minz_in_cloud.size()=%d,std_min_x=%f\n",i,sdl_minz_in_cloud.size(),std_min_x);      
    		}

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}	

			cv::imshow("sdl_image_white", sdl_image);  

			key = cv::waitKey(); 			

			for (int i = 0; i < sdl_minz_in_cloud.size(); i++)
			{
				sdl_image.at<uchar>(sdl_minz_in_cloud[i].y, sdl_minz_in_cloud[i].x) = 0;
			}

			cv::imshow("sdl_min_image1", sdl_image);
			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/sdl_min_image1_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);

			printf("\nsdl_minz finish!\n");

			key = cv::waitKey();
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			std::vector<SDL_CLOUD_DATA> sdl_inz_in_cloud;
    		last_val = mean_in_x;
   	 		for (int j = 0; j < transfer_cnt; j++)
    		{
				printf("sdl_inz_cloud.size()=%d,std_in_x=%f\n",sdl_inz_cloud.size(),std_in_x);
				sdl_inz_in_cloud.clear();
        		for (int i = 0; i < sdl_inz_cloud.size(); i++)
        		{
            		
					float in_in = fabs(sdl_inz_cloud[i].sdl_point_value.x - mean_in_x);
					
					//printf("%d:in_in=%f,std_in_x=%f\n",i,in_in,std_in_x);
            		if (fabs(sdl_inz_cloud[i].sdl_point_value.x - mean_in_x) <= std_in_x) 
                		sdl_inz_in_cloud.push_back(sdl_inz_cloud[i]);
					//printf("sdl_inz_in_cloud.size()=%d\n",sdl_inz_in_cloud.size());
        		}

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_inz_in_cloud.size(); i++)
        		{
					if(sdl_inz_in_cloud[i].y < min_y)
						min_y = sdl_inz_in_cloud[i].y;
					if(sdl_inz_in_cloud[i].y > max_y)
						max_y = sdl_inz_in_cloud[i].y;
					if(sdl_inz_in_cloud[i].x < min_x)
						min_x = sdl_inz_in_cloud[i].x;
					if(sdl_inz_in_cloud[i].x > max_x)
						max_x = sdl_inz_in_cloud[i].x;
		        	//min_y = std::min(min_y, sdl_maxz_in_cloud[i].y);
        		//	max_y = std::max(max_y, sdl_maxz_in_cloud[i].y);					
		        //	min_x = std::min(min_x, sdl_maxz_in_cloud[i].x);
        		//	max_x = std::max(max_x, sdl_maxz_in_cloud[i].x);					
				}

				printf("\nsdl_inz_in_cloud_transfer:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}	

				cv::imshow("sdl_image_white", sdl_image);  

				key = cv::waitKey(); 

				for (int i = 0; i < sdl_inz_in_cloud.size(); i++)
				{
					sdl_image.at<uchar>(sdl_inz_in_cloud[i].y, sdl_inz_in_cloud[i].x) = 0;
				}



				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

				printf("\nj = %d \n",j);

				char transfer_n[100];
				sprintf(transfer_n,"sdl_inz_cloud_trans%d",j);
				cv::imshow(transfer_n, sdl_image);    

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_inz_cloud_tran_");
				sprintf(string,"%d_",j);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);

				key = cv::waitKey(); 

        		sum_in_x = 0.0;
       		 	in_count = 0.0;
       			
				for (int i = 0; i < sdl_inz_in_cloud.size(); i++)
        		{
            		sum_in_x += sdl_inz_in_cloud[i].sdl_point_value.x;
            		in_count += 1.0;
        		}
        		if(in_count > 0)
            		mean_in_x = sum_in_x / in_count;

				printf("mean_in_x=%f\n",mean_in_x);

        		if (j > 0 && (last_val - mean_in_x)*(last_val - mean_in_x) < transfer_limit * std_in_x) 
            		break;
        		last_val = mean_in_x;       
    		}

			last_val = std_in_x;
			std::vector<SDL_CLOUD_DATA> sdl_inz_trans_cloud;
			for (int i = 0; i < sdl_inz_cloud.size(); i++)
        	{
				sdl_inz_trans_cloud.push_back(sdl_inz_cloud[i]);
			}
    		for (int i = 0; i < restrain_cnt; i++)
   			{
		        sdl_inz_cloud.clear();
				        		
				printf("%d:sdl_inz_in_cloud.size()=%d,std_in_x=%f\n",i,sdl_inz_in_cloud.size(),std_in_x); 
				for (int i = 0; i < sdl_inz_in_cloud.size(); i++)
        		{            
            		sdl_inz_cloud.push_back(sdl_inz_in_cloud[i]);
        		}

        		sum_in_x = 0.0;
        		in_count = 0.0;
        		
				for (int i = 0; i < sdl_inz_cloud.size(); i++)
        		{
            		sum_in_x += sdl_inz_cloud[i].sdl_point_value.x;
            		in_count += 1.0;
        		}
        		if(in_count > 0)
            		mean_in_x = sum_in_x / in_count;

        		float sum_std_in_x = 0.0;
        		float std_in_count = 0.0;
    
				for (int i = 0; i < sdl_inz_cloud.size(); i++)
        		{
            		sum_std_in_x += (sdl_inz_cloud[i].sdl_point_value.x - mean_in_x) * (sdl_inz_cloud[i].sdl_point_value.x - mean_in_x);
            		std_in_count += 1.0;
        		}
        		if(std_in_count > 0)
            		std_in_x = std::sqrt(sum_std_in_x / (std_in_count - 1));     

				printf("%d:sdl_inz_cloud.size()=%d,std_in_x=%f\n",i,sdl_inz_cloud.size(),std_in_x); 
				sdl_inz_in_cloud.clear();
				for (int i = 0; i < sdl_inz_cloud.size(); i++)
        		{            		
            		if (fabs(sdl_inz_cloud[i].sdl_point_value.x - mean_in_x) <= std_in_x) 
                		sdl_inz_in_cloud.push_back(sdl_inz_cloud[i]);
        		}   

				int min_y = 1080;
    			int max_y = 0;
				int min_x = 1920;
    			int max_x = 0;

        		for (int i = 0; i < sdl_inz_in_cloud.size(); i++)
        		{
					if(sdl_inz_in_cloud[i].y < min_y)
						min_y = sdl_inz_in_cloud[i].y;
					if(sdl_inz_in_cloud[i].y > max_y)
						max_y = sdl_inz_in_cloud[i].y;
					if(sdl_inz_in_cloud[i].x < min_x)
						min_x = sdl_inz_in_cloud[i].x;
					if(sdl_inz_in_cloud[i].x > max_x)
						max_x = sdl_inz_in_cloud[i].x;
		        	//min_y = std::min(min_y, sdl_maxz_in_cloud[i].y);
        		//	max_y = std::max(max_y, sdl_maxz_in_cloud[i].y);					
		        //	min_x = std::min(min_x, sdl_maxz_in_cloud[i].x);
        		//	max_x = std::max(max_x, sdl_maxz_in_cloud[i].x);					
				}

				printf("\nsdl_inz_in_cloud_restrain:min_y = %d,max_y = %d,min_x = %d,max_x = %d\n",min_y,max_y,min_x,max_x);

				for (int i = 0; i < new_width; i++)
				{
					for (int j = 0; j < new_height; j++)
					{
						sdl_image.at<uchar>(j, i) = 255;
					}
				}

				cv::imshow("sdl_image_white", sdl_image);  

				key = cv::waitKey(); 

				for (int i = 0; i < sdl_inz_in_cloud.size(); i++)
				{
					sdl_image.at<uchar>(sdl_inz_in_cloud[i].y, sdl_inz_in_cloud[i].x) = 0;
				}



				cv::rectangle(sdl_image, cv::Rect(min_x, min_y, max_x-min_x, max_y-min_y), cv::Scalar(0), 2);

				printf("\ni = %d \n",i);

				char restrain_n[100];
				sprintf(restrain_n,"sdl_inz_cloud_res%d",i);
				cv::imshow(restrain_n, sdl_image);    

				memset(savejpg_name,'\0',sizeof(savejpg_name));
				strcpy(savejpg_name, saveimg_dir);
				strcat(savejpg_name,"/sdl_inz_cloud_restr_");
				sprintf(string,"%d_",i);
				strcat(savejpg_name,string);
				sprintf(string,"%ld",save_count);
				strcat(savejpg_name,string);
				strcat(savejpg_name,".jpg");
				cv::imwrite(savejpg_name,sdl_image);				                
				
				key = cv::waitKey();                 

        		if (i > 0 && (last_val - std_in_x)*(last_val - std_in_x) < restrain_limit * std_in_x) 
            		break;
        		last_val = std_in_x;       
				printf("%d:sdl_inz_in_cloud.size()=%d,std_in_x=%f\n",i,sdl_inz_in_cloud.size(),std_in_x);      
    		}

			for (int i = 0; i < new_width; i++)
			{
				for (int j = 0; j < new_height; j++)
				{
					sdl_image.at<uchar>(j, i) = 255;
				}
			}	

			cv::imshow("sdl_image_white", sdl_image);  

			key = cv::waitKey(); 			

			for (int i = 0; i < sdl_inz_in_cloud.size(); i++)
			{
				sdl_image.at<uchar>(sdl_inz_in_cloud[i].y, sdl_inz_in_cloud[i].x) = 0;
			}

			cv::imshow("sdl_in_image1", sdl_image);
			memset(savejpg_name,'\0',sizeof(savejpg_name));
			strcpy(savejpg_name, saveimg_dir);
			strcat(savejpg_name,"/sdl_in_image1_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			cv::imwrite(savejpg_name,sdl_image);
			printf("\nsdl_inz finish!\n");

			key = cv::waitKey();

			save_count++;

#endif

		//	cv::imshow("Image1", depthim);
	
			cv::imshow("Image", image_ocv);

			key = cv::waitKey(10);
		
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
