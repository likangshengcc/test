
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
#include <iostream>
#include <vector>
#include <cmath>
#include <random>


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


float img[6400][3600], X[6400][3600], Y[6400][3600], Z[6400][3600];
float Z1[6400][3600],Z0[6400][3600];



cv::Mat slMat2cvMat(Mat& input);

#define DEST_PORT 8282
#define DEST_IP_ADDRESS "192.168.0.200"

#define IMG_WIDTH	960
#define IMG_HEIGHT	540

unsigned char imgdata[IMG_WIDTH * IMG_HEIGHT * 3];


struct Point3D {
    double x, y, z;
};

struct Line3D {
    Point3D point;
    Point3D direction;
};

class SimpleLineFitter3D {
	private:
    	std::vector<Point3D> points;
    
	public:
    // 添加点
    void addPoint(double x, double y, double z) {
		points.push_back({x, y, z});
    }
    
    // 拟合直线 - 使用最小二乘法
    Line3D fitLine() {
        Line3D line;
        
        if (points.size() < 2) {
            std::cerr << "must two point" << std::endl;
            return line;
        }
        
        // 计算质心
        double meanX = 0, meanY = 0, meanZ = 0;
        for (const auto& p : points) {
            meanX += p.x;
            meanY += p.y;
            meanZ += p.z;
        }
        meanX /= points.size();
        meanY /= points.size();
        meanZ /= points.size();
        
        line.point = {meanX, meanY, meanZ};
        
        // 构建矩阵A^T A
        double a11 = 0, a12 = 0, a13 = 0;
        double a22 = 0, a23 = 0, a33 = 0;
        
        for (const auto& p : points) {
            double dx = p.x - meanX;
            double dy = p.y - meanY;
            double dz = p.z - meanZ;
            
            a11 += dx * dx;
            a12 += dx * dy;
            a13 += dx * dz;
            a22 += dy * dy;
            a23 += dy * dz;
            a33 += dz * dz;
        }
        
        // 对称部分
        double a21 = a12;
        double a31 = a13;
        double a32 = a23;
        
        // 使用幂迭代法求最大特征值对应的特征向量
        double vx = 1.0, vy = 1.0, vz = 1.0;
        
        for (int iter = 0; iter < 100; ++iter) {
            double newVx = a11 * vx + a12 * vy + a13 * vz;
            double newVy = a21 * vx + a22 * vy + a23 * vz;
            double newVz = a31 * vx + a32 * vy + a33 * vz;
            
            double norm = sqrt(newVx * newVx + newVy * newVy + newVz * newVz);
            if (norm > 0) {
                vx = newVx / norm;
                vy = newVy / norm;
                vz = newVz / norm;
            }
        }
        
        line.direction = {vx, vy, vz};
        return line;
    }
    
    // 计算点到直线的距离
    static double distancePointToLine(const Point3D& point, const Line3D& line) {
        // 向量AP
        double ax = point.x - line.point.x;
        double ay = point.y - line.point.y;
        double az = point.z - line.point.z;
        
        // 方向向量
        double dx = line.direction.x;
        double dy = line.direction.y;
        double dz = line.direction.z;
        
        // 叉积
        double crossX = ay * dz - az * dy;
        double crossY = az * dx - ax * dz;
        double crossZ = ax * dy - ay * dx;
        
        double crossNorm = sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);
        double dirNorm = sqrt(dx * dx + dy * dy + dz * dz);
        
        return crossNorm / dirNorm;
    }
};

class PointLineDistance3D {
public:
    // 方法1：使用叉积计算点到直线的距离
    static double distanceUsingCrossProduct(const Point3D& point, const Line3D& line) {
        // 计算向量AP = point - line.point
        double ap_x = point.x - line.point.x;
        double ap_y = point.y - line.point.y;
        double ap_z = point.z - line.point.z;
        
        // 方向向量
        double v_x = line.direction.x;
        double v_y = line.direction.y;
        double v_z = line.direction.z;
        
        // 计算叉积 AP × v
        double cross_x = ap_y * v_z - ap_z * v_y;
        double cross_y = ap_z * v_x - ap_x * v_z;
        double cross_z = ap_x * v_y - ap_y * v_x;
        
        // 计算叉积的模长
        double cross_magnitude = sqrt(cross_x * cross_x + 
                                      cross_y * cross_y + 
                                      cross_z * cross_z);
        
        // 计算方向向量的模长
        double v_magnitude = sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        
        // 避免除以零
        if (v_magnitude < 1e-10) {
            std::cerr << "警告：方向向量模长为零" << std::endl;
            return -1.0;
        }
        
        return cross_magnitude / v_magnitude;
    }
}


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
				if((sdl_point_nogrand_set[i].sdl_point_value.x < 1.0)&&(sdl_point_nogrand_set[i].sdl_point_value.x > -0.5)&&(sdl_point_nogrand_set[i].sdl_point_value.y < 	
		                     0.2)&&(sdl_point_nogrand_set[i].sdl_point_value.y > -1.5))
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
			cv::imwrite(savejpg_name,image_ocv);

			SimpleLineFitter3D fitter;
			
    		// 添加拟合点
    		for (int i = 0; i < sdl_cloud.size(); i++)
			{
				fitter.addPoint(sdl_cloud[i].sdl_point_value.x, sdl_cloud[i].sdl_point_value.y, sdl_cloud[i].sdl_point_value.z);
			}
    
    		// 拟合直线
    		Line3D line = fitter.fitLine();
    
    		std::cout << "拟合直线：" << std::endl;
   			std::cout << "点: (" << line.point.x << ", " << line.point.y << ", " << line.point.z << ")" << std::endl;
    		std::cout << "方向: (" << line.direction.x << ", " << line.direction.y << ", " << line.direction.z << ")" << std::endl;			

	

		
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
