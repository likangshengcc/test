
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




using namespace sl;

float speed_test[20] = {18.646,17.531,16.401,15.221,14.021,12.851,11.611,10.491,9.331,8.141,7.001,5.891,4.741,3.631,2.631,1.531};

float img[6400][3600], X[6400][3600], Y[6400][3600], Z[6400][3600];
float Z1[6400][3600];

std::vector<double>leftdis,frontdis,rightdis,leftoutput,frontoutput,rightoutput,left_output_v,front_output_v,right_output_v,left_output_a,front_output_a,right_output_a;

cv::Mat slMat2cvMat(Mat& input);

#define DEST_PORT 8181
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

// 障碍物
struct ITEMDATA 
{
	cv::Point pt1; 	// 左上角
	cv::Point pt2; 	// 右下角
	float dis;	// 与camera的距离
	float lastdis;
	float spd;
	float zuoyou;	//新添加的距离车的左右位置
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
		if (abs(int(hline[1] - hline[3])) < 3 && abs(int(hline[0] - hline[2])) > 20) {
			udnum++;
			udpro += hline[1] + 310;
			//cout << "find line!" << endl;
		}
	}
	if (udnum)
	{
		up = udpro / udnum;
		down = udpro / udnum + 15;
	}
}

int main(int argc, char **argv) {

// initial udp   
	int sock_fd;
	int send_num;
	int recv_num;
	socklen_t dest_len;
	char send_buf[200] = {"hello "};
	char recv_buf[200];
	struct sockaddr_in addr_serv;

	sock_fd = socket(AF_INET,SOCK_DGRAM,0);
	
	memset(&addr_serv,0,sizeof(addr_serv));
	addr_serv.sin_family = AF_INET;
	addr_serv.sin_addr.s_addr = inet_addr(DEST_IP_ADDRESS);
	addr_serv.sin_port = htons(DEST_PORT);

	dest_len = sizeof(struct sockaddr_in);
	printf("begin send:\n");

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
   	if (argc > 1) 
		init_params.input.setFromSVOFile(argv[1]);
        
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



//所有涉及使用new生成的指针、矩阵等需要占用堆栈的变量均要做销毁处理。（未完全优化）
	int n = 5; 	// GaussianFunc参数
	int v = 30;	// GaussianFunc参数
	int f = 0;	// GaussianFunc参数
	
	std::vector<ItemSingleData> vecnoit; // 没有迭代过程的最终障碍物数组
	 
	
	int fnameindex = 0; // 保存点云文件名的索引增量
    char key = ' ';
	std::string fnamestr = "../depth/depth%d.bmp"; // 转存位图文件名
	int maxitemnum = 3; // 最多障碍物数量，默认1，最大3
	int rectsize = 5; // GaussianFunc搜索框的size，数越大框越小，默认2框最大，递增1则越来越小
	bool bsavepointcloud = false;//保存点云数据的开关
	bool bdrawmerged = true;// 绘制合并后的障碍物的开关
	bool breplay = false;// 显示迭代过程的开关

	bool stop_flag = false;//stop_line on/off

	bool first_frame = true; // the first frame

	int speedt_count = 0;
	SDLDATA itdata; // 用于迭代过程的数据
	float mindislast[4];
	printf("按键说明\n\
		a和b是改变GaussianFunc的矩形尺寸\n\
		c是设置障碍物数量（1或3个）\n\
		d是呈现迭代过程的开关\n\
		e是开启保存点云数据的开关, 是持续保存，不是单帧，别忘了关掉，否则很卡\n\
		f是重叠合并障碍物的开关\n\
		q是退出");
	char msg[128]  = {};
	char msg1[128] = {};
	char msg2[128] = {};
	char msg3[128] = {};

	const char save_dir[256] = "../video/video_";
	long int save_count = 0;

   
	while (key != 'q') 
	{
		gettimeofday(&start1,NULL);	
		//gettimeofday(&start,NULL);
		switch(key){
			case 'a':
				//printf("\ninput a\n");
				rectsize++;
				if (rectsize > 8)
					rectsize = 8;
				break;
			case 'b':
				//printf("\ninput b\n");
				rectsize--;
				if (rectsize < 2)
					rectsize = 2;
				break;
			case 'c':			
				//printf("\ninput c\n");
				if (maxitemnum == 3)
					maxitemnum = 1;
				else
					maxitemnum = 3;
				break;
			case 'd':
				//printf("\ninput d\n");
				breplay = !breplay;
				itdata.reset();
				break;
			case 'e':
				//printf("\ninput e\n");
				bsavepointcloud = !bsavepointcloud;
				break;
			case 'f':
				//printf("\ninput f\n");
				bdrawmerged = !bdrawmerged;
				break;
			case 'g':
				//printf("\ninput f\n");
				stop_flag = !stop_flag;
				break;
		
			default:
				break;
		}
		
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
				printf("\nrun_timeuse = %fms",run_timeuse);
				
			}
			gettimeofday(&start,NULL);

			cv::Mat matreplay;
			cv::Mat depthim;

			cv::imshow("depth_image_zed", depth_image_ocv);

			cv::cvtColor(depth_image_ocv, depthim, CV_RGBA2GRAY);
			
			cv::cvtColor(image_ocv, image_ocv_copy, 1);

			int x, y, z;

			sl::float4 point_cloud_value;

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

						//printf("\ni=%d,j=%d,x=%f,y=%f,z=%f,d=%f\n",i,j,X[i][j],Y[i][j],point_cloud_value.z,Z1[i][j]);
						
						if(X[i][j] < 0.8 && X[i][j] > -0.8)
						{
							if (Y[i][j] < 0.01&&Y[i][j]>-1)//去除地面，地面在视场的远方会出现在较高的位置造成干扰，只保留地面20cm以上的信息。
							{
								img[i][j] = sqrt(point_cloud_value.x * point_cloud_value.x + point_cloud_value.y * point_cloud_value.y + point_cloud_value.z * point_cloud_value.z);//tongdao 
							}
							else
							{
								depthim.at<uchar>(j, i) = 255;
								Z1[i][j] = 40;
								img[i][j] = 40;
							}
						}
						else
						{
							if (Y[i][j] < 0.01&&Y[i][j]>-1)//去除地面，地面在视场的远方会出现在较高的位置造成干扰，只保留地面20cm以上的信息。
							{
								img[i][j] = sqrt(point_cloud_value.x * point_cloud_value.x + point_cloud_value.y * point_cloud_value.y + point_cloud_value.z * point_cloud_value.z);//tongdao 
							}
							else
							{
								depthim.at<uchar>(j, i) = 255;
								Z1[i][j] = 40;
								img[i][j] = 40;
							}					
						}
					

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
					}					

				}
			}
			
			cv::imshow("Image1", depthim);
			cv::cvtColor(depthim, matreplay, CV_GRAY2RGB);

			int w;
			int w_start;
			int w_end;
			int h;
			int h_start;
			int h_end;
			int r;
			SingleData re1[3];
			int row = matreplay.rows;
			int col = matreplay.cols;
			for (int ind = 0; ind < 3; ++ind)
			{
				BITMAPINFOHEADER head;
				switch (ind) {
					case 0:
						w = col;
						h = row;
						w_start = 0;
						w_end	= col;
						h_start = 0;
						h_end	= row;
						head.biWidth = w;
						head.biHeight = h;
						head.biBitCount = 24;
						break;
					case 1:
						w = col/2;
						h = row;
						w_start = 7 * col / 16;
						w_end = 7 * col / 16 + col / 2;
						h_start = 0;
						h_end = row;
						head.biWidth = w;
						head.biHeight = h;
						head.biBitCount = 24;
						break;
					case 2:
						w = col / 2;
						h = row;
						w_start = col / 16;
						w_end = col / 16 + col / 2;
						h_start = 0;
						h_end = row;
						head.biWidth = w;
						head.biHeight = h;
						head.biBitCount = 24;
						break;
					default:
						break;
				}
	

				for (int nrow = h_start,i = 0; nrow < h_end; nrow++,i++)
				{
					uchar* data = matreplay.ptr<uchar>(nrow);

					for (int ncol = w_start * 3,j = 0; ncol < w_end * 3; )
					{
						imgdata[i*w * 3 + j] = data[ncol];
						imgdata[i*w * 3 + j + 1] = data[ncol + 1];
						imgdata[i*w * 3 + j + 2] = data[ncol + 2];
						ncol += 3;
						j += 3;
					}
				}


				r = w < h ? w / rectsize : h / rectsize;

				auto re = GaussianFunc_test(w / 2, h / 2, r, n, v / 10.0, imgdata, head);

				if (g_num)
				{
					// 准备用于迭代的数据
					if (breplay)
					{
						std::vector<ItemSingleData> que;
						for (int m=0; m<g_num-1; ++m)
						{
							ItemSingleData t;
							t.x = g_circle[m].x + w_start; // 加上对于分块的xy的偏移量
							t.y = g_circle[m].y + h_start;
							t.r = g_circle[m].r;
							t.n = ind;
							que.push_back(t);
						}
						itdata.vecItem.push_back(que);
					}
				}
					// 最终结果的数据
					
					// 加上对于分块的xy的偏移量
					ItemSingleData t2;
					t2.x = re.x + w_start;
					t2.y = re.y + h_start;
					t2.r = re.r;
					t2.n = ind;
					vecnoit.push_back(t2);
			}
			std::vector<ITEMDATA> vecitem; // 障碍物数组
			// 取点云用于计算距离
			// 目前的障碍探测及排序是直接按最小距离排序，
			//gettimeofday(&end,NULL);

			//printf("\nvecnoit.size = %d",vecnoit.size());
			for (auto it = vecnoit.begin(); it != vecnoit.end(); ++it)
			{
				// 根据障碍物的位置，获取点云数据位于的范围
				int startx, endx, starty, endy;
				SingleData2xy(*it, new_image_size, startx, starty, endx, endy);
				// 在此范围里计算障碍物范围内对应的所有点与摄像机的距离
				std::vector<float> vecarr;
				std::vector<float> vecarrn;
				for (int m = starty; m < endy; ++m)
				{
					for (int n = startx; n < endx; ++n)
					{
						if (isnormal(X[n][m])&&(img[n][m] < 39))
						{
							auto dis = img[n][m];
							auto zuoyou = X[n][m];
							vecarr.push_back(dis);
							vecarrn.push_back(abs(zuoyou));
						}
					}
				}
				
				float propotion;
				long int sum = (endx - startx) * (endy - starty);
				propotion =(float) vecarr.size() / (float)sum;
				
				if (propotion > 0.1 )
				{
					// 取距离最近的那个点作为障碍物的距离
					float mindis = *std::min_element(vecarr.begin(), vecarr.end());
					float minzuoyou = *std::min_element(vecarrn.begin(), vecarrn.end());

				//	run_timeuse = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;

				//	run_timeuse /= 1000;
					float speed = 0;
					if(!first_frame)					
						speed = 1000*(mindis - mindislast[it->n]) / run_timeuse;

					mindislast[it->n] = mindis;
					ITEMDATA item;


					item.dis = mindis;
					item.zuoyou = minzuoyou;
					item.spd = speed;
					item.pt1 = cv::Point(startx, starty);
					item.pt2 = cv::Point(endx, endy);
					item.n = it->n;

					vecitem.push_back(item); // 添加一个障碍物
				}
			} 


			int start = 0;
			int mid_x[3] ;	
			// 绘制合并的障碍物
			std::vector<ITEMDATA> mergearr; 
			printf("\n\n\nvecitem.size = %d",vecitem.size());
			if (bdrawmerged && maxitemnum == 3 && vecitem.size() > 0)
			{
				// 合并重叠的障碍物
				mergearr = MergeItem(vecitem);

				start = 0;
				for (auto it = mergearr.begin(); it!= mergearr.end();++it, ++start)
				{
					mid_x[start] = (it->pt2.x + it->pt1.x) / 2;
				//	printf("\nmid_x[%d] = %d",start,mid_x[start]);
				}
			
				int temp;
				if(start == 3)
				{
					if(mid_x[0] >= mid_x[1])
					{
						if(mid_x[0] <= mid_x[2])
						{
							mergearr[0].n = 1;
							mergearr[1].n = 0;
							mergearr[2].n = 2;							
						}
						else
						{
							mergearr[0].n = 2;
							if(mid_x[1] <= mid_x[2])
							{
								mergearr[1].n = 0;
								mergearr[2].n = 1;	
							}
							else
							{
								mergearr[1].n = 1;
								mergearr[2].n = 0;	
							}	
						}
			
					}
					else
					{
						if(mid_x[1] <= mid_x[2])
						{
							mergearr[0].n = 0;
							mergearr[1].n = 1;
							mergearr[2].n = 2;	
						}	
						else
						{
							mergearr[1].n = 2;
							if(mid_x[0] >= mid_x[2])
							{
								mergearr[0].n = 1;
								mergearr[2].n = 0;	
							}
							else
							{
								mergearr[0].n = 0;
								mergearr[2].n = 1;
							}

						}				
					}
				}
				else if(start == 2)
				{
					if(mid_x[0] <= 320 )
						mergearr[0].n = 0;
					else if(mid_x[0] >= 640)
						mergearr[0].n = 2;
					else
						mergearr[0].n = 1;

					if(mid_x[0] >= mid_x[1])
					{
						switch(mergearr[0].n){
							case 0:
									mergearr[0].n = 1;
									mergearr[1].n = 0;
									break;
							case 1:
									mergearr[1].n = 0;
									break;
							case 2:
									if(mid_x[1] <= 320 )
										mergearr[1].n = 0;
									else if(mid_x[1] >= 640)
										mergearr[1].n = 1;
									else
										mergearr[1].n = 1;
									break;
							default:
									break;
						}
					}			
					else
					{
						switch(mergearr[0].n){
							case 0:
									if(mid_x[1] <= 320 )
										mergearr[1].n = 1;
									else if(mid_x[1] >= 640)
										mergearr[1].n = 2;
									else
										mergearr[1].n = 1;
									break;
							case 1:
									mergearr[1].n = 2;
									break;
							case 2:
									mergearr[0].n = 1;
									mergearr[1].n = 2;
									break;
							default:
									break;
						}
					}			
				}	
				else if(start == 1)
				{
					if(mid_x[0] <= 320 )
						mergearr[0].n = 0;
					else if(mid_x[0] >= 640)
						mergearr[0].n = 2;
					else
						mergearr[0].n = 1;
				}		

				int mergearr_num = mergearr.size();
				if(mergearr_num < 3)
				{
					switch(mergearr_num){
						case 2:
								if(mergearr[1].n!=0&&mergearr[0].n!=0)
								{
									leftdis.clear();
									leftoutput.clear();
									left_output_v.clear();
									left_output_a.clear();
								}									
								if(mergearr[1].n!=1&&mergearr[0].n!=1)
								{
									frontdis.clear();
									frontoutput.clear();
									front_output_v.clear();
									front_output_a.clear();
								}
								if(mergearr[1].n!=2&&mergearr[0].n!=2)
								{
									rightdis.clear();
									rightoutput.clear();
									right_output_v.clear();
									right_output_a.clear();
								}									
								break;
						case 1:	
								if(mergearr[0].n!=0)
								{
									leftdis.clear();
									leftoutput.clear();
									left_output_v.clear();
									left_output_a.clear();
								}
								if(mergearr[0].n!=1)
								{
									frontdis.clear();
									frontoutput.clear();
									front_output_v.clear();
									front_output_a.clear();
								}
								if(mergearr[0].n!=2)
								{
									rightdis.clear();
									rightoutput.clear();
									right_output_v.clear();
									right_output_a.clear();
								}
								break;
					}
				}
				
				for (auto it = mergearr.begin(); it != mergearr.end(); ++it)
				{
					ScaleRect(it->pt1, it->pt2, -0.025f);
					rectangle(matreplay, cv::Point(it->pt1.x - 15, it->pt1.y - 25), it->pt2, cv::Scalar(0, 255, 0), 3);
					rectangle(image_ocv, it->pt1, it->pt2, cv::Scalar(0, 255, 0), 3);
					rectangle(depth_image_ocv, it->pt1, it->pt2, cv::Scalar(0, 0, 255), 3);
					rectangle(depthim, it->pt1, it->pt2, cv::Scalar(0, 0, 255), 3);

					switch(it->n){
						case 0:
								strcpy(it->dirc,"left");
								break;
						case 1:
								strcpy(it->dirc,"front");
								break;
						case 2:
								strcpy(it->dirc,"right");
								break;
						default:
								break;
					}

					if(!first_frame)
					{
						int k ;
						switch(it->n){
							case 0:
									//leftdis.push_back(it->dis);
									if(speedt_count > 18)
										speedt_count = 0;
									leftdis.push_back(speed_test[speedt_count++]);
									printf("\nBefore erase left first element: \n"); 
									for(int i = 0; i < leftdis.size(); i++)
										printf("%f,",leftdis[i]);
									printf("\n");

									if(leftdis.size() >= 3)
									{
										//predict_second(leftdis, leftoutput, left_output_v, left_output_a, 0.8, 3, 3, run_timeuse);
										predict_second(leftdis, leftoutput, left_output_v, left_output_a, 0.8, 3, 3, 110);
										it->spd = left_output_v[0];
										printf("\nleft speed it->spd : %f\n",it->spd);
										leftdis.erase(leftdis.begin());
									}
									else
									{
										it->spd = 0;
									}

									if(left_output_v.size() > 0)
									{
										k = 0;
										for(auto itt = left_output_v.begin(); k < 3 || itt!= left_output_v.end() ; ++k)
											left_output_v.erase(itt);
									}
									if(left_output_a.size() > 0)
									{
										k = 0;
										for(auto itt = left_output_a.begin(); k < 3 || itt!= left_output_a.end(); ++k)
											left_output_a.erase(itt);
									}
									break;
							case 1:
									frontdis.push_back(it->dis);
									printf("\nBefore erase front first element: \n"); 
									for(int i = 0; i < frontdis.size(); i++)
										printf("%f,",frontdis[i]);
									printf("\n");
									if(frontdis.size() >= 3)
									{
										predict_second(frontdis, frontoutput, front_output_v, front_output_a, 0.8, 3, 3, run_timeuse);
										it->spd = front_output_v[0];
										printf("\nfront speed it->spd : %f\n",it->spd);
										frontdis.erase(frontdis.begin());
									}
									else
									{
										it->spd = 0;
									}

									if(front_output_v.size() > 0)
									{
										k = 0;
										for(auto itt = front_output_v.begin(); k < 3 || itt!= front_output_v.end() ; ++k)
											front_output_v.erase(itt);
									}
									if(front_output_a.size() > 0)
									{
										k = 0;
										for(auto itt = front_output_a.begin(); k < 3 || itt!= front_output_a.end(); ++k)
											front_output_a.erase(itt);
									}

									break;
							case 2:
									rightdis.push_back(it->dis);
									printf("\nBefore erase right first element: \n"); 
									for(int i = 0; i < rightdis.size(); i++)
										printf("%f,",rightdis[i]);
									printf("\n");
									if(rightdis.size() >= 3)
									{
										predict_second(rightdis, rightoutput, right_output_v, right_output_a, 0.8, 3, 3, run_timeuse);
										it->spd = right_output_v[0];
										printf("\nright speed it->spd : %f\n",it->spd);
										rightdis.erase(rightdis.begin());
									}
									else
									{
										it->spd = 0;
									}
	
									if(right_output_v.size() > 0)
									{
										k = 0;
										for(auto itt = right_output_v.begin(); k < 3 || itt!= right_output_v.end() ; ++k)
											right_output_v.erase(itt);
									}
									if(right_output_a.size() > 0)
									{
										k = 0;
										for(auto itt = right_output_a.begin(); k < 3 || itt!= right_output_a.end(); ++k)
											right_output_a.erase(itt);
									}
								
									break;
							default:
									break;
						}	
					}
					
					first_frame = false;
				
					sprintf(msg, "%d: %s:dis:%.3fm", it->n,it->dirc,it->dis);
					sprintf(msg1, ",offset:%.3fm", it->zuoyou);
					sprintf(msg2, ",speed:%.3fKm/h", it->spd);
					sprintf(msg3, "run_timeuse: %fms", run_timeuse);
					//cv::putText(matreplay, msg, cv::Point(it->pt1.x + 5, it->pt1.y + 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));

					//cv::putText(image_ocv, msg, cv::Point(it->pt1.x + 5, it->pt1.y + 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
					cv::putText(image_ocv, msg, cv::Point(it->pt1.x , it->pt1.y - 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
					cv::putText(image_ocv, msg1, cv::Point(it->pt1.x, it->pt1.y - 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
					cv::putText(image_ocv, msg2, cv::Point(it->pt1.x, it->pt1.y - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
					cv::putText(image_ocv, msg3, cv::Point(400 , 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
				}
			}
			else
			{
				leftdis.clear();
				leftoutput.clear();
				left_output_v.clear();
				left_output_a.clear();
				frontdis.clear();
				frontoutput.clear();
				front_output_v.clear();
				front_output_a.clear();
				rightdis.clear();
				rightoutput.clear();
				right_output_v.clear();
				right_output_a.clear();
			}
			 
			itdata.reset();
           	



			float stop_dis = 0.0;
			if(stop_flag)
			{
				cv::Mat image_copy;
				int up, down;
				cv::Mat src, src1;
				
				//if (capture.read(image_copy))
				//{
				//	resize(image_copy, image_copy, cv::Size(600, 400));
					//cv::imshow("1", image_copy);
					cv::cvtColor(image_ocv_copy, image_copy, 1);
					cv::Mat gray;
					cvtColor(image_copy, gray, cv::COLOR_BGR2GRAY);

					//cv::imshow("11", gray);

					cv::cvtColor(image_copy, src1, 1);
					cv::rectangle(src1, cv::Rect(410, 300, 150, 145), cv::Scalar(255, 0, 0), 2);
					//cv::imshow("2", src1);
					src = src1(cv::Rect(411, 310, 148, 130));

					cvtColor(src, gray, cv::COLOR_BGR2GRAY);

					//cv::imshow("22", gray);

					cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0);
					cv::Mat threshold_output(gray.rows, gray.cols, CV_8UC1);
					cv::threshold(gray, threshold_output, 140, 255, cv::THRESH_BINARY);

					//cv::imshow("3", threshold_output);

					stopline(threshold_output, up, down);

					printf("\nup=%d,down=%d",up,down);
					int js = 1;
					float sum2 = 0;

					if(up > 260)
					{
						cv::rectangle(src1, cv::Rect(410, up, 150, 4), cv::Scalar(255, 255, 0), 2);
						cv::rectangle(image_ocv, cv::Rect(410, up, 150, 4), cv::Scalar(255, 255, 0), 2);
						for(int i = (480 - 40); i < (480 + 40); i++)
						{
							for(int j = up; j <= down; j++)
							{
								if(Z1[i][j] < 40 && Z1[i][j] > 0)
								{
									sum2 += Z1[i][j];
									js++;
								}
							}
						}
					}
					printf("\njs = %d",js);
					stop_dis = sum2 / js;
					if(stop_dis > 7 || stop_dis < 0.2)
						stop_dis = 0;

					sprintf(msg, "stop_dis:%.2fm", stop_dis);
					//路测拍视频的程序添加了障碍距离车辆的横向距离

					cv::putText(src1, msg, cv::Point(410 , 460), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0));
					cv::rectangle(image_ocv, cv::Rect(410, 300, 150, 145), cv::Scalar(255, 0, 0), 2);
					cv::putText(image_ocv, msg, cv::Point(410 , 460), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0));
					cv::imshow("33", src1);					
					//waitKey();
				//}
			}

			cv::imshow("Image", image_ocv);

			char savejpg_name[100];
			char string[50];
			strcpy(savejpg_name, save_dir);
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg");
			//cv::imwrite(savejpg_name,image_ocv);
			save_count++;

			key = cv::waitKey(10);

			if(bsavepointcloud)
			{
				extern int errno;
				char savedir_name[100];
				char save_name[100];
				char string[50];
				strcpy(savedir_name, "../save/point_save");
				sprintf(string,"%ld",save_count);
				strcat(savedir_name,string);
				
				char *path = savedir_name;
				if(mkdir(path,0766)==0)
				{
					//printf("\ncreated the directory %s.\n",path);
				}
				else
				{
					printf("can't create the directory %s.\n",path);
					printf("errno:%d\n",errno);
					printf("ERR:%s\n",strerror(errno));
				}
				
				strcpy(save_name, savedir_name);
				strcat(save_name,"/");
				strcat(save_name,string);
				strcat(save_name,".jpg");			
				cv::imwrite(save_name,image_ocv);

				strcpy(save_name, savedir_name);
				strcat(save_name,"/depth_image_ocv");
				strcat(save_name,string);
				strcat(save_name,".jpg");			
				cv::imwrite(save_name,depth_image_ocv);

				strcpy(save_name, savedir_name);
				strcat(save_name,"/depthim");
				strcat(save_name,string);
				strcat(save_name,".jpg");			
				cv::imwrite(save_name,depthim);
				save_count++;
					
				int fd = -1;
				sl::Mat point_cloud1;
				auto err = zed.retrieveMeasure(point_cloud1, MEASURE::XYZRGBA, MEM::CPU, new_image_size);
				/*char fname[256] = {};
				strcpy(fname, savedir_name);
				strcat(fname,"/point_cloud");
				sprintf(string,"%d",fnameindex);
				strcat(fname,string);
				strcat(fname,".pcd");
				
				auto state = point_cloud1.write(fname);
				fnameindex++;*/
				int start = 0;
				sl::float4 point_cloud_value1;
				//int start = 0;
				auto it = vecitem.begin();
				auto it_end = vecitem.end();
				if (bdrawmerged)
				{
					it = mergearr.begin();
					it_end = mergearr.end();
				}
				for (; it!= it_end && start < maxitemnum;++it, ++start)
				{
					strcpy(save_name, savedir_name);
					if (!bdrawmerged)
					{
						strcat(save_name,"/vecitem_");
					}
					else
					{
						strcat(save_name,"/mergearr_");
					}
					sprintf(string,"%d",start);
					strcat(save_name,string);
					strcat(save_name,".csv");

					fd = open(save_name,O_WRONLY | O_CREAT,0777);
					char str[200];
					int length,res;
					memset(str,'\0',sizeof(str));
					strcpy(str,"x,y,point_cloud.x,point_cloud.y,point_cloud.z,dis,offset\n");
					length = strlen(str);
					if((res = write(fd,str,length)) != length)
					{
						printf("\nError writing to the file.\n");
						exit(1);
					}

					for(int i = it->pt1.x; i < it->pt2.x; i++)
					{
						for(int j = it->pt1.y; j < it->pt2.y; j++)
						{
							point_cloud1.getValue(i, j, &point_cloud_value1);

							memset(str,'\0',sizeof(str));
							sprintf(string,"%d,",i);
							strcat(str,string);
							sprintf(string,"%d,",j);
							strcat(str,string);	
							sprintf(string,"%f,",point_cloud_value1.x);
							strcat(str,string);	
							sprintf(string,"%f,",point_cloud_value1.y);
							strcat(str,string);
							sprintf(string,"%f,",point_cloud_value1.z);
							strcat(str,string);		
							sprintf(string,"%f,",img[i][j]);
							strcat(str,string);	
							sprintf(string,"%f",X[i][j]);
							strcat(str,string);
							strcat(str,"\n");
							length = strlen(str);
							if((res = write(fd,str,length)) != length)
							{
								printf("\nError writing to the file.\n");
								exit(1);
							}	
											
							//printf("\npoint(%d,%d):x=%f, y=%f, z=%f\n",i,j,point_cloud_value1.x,point_cloud_value1.y,point_cloud_value1.z);
						}
					}
					close(fd);
						//printf("\n---------------------------------------------------------------\n");
						//cv::waitKey();
				}

				bsavepointcloud = false;
			}

			

			gettimeofday(&end1,NULL);
			run_timeuse1 = 1000000 * (end1.tv_sec - start1.tv_sec) + end1.tv_usec - start1.tv_usec;
			run_timeuse1 /= 1000;
			printf("\nrun_timeuse1 = %fms",run_timeuse1);
		

			memset(send_buf,'\0',sizeof(send_buf));

			strcpy(savejpg_name, "video_");
			sprintf(string,"%ld",save_count);
			strcat(savejpg_name,string);
			strcat(savejpg_name,".jpg\n");
			strcpy(send_buf,savejpg_name);


			SortItem(mergearr);
			start = 0;
			for (auto it = mergearr.begin(); it!= mergearr.end() && start < maxitemnum;++it, ++start)
			{
				sprintf(msg, "%d: %s:dis:%.3fm", it->n,it->dirc,it->dis);
				sprintf(msg1, ",offset:%.3fm", it->zuoyou);
				sprintf(msg2, ",speed:%.6fm/s\n", it->spd);	
	
				strcat(send_buf,msg);
				strcat(send_buf,msg1);
				strcat(send_buf,msg2);
			}
				
			if(stop_flag)
			{
				sprintf(msg, "stop_dis:%.3fm\n", stop_dis);
				strcat(send_buf,msg);
			}
			send_num = sendto(sock_fd,send_buf,sizeof(send_buf),0,(struct sockaddr*)&addr_serv,dest_len);	

			if(send_num < 0){
				//perror("sendto");
				//exit(1);
			}else{
			//	printf("send successful:%s\n",send_buf);
			}
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
