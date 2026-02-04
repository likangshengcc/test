
#include "bmpHelper.h"
#include <vector>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std;

#define   WIDTHBYTES(bits) ((((bits)+31)>>5)<<2)

struct SingleData g_circle[20];
int 	g_num = 0;

struct SingleData getCenter_test(int bmpHeight, int bmpWidth, int ix, int iy, int ir, unsigned char* bmpdata)
{
	// Probability space shift
	// Calculating the center coordinates of the rectangle

	struct SingleData sd;
	double sum_x = 0.0f;
	double sum_y = 0.0f;
	long len = 0;

	for (int i = (0 > (iy - ir) ? 0 : (iy - ir)); i < (bmpHeight < (iy + ir) ? bmpHeight : (iy + ir)); i++)
	{
		for (int j = (0 >(ix - ir) ? 0 : (ix - ir)); j < (bmpWidth < (ix + ir) ? bmpWidth : (ix + ir)); j++)

		{
			if ((bmpdata[i * bmpWidth * 3 + j * 3] == 0) && (bmpdata[i * bmpWidth * 3 + j * 3 + 1] == 0) && (bmpdata[i * bmpWidth * 3 + j * 3 + 2] == 0))
			{
				sum_x = sum_x + (double)(j - ix);
				sum_y = sum_y + (double)(i - iy);
				len++;
			}
		}
	}
	if (len > 0)
	{
		sd.x = (int)(sum_x / (double)len) + ix;
		sd.y = (int)(sum_y / (double)len) + iy;
	}
	else
	{
		sd.x = ix;
		sd.y = iy;
	}

	sd.r = ir;

	return sd;
}

float getMeasurement_AveCoord_test(int bmpHeight, int bmpWidth, int ix, int iy, int ir, unsigned char *bmpdata) {


	float R = 0;
	float sum_x = 0;
	float sum_y = 0;


	long len = 0;
	for (int i = (0 > (iy - ir) ? 0 : (iy - ir)); i < (bmpHeight < (iy + ir) ? bmpHeight : (iy + ir)); i++)
	{
		for (int j = (0 >(ix - ir) ? 0 : (ix - ir)); j < (bmpWidth < (ix + ir) ? bmpWidth : (ix + ir)); j++)

		{
			if ((bmpdata[i * bmpWidth * 3 + j * 3] == 0) && (bmpdata[i * bmpWidth * 3 + j * 3 + 1] == 0) && (bmpdata[i * bmpWidth * 3 + j * 3 + 2] == 0))
			{
				sum_x = sum_x + fabs((float)j - ix);
				sum_y = sum_y + fabs((float)i - iy);
				len++;
			}
		}
	}
	sum_x = sum_x / (float)len;
	sum_y = sum_y / (float)len;
	R = sqrt(sum_x * sum_x + sum_y * sum_y);

	return R;
}


struct SingleData GaussianFunc_test(int ix, int iy, int ir, int n, double v, unsigned char* pBmpBuf, BITMAPINFOHEADER head )
{

	uint32_t bmpHeight = head.biHeight;
	uint32_t bmpWidth = head.biWidth;
	uint16_t biBitCount = head.biBitCount;

	g_num = 0;


	struct SingleData sd;
	sd.x = ix;
	sd.y = iy;
	sd.r = ir;
	int i = 0;
	for (; i < n; i++)
	{
		sd.x = getCenter_test(bmpHeight, bmpWidth, sd.x, sd.y, sd.r, pBmpBuf).x;
		sd.y = getCenter_test(bmpHeight, bmpWidth, sd.x, sd.y, sd.r, pBmpBuf).y;

		if ((sd.x != ix) || (sd.y != iy))
			g_circle[g_num++] = sd;
		if (g_num > 1)
		{
			if ((abs(g_circle[g_num - 1].x - g_circle[g_num - 2].x) <  v) & (abs(g_circle[g_num - 1].y - g_circle[g_num - 2].y) < v))
				break;
			// The convergence threshold generally is 30
		}

	}


	for (i = 0; i < 1; i++)
	{
		sd.r = getMeasurement_AveCoord_test(bmpHeight, bmpWidth, sd.x, sd.y, sd.r, pBmpBuf)*1.2;
	}

	return sd;

}




