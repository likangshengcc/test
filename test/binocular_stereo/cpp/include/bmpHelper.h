#ifndef _BMPHELPER_H
#define _BMPHELPER_H

#pragma once
#include "iostream"
#include "fstream"
#include <string>
#include <vector>
#pragma pack(2)

#define   WIDTHBYTES(bits) ((((bits)+31)>>5)<<2)
#define	  PI	3.14159265359


typedef unsigned short      WORD;
typedef unsigned int        DWORD;
typedef int		    		LONG;

typedef struct  {
	unsigned short    bfType;
	unsigned int      bfSize;
	unsigned short    bfReserved1;
	unsigned short    bfReserved2;
	unsigned int      bfOffBits;
} BITMAPFILEHEADER;


typedef struct  {
	DWORD      biSize;
	LONG       biWidth;
	LONG       biHeight;
	WORD       biPlanes;
	WORD       biBitCount;
	DWORD      biCompression;
	DWORD      biSizeImage;
	LONG       biXPelsPerMeter;
	LONG       biYPelsPerMeter;
	DWORD      biClrUsed;
	DWORD      biClrImportant;
} BITMAPINFOHEADER;


struct SingleData
{
	int x;
	int y;
	float r;
};

extern struct SingleData g_circle[20];
extern int 	g_num;

struct SingleData getCenter_test(int bmpHeight, int bmpWidth, int ix, int iy, int ir, unsigned char* bmpdata);
float getMeasurement_AveCoord_test(int bmpHeight, int bmpWidth, int ix, int iy, int ir, unsigned char *bmpdata);
struct SingleData GaussianFunc_test(int ix, int iy, int ir, int n, double v, unsigned char* pBmpBuf, BITMAPINFOHEADER head);

#endif
