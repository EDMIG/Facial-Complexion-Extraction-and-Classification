/**********************************************************************************
*
*					脸部皮肤检测 FaceSkinDetectionByGMM
*					   by Hu yangyang 2016/12/2
*
***********************************************************************************/
#ifndef FACESKINDETECTIONBYGMM_H
#define FACESKINDETECTIONBYGMM_H

#include "Global.h"
#include "GMM.h"

class FaceSkinDetectionByGMM
{
public:
	/*
	* src: 输入的含人脸的原始图像
	* haarFaceDetectSwitch: haar级联人脸检测控制开关
	*/
	FaceSkinDetectionByGMM(IplImage* src,const int haarFaceDetectSwitch=0);
	~FaceSkinDetectionByGMM();
	bool ProcessFlow();
	CvScalar extractFaceSkinColorFeature();//提取脸部肤色值(L,a,b)
	IplImage* getFaceSkinImage();
	IplImage* getFaceImage();

private:
	void DynamicScale(const IplImage* inputImage);
	void BuildComplexionGMM();
	void ComputeComplexionProbabilityMap();
	void calculateDarkAndBrightThreshold(const IplImage* mask,const IplImage* grayImage,int& i_dark,int& i_bright,double rate_dark,double rate_bright);
	//int colorhist(float *data,int i_dark,int i_bright);
	void IterativeDetectSkin();

private:
	IplImage* inputImage;
	IplImage* face;
	IplImage* faceImage;
	IplImage* faceImage_Lab;
	IplImage* faceImage_Gauss;
	IplImage* faceSkinImage;
	IplImage* faceSkinMask;

	int nGMM;
	GMM* mComplexionGMM;

	int switchHaarFace;//haar级联人脸检测控制开关：0->关，1->开，默认为关（即不开启级联检测）
	bool isDetectFace;//是否检测到人脸
};

#endif