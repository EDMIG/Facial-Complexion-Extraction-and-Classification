/**********************************************************************************
*
*					����Ƥ����� FaceSkinDetectionByGMM
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
	* src: ����ĺ�������ԭʼͼ��
	* haarFaceDetectSwitch: haar�������������ƿ���
	*/
	FaceSkinDetectionByGMM(IplImage* src,const int haarFaceDetectSwitch=0);
	~FaceSkinDetectionByGMM();
	bool ProcessFlow();
	CvScalar extractFaceSkinColorFeature();//��ȡ������ɫֵ(L,a,b)
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

	int switchHaarFace;//haar�������������ƿ��أ�0->�أ�1->����Ĭ��Ϊ�أ���������������⣩
	bool isDetectFace;//�Ƿ��⵽����
};

#endif