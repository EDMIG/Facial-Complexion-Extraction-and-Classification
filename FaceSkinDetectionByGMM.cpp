/**********************************************************************************
*
*					����Ƥ����� FaceSkinDetectionByGMM
*					   by Hu yangyang 2016/12/2
*
***********************************************************************************/
#include "FaceSkinDetectionByGMM.h"
#include "RemoveNoise.h"
#include "FaceHaarDetect.h"

/*
* src: ����ĺ�������ԭʼͼ��
* haarFaceDetectSwitch: haar�������������ƿ���
*/
FaceSkinDetectionByGMM::FaceSkinDetectionByGMM(IplImage* src,const int haarFaceDetectSwitch/*=0*/)
{
	inputImage = cvCloneImage(src);

	switchHaarFace = haarFaceDetectSwitch;//haar�������������ƿ���
	isDetectFace = true;//��ʼΪtrue

	nGMM = 3;

	face = NULL;
	faceImage = NULL;
	faceImage_Lab = NULL;
	faceImage_Gauss = NULL;
	faceSkinImage = NULL;
	faceSkinMask = NULL;

	mComplexionGMM = NULL;
}
FaceSkinDetectionByGMM::~FaceSkinDetectionByGMM()
{
	if(inputImage != NULL) cvReleaseImage(&inputImage);
	if(face != NULL) cvReleaseImage(&face);
	if(faceImage != NULL) cvReleaseImage(&faceImage);
	if(faceImage_Lab != NULL) cvReleaseImage(&faceImage_Lab);
	if(faceImage_Gauss != NULL) cvReleaseImage(&faceImage_Gauss);
	if(faceSkinImage != NULL) cvReleaseImage(&faceSkinImage);
	if(faceSkinMask != NULL) cvReleaseImage(&faceSkinMask);

	if(mComplexionGMM != NULL) delete mComplexionGMM;
}

//����ͼ��ߴ綯̬����
void FaceSkinDetectionByGMM::DynamicScale(const IplImage* inputImage)
{
	int size;
	size = (inputImage->height > inputImage->width) ? inputImage->height : inputImage->width;
	double type = ((double)size)/300.0;

	if (type<=1)
	{
		faceImage = cvCloneImage(inputImage);
	}
	else
	{
		faceImage = cvCreateImage(cvSize(cvRound(inputImage->width/type),cvRound(inputImage->height/type)),inputImage->depth,inputImage->nChannels);
		cvResize(inputImage,faceImage);
	}
}

bool FaceSkinDetectionByGMM::ProcessFlow()
{
	CvRect faceROIRect;
	//haar�����������
	if (switchHaarFace == 1)//open
	{
		FaceDetection faceDetect = FaceDetection(inputImage);
		isDetectFace = faceDetect.detect();
		if (!isDetectFace)//δ��⵽����
		{
			return isDetectFace;
		}
		else
		{
			faceROIRect = faceDetect.getFaceRect();
			faceROIRect.y = faceROIRect.y - faceROIRect.height/10;
			faceROIRect.height = faceROIRect.height*6/5;
			if (faceROIRect.y<0) faceROIRect.y = 0;
			if(faceROIRect.height>inputImage->height) faceROIRect.height = inputImage->height;
		}
	}
	else//close
	{
		faceROIRect.x = 0;
		faceROIRect.y = 0;
		faceROIRect.width = inputImage->width;
		faceROIRect.height = inputImage->height;
	}

	cvSetImageROI(inputImage,faceROIRect);
	face = cvCreateImage(cvGetSize(inputImage),inputImage->depth,inputImage->nChannels);
	cvCopy(inputImage,face);
	cvResetImageROI(inputImage);
	//cvShowImage("face",face);

	DynamicScale(face);

	BuildComplexionGMM();
	ComputeComplexionProbabilityMap();
	IterativeDetectSkin();

	return isDetectFace;
}

void FaceSkinDetectionByGMM::BuildComplexionGMM()
{
	//�����������Լӿ�GMMѵ���ٶ�
	int len;
	len = (faceImage->height > faceImage->width) ? faceImage->height : faceImage->width;
	double type = ((double)len)/50.0;
	IplImage* faceImageScale = cvCreateImage(cvSize(cvRound(faceImage->width/type),cvRound(faceImage->height/type)),faceImage->depth,faceImage->nChannels);
	cvResize(faceImage,faceImageScale);
	//cvShowImage("faceImageScale",faceImageScale);

	//��ûҶ�ͼ��
	IplImage* faceImageScaleGray = cvCreateImage(cvGetSize(faceImageScale),8,1);
	cvCvtColor(faceImageScale,faceImageScaleGray, CV_BGR2GRAY);
	//cvShowImage("gray",faceImageScaleGray);

	//��ȡ��Բ��
	IplImage* faceScaleEllipseMask = cvCreateImage(cvGetSize(faceImageScale),8,1);
	cvZero(faceScaleEllipseMask);
	CvPoint center = cvPoint(cvRound(faceScaleEllipseMask->width*0.5),cvRound(faceScaleEllipseMask->height*0.5));
	CvSize size = cvSize(cvRound(faceScaleEllipseMask->width*0.36),cvRound(faceScaleEllipseMask->height*0.50));
	cvEllipse(faceScaleEllipseMask,center,size,0,0,360,cvScalar(255),CV_FILLED);//������ȡ��Բ�ε�����
	//cvShowImage("ellipse",faceScaleEllipseMask);

	//cvEllipse(faceImageScale,center,size,0,0,360,cvScalar(255));
	//cvShowImage("faceellipse",faceImageScale);

	//ȥ��ͷ����üë���۾��ȷǷ�ɫ����
	int i_dark,i_bright;
	double rate_dark = 0.2, rate_bright = 0;
	calculateDarkAndBrightThreshold(faceScaleEllipseMask,faceImageScaleGray,i_dark,i_bright,rate_dark,rate_bright);
	for (int y=0;y<faceScaleEllipseMask->height;++y)
	{
		for (int x=0;x<faceScaleEllipseMask->width;++x)
		{
			if (cvGetReal2D(faceImageScaleGray,y,x)<i_dark)
			{
				cvSetReal2D(faceScaleEllipseMask,y,x,0);
			}
		}
	}
	//cvShowImage("ellipseFine",faceScaleEllipseMask);

	//ȡLab��ɫ�ռ�
	IplImage* faceImageScale_Lab = NULL;
	faceImageScale_Lab = cvCreateImage(cvGetSize(faceImageScale),8,3);
	cvCvtColor(faceImageScale,faceImageScale_Lab,CV_BGR2Lab); //������ɫ�ռ�
	
	//׼��ѵ������
	uint cnt = 0,nrows = 0;
	for (int y=0;y<faceScaleEllipseMask->height;y++)
	{
		for (int x=0;x<faceScaleEllipseMask->width;x++)
		{
			if (cvGetReal2D(faceScaleEllipseMask,y,x)>200)
			{
				nrows++;
			}
		}
	}

	double** data ;
	data = (double**)malloc(nrows*sizeof(double*));
	for (int i = 0; i < nrows; i++) data[i] = (double*)malloc(3*sizeof(double));//����Ϊ��ά��˹ģ��

	//	copy the data from the color array to a temp array 
	//	and assin each sample a random cluster id
	for (int y=0;y<faceScaleEllipseMask->height;y++)
	{
		for (int x=0;x<faceScaleEllipseMask->width;x++)
		{
			if (cvGetReal2D(faceScaleEllipseMask,y,x)>200)
			{
				data[cnt][0] = cvGet2D(faceImageScale_Lab,y,x).val[0];//������ɫ�ռ�����������ڸ�˹��ģ
				data[cnt][1] = cvGet2D(faceImageScale_Lab,y,x).val[1];
				data[cnt++][2] = cvGet2D(faceImageScale_Lab,y,x).val[2];
			}
		}
	}
	mComplexionGMM = new GMM(nGMM);
	mComplexionGMM->Build(data,nrows);

	for (int i = 0; i < nrows; i++) free(data[i]);
	free(data);

	//release
	cvReleaseImage(&faceImageScale);
	cvReleaseImage(&faceImageScaleGray);
	cvReleaseImage(&faceScaleEllipseMask);
	cvReleaseImage(&faceImageScale_Lab);
}

void FaceSkinDetectionByGMM::ComputeComplexionProbabilityMap()
{
	//ȡLab��ɫ�ռ�
	faceImage_Lab = cvCreateImage(cvGetSize(faceImage),8,3);
	cvCvtColor(faceImage,faceImage_Lab,CV_BGR2Lab); //������ɫ�ռ�

	//�����ɫ����
	faceImage_Gauss = cvCreateImage(cvGetSize(faceImage),IPL_DEPTH_64F,1);
	cvZero(faceImage_Gauss);
	for (int y=0;y<faceImage_Gauss->height;y++)
	{
		for (int x=0;x<faceImage_Gauss->width;x++)
		{
			CvScalar pixel = cvGet2D(faceImage_Lab,y,x);
			Color c(pixel.val[0],pixel.val[1],pixel.val[2]);//��ά��˹ģ�͵Ĳ�������
			float px =  mComplexionGMM->p(c);
			cvSetReal2D(faceImage_Gauss,y,x,px);
		}
	}
	cvNormalize(faceImage_Gauss,faceImage_Gauss,1.0,0.0,CV_C);

	//cvShowImage("ComplexionProbabilityOriginal",faceImage_Gauss);

	cvSmooth(faceImage_Gauss,faceImage_Gauss,CV_GAUSSIAN,5,5);//�Ľ�

	//cvShowImage("ComplexionProbabilityFine",faceImage_Gauss);
}

void FaceSkinDetectionByGMM::IterativeDetectSkin()
{
	faceSkinMask = cvCreateImage(cvGetSize(faceImage),8,1);
	cvZero(faceSkinMask);

	IplImage* ellipseMask = cvCreateImage(cvGetSize(faceImage),8,1);
	cvZero(ellipseMask);
	CvPoint center = cvPoint(cvRound(ellipseMask->width*0.5),cvRound(ellipseMask->height*0.5));
	CvSize size = cvSize(cvRound(ellipseMask->width*0.36),cvRound(ellipseMask->height*0.50));
	cvEllipse(ellipseMask,center,size,0,0,360,cvScalar(255),CV_FILLED);//������ȡ��Բ�ε�����
	//cvShowImage("ellipseM",ellipseMask);

	long ellipseMask_TotalCnt = 0;
	for (int y=0;y<ellipseMask->height;++y)
	{
		for (int x=0;x<ellipseMask->width;++x)
		{
			if (cvGetReal2D(ellipseMask,y,x)>200)
			{
				ellipseMask_TotalCnt++;
			}
		}
	}

	//��Ƶ�����ֵ
	double iterate_Threshold[17]={0.3,0.2,0.1,0.09,0.07,
		0.05,0.03,0.01,0.009,0.007,0.005,0.003,0.001,0.0005,0.0001,0.00005,0.00001 };

	long skinCntInEllipse = 0;
	for (int i = 0;i<17;i++)
	{
		for (int y=0;y<faceImage_Gauss->height;y++)
		{
			for (int x=0;x<faceImage_Gauss->width;x++)
			{
				if(cvGetReal2D(faceImage_Gauss,y,x)>=iterate_Threshold[i])
				{
					if (cvGetReal2D(faceSkinMask,y,x)<100)
					{
						cvSetReal2D(faceSkinMask,y,x,255);
						if (cvGetReal2D(ellipseMask,y,x)>200)
						{
							skinCntInEllipse++;
						}
					}
				}
			}
		}

		double skinRate = (double)skinCntInEllipse/(double)ellipseMask_TotalCnt;

		//cout<<i<<":"<<iterate_Threshold[i]<<"	"<<"rate:"<<skinRate<<endl;
		//cvShowImage("itface",faceSkinMask);

		if (skinRate>0.82 || iterate_Threshold[i]<0.01)
		{
			break;
		}

		//cvWaitKey(0);
	}

	cvErode(faceSkinMask,faceSkinMask,NULL,1);

	RemoveNoise rn;
	rn.LessConnectedRegionRemove(faceSkinMask,faceSkinMask->height*faceSkinMask->width/8);
	//cvShowImage("faceSkinMask",faceSkinMask);

	//��ȡƤ��
	faceSkinImage = cvCloneImage(faceImage);
	for (int y=0;y<faceSkinMask->height;++y)
	{
		for (int x=0;x<faceSkinMask->width;++x)
		{
			if (cvGetReal2D(faceSkinMask,y,x)<100)
			{
				cvSet2D(faceSkinImage,y,x,CV_RGB(255,255,255));
			}
		}
	}
	//cvShowImage("faceSkinImage",faceSkinImage);

	//release
	cvReleaseImage(&ellipseMask);
}

CvScalar FaceSkinDetectionByGMM::extractFaceSkinColorFeature()
{
	//��ûҶ�ͼ��
	IplImage* faceImageGray = cvCreateImage(cvGetSize(faceImage),8,1);
	cvCvtColor(faceImage,faceImageGray, CV_BGR2GRAY);
	//cvShowImage("fgray",faceImageGray);

	//��һ��ȥ��һЩ����������
	int i_dark,i_bright;
	double rate_dark = 0.1, rate_bright = 0.02;
	calculateDarkAndBrightThreshold(faceSkinMask,faceImageGray,i_dark,i_bright,rate_dark,rate_bright);
	for (int y=0;y<faceSkinMask->height;++y)
	{
		for (int x=0;x<faceSkinMask->width;++x)
		{
			if (cvGetReal2D(faceImageGray,y,x)<i_dark || cvGetReal2D(faceImageGray,y,x)>i_bright)
			{
				cvSetReal2D(faceSkinMask,y,x,0);
			}
		}
	}
	//cvShowImage("faceSkinMaskFine",faceSkinMask);

	CvScalar faceSkinColorFeature = cvAvg(faceImage_Lab,faceSkinMask);

	//release
	cvReleaseImage(&faceImageGray);

	return faceSkinColorFeature;
}

void FaceSkinDetectionByGMM::calculateDarkAndBrightThreshold(const IplImage* mask,const IplImage* grayImage,int& i_dark,int& i_bright,double rate_dark,double rate_bright)
{
	//hash table
	float data[256] = {0};
	int total = 0;
	for (int y=0;y<mask->height;y++)
	{
		for (int x=0;x<mask->width;x++)
		{
			if (cvGetReal2D(mask,y,x)>200)
			{
				int intensity = cvRound(cvGetReal2D(grayImage,y,x));
				data[intensity]++;
				total++;
			}
		}
	}

	//��ȥ�ϰ���rate_dark
	float sum_dark =0;
	for (int i=0;i<256;i++)
	{
		sum_dark += data[i];
		if ((sum_dark)/((double)total)>rate_dark)
		{
			i_dark = i;
			break;
		}
	}
	//printf("T_dark=%d\n",i_dark);

	//��ȥ������rate_bright
	float sum_bright =0;
	for (int i=255;i>=0;i--)
	{
		sum_bright += data[i];
		if ((sum_bright)/((double)total)>rate_bright)
		{
			i_bright = i;
			break;
		}
	}
	//printf("i_bright=%d\n",i_bright);

	//colorhist(data,i_dark,i_bright);//ֱ��ͼ��ʾ
}

//int FaceSkinDetectionByGMM::colorhist(float *data,int i_dark,int i_bright)//��ֱ��ͼ��ʵ��Ӧ�ó����в���Ҫ��
//{
//	int hist_size = 256;
//	int hist_height = 200;
//	float range[] = {0,255};
//	float *ranges[] = {range};
//	//����һάֱ��ͼ
//	CvHistogram* hist = cvCreateHist(1,&hist_size,CV_HIST_ARRAY,ranges,1);
//
//	//�����Ѹ��������ݴ���ֱ��ͼ
//	cvMakeHistHeaderForArray(1,&hist_size,hist,data,ranges,1);
//	//��һ��ֱ��ͼ
//	cvNormalizeHist(hist,1.0);
//
//	//����һ��һάֱ��ͼ�ġ�ͼ����������Ϊ�Ҷȼ���������Ϊ���ظ���
//	int scale = 2;
//	IplImage* hist_image = cvCreateImage(cvSize(hist_size*scale,hist_height),8,3);
//	cvZero(hist_image);
//
//	//ͳ��ֱ��ͼ�е����bin
//	float max_value = 0;
//	cvGetMinMaxHistValue(hist,0,&max_value,0,0);
//
//	//�ֱ�ÿ��bin��ֵ������ͼ��
//	for (int i=0;i<hist_size;i++)
//	{
//		float bin_val = cvQueryHistValue_1D(hist,i);
//		int intensity = cvRound(bin_val*hist_height/max_value);//Ҫ���Ƶĸ߶�
//		if (i==i_dark||i==i_bright)
//		{
//			cvRectangle(hist_image,cvPoint(i*scale,hist_height-1),cvPoint((i+1)*scale-1,hist_height-intensity),CV_RGB(255,0,0));
//		}
//		else
//		{
//			cvRectangle(hist_image,cvPoint(i*scale,hist_height-1),cvPoint((i+1)*scale-1,hist_height-intensity),CV_RGB(0,0,255));
//		}
//	}
//
//	cvShowImage("hist",hist_image);
//
//	return 0;
//}

IplImage* FaceSkinDetectionByGMM::getFaceSkinImage()
{
	return faceSkinImage;
}

IplImage* FaceSkinDetectionByGMM::getFaceImage()
{
	return face;
}