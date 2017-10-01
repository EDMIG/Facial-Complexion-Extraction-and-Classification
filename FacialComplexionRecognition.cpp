/**********************************************************************************
*
*					脸部肤色识别 FaceComplexionRecognition
*					   by Hu yangyang 2016/12/9
*
***********************************************************************************/

#include "FacialComplexionRecognition.h"

FaceComplexionRecognition::FaceComplexionRecognition()
{
	//工程应用时，修改成模型实际加载路径!!!
	colorClassifierPathName = "model\\svm_facecomplexion.xml";
	glossModelPathName = "model\\facegloss_model.xml";
}

FaceComplexionRecognition::~FaceComplexionRecognition(){}

int FaceComplexionRecognition::colorPredict(const CvScalar skinColor)
{
	int response = 5;
	float L;
	float feature[2];//<a,b>
	L = skinColor.val[0];
	feature[0] = skinColor.val[1];//a
	feature[1] = skinColor.val[2];//b

	if (L>=165)
	{
		response = 0;//白
	}
	else if (L<=125)
	{
		response = 1;//黑
	}
	else
	{
		//设置测试数据
		Mat testDataMat(1,2,CV_32FC1,feature);  //测试数据
		//预测
		CvSVM svm = CvSVM();
		svm.load(colorClassifierPathName);
		response = (int)svm.predict(testDataMat);
	}
	return response;
}

/**********************************************************************************
*
*					脸部光泽识别 FaceComplexionRecognition::glossPredict
*                   paper:"中医望诊中光泽判别的研究"
*					   by Hu yangyang 2016/12/9
*
***********************************************************************************/
typedef struct
{
	double ScaleX;
	double ScaleY;
	double ScaleW;
	double ScaleH;
} deRctnglScale;

CvScalar FaceComplexionRecognition::glossPredict(IplImage* inputImage)
{
	int size;
	size = (inputImage->height > inputImage->width) ? inputImage->height : inputImage->width;
	double type = ((double)size)/600.0;
	IplImage* faceImage = NULL;
	if (type<=1)
	{
		faceImage = cvCloneImage(inputImage);
	}
	else
	{
		faceImage = cvCreateImage(cvSize(cvRound(inputImage->width/type),cvRound(inputImage->height/type)),inputImage->depth,inputImage->nChannels);
		cvResize(inputImage,faceImage);
	}

	CvScalar glossResult;//<光泽预测值（0有,1少,2无），有光泽指数，少光泽指数，无光泽指数>
	//cvShowImage("gloss",faceImage);
	//提取右脸颊
	deRctnglScale cheekCoordScale = {0.65, 0.5, 0.6, 0.35};
	int imgScale = 1;
	int H = 100;
	int W = 100;
	CvRect cheekRect;
	cheekRect.x		= cvRound((0 + faceImage->width *cheekCoordScale.ScaleX) * imgScale);
	cheekRect.y		= cvRound((0 + faceImage->height*cheekCoordScale.ScaleY) * imgScale);
	cheekRect.width	= cvRound(faceImage->width * (1-cheekCoordScale.ScaleX) * cheekCoordScale.ScaleW * imgScale);
	cheekRect.height	= cvRound(faceImage->height* (1-cheekCoordScale.ScaleY) * cheekCoordScale.ScaleH * imgScale);
	
	//cvRectangle(faceImage,cvPoint(cheekRect.x,cheekRect.y),cvPoint(cheekRect.x+cheekRect.width,cheekRect.y+cheekRect.height),CV_RGB(255,0,0),2);
	//cvShowImage("cheek",faceImage);

	cvSetImageROI(faceImage,cheekRect);
	IplImage* cheekImage = cvCreateImage(cvGetSize(faceImage),faceImage->depth,faceImage->nChannels);
	cvCopy(faceImage,cheekImage);
	cvResetImageROI(faceImage);
	//cvShowImage("cheekImage",cheekImage);

	//resize
	Mat rszImageM,cheekImageTemp(cheekImage,0);
	resize(cheekImageTemp, rszImageM, Size(W, H), 0, 0);
	//convert to HSV color space
	Mat HSVImageM;
	cvtColor(rszImageM, HSVImageM, CV_RGB2HSV);
	//change format
	Mat img32FC1;
	Mat SnglChnlImageM = Mat(H, 3*W, CV_8UC1, (float*)HSVImageM.data);
	SnglChnlImageM.convertTo(img32FC1, CV_32FC1);
	
	//load XMl
	Mat prjctdM;
	Mat mchIPM;
	Mat lssIPM;
	Mat nnIPM;
	FileStorage fs(glossModelPathName, FileStorage::READ);
	fs["prjctdM"]>> prjctdM;
	fs["mchIPM"] >> mchIPM;
	fs["lssIPM"] >> lssIPM;
	fs["nnIPM"]  >> nnIPM;
	fs.release();
	//project image
	Mat tstPatternM;
	gemm(img32FC1, prjctdM, 1, Mat(), 0, tstPatternM, GEMM_2_T);
	tstPatternM.cols = tstPatternM.cols*tstPatternM.rows;
	tstPatternM.rows = 1;

	Mat MDist(1, &mchIPM.rows, CV_64FC1);
	double tmp;
	for (int i = 0; i < mchIPM.rows; i++)
	{
		tmp = 1 - abs(tstPatternM.dot(mchIPM.row(i))/(norm(tstPatternM, NORM_L2)*norm(mchIPM.row(i), NORM_L2)));
		MDist.at<double>(i) = 1 - sqrt(1 - tmp*tmp);
	}
	minMaxLoc(MDist, 0, &glossResult.val[1]);/*pp.570*/
	MDist.release();
	//
	Mat MDist1(1, &lssIPM.rows, CV_64FC1);
	for (int i = 0; i < lssIPM.rows; i++)
	{
		tmp = 1 - abs(tstPatternM.dot(lssIPM.row(i))/(norm(tstPatternM, NORM_L2)*norm(lssIPM.row(i), NORM_L2)));
		MDist1.at<double>(i) = 1 - sqrt(1 - tmp*tmp);
	}
	minMaxLoc(MDist1, 0, &glossResult.val[2]);/*pp.570*/
	MDist1.release();
	//
	Mat MDist2(1, &nnIPM.rows, CV_64FC1);
	for (int i = 0; i < nnIPM.rows; i++)
	{
		tmp = 1 - abs(tstPatternM.dot(nnIPM.row(i))/(norm(tstPatternM, NORM_L2)*norm(nnIPM.row(i), NORM_L2)));
		MDist2.at<double>(i) = 1 - sqrt(1 - tmp*tmp);
	}
	minMaxLoc(MDist2, 0, &glossResult.val[3]);/*pp.570*/
	MDist2.release();
	//max value
	glossResult.val[0] = glossResult.val[1]>glossResult.val[2]? (glossResult.val[1]>glossResult.val[3]? 1:3) : (glossResult.val[2]>glossResult.val[3]? 2:3);
	glossResult.val[0] -= 1;//从0开始索引
	//cout<<glossResult.val[0]<<"	"<<glossResult.val[1]<<"	"<<glossResult.val[2]<<"	"<<glossResult.val[3]<<endl;
	
	//release
	cvReleaseImage(&faceImage);
	cvReleaseImage(&cheekImage);
	rszImageM.release();
	cheekImageTemp.release();
	HSVImageM.release();
	SnglChnlImageM.release();
	img32FC1.release();
	prjctdM.release();
	mchIPM.release();
	lssIPM.release();
	nnIPM.release();
	tstPatternM.release();

	return glossResult;
}
