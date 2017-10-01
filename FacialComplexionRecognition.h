/**********************************************************************************
*
*					脸部肤色识别 FaceComplexionRecognition
*					   by Hu yangyang 2016/12/9
*
***********************************************************************************/
#ifndef FACECOMPLEXIONRECOGNITION_H
#define FACECOMPLEXIONRECOGNITION_H

#include "Global.h"

class FaceComplexionRecognition
{
public:
	FaceComplexionRecognition();
	~FaceComplexionRecognition();
	
	//返回脸部肤色预测值：0->面白，1->面黑，2->面红,3->面黄，4->面青，5->正常
	int colorPredict(const CvScalar skinColor);

	//返回脸部光泽结果：<光泽预测值（0有,1少,2无），有光泽指数，少光泽指数，无光泽指数>
	CvScalar glossPredict(IplImage* inputImage);

private:
	char* colorClassifierPathName;
	char* glossModelPathName;
};

#endif