/**********************************************************************************
*
*					������ɫʶ�� FaceComplexionRecognition
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
	
	//����������ɫԤ��ֵ��0->��ף�1->��ڣ�2->���,3->��ƣ�4->���࣬5->����
	int colorPredict(const CvScalar skinColor);

	//����������������<����Ԥ��ֵ��0��,1��,2�ޣ����й���ָ�����ٹ���ָ�����޹���ָ��>
	CvScalar glossPredict(IplImage* inputImage);

private:
	char* colorClassifierPathName;
	char* glossModelPathName;
};

#endif