#include <iostream>
#include <opencv2/opencv.hpp>    // �������ǰ�� ���ܲ�������Ĵ���ʹ��
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "ann_model_loader.h"

using namespace cv;
using namespace tensorflow;
using namespace std;

int main(int argc, char* argv[]) {

	//python�д���
	//pil_image = pil_image.resize(_IMAGE_SIZE,Image.ANTIALIAS) - np.array([123, 117, 104])  # ����ͼƬ��С,�������뵽ģ�����.pbģ���ļ������Ϊ224,224,3
	//pil_image = np.expand_dims(pil_image[:, :, ::-1], axis=0)  # ����һ��ά��


	if (argc != 5) {
        	std::cout << "WARNING: Input Args missing" << std::endl;
        	return 0;
    	}
	std::string image_path = argv[1];  //ͼƬ�ļ���ַ
    	std::string model_path = argv[2];  // pbģ���ļ���ַ
        std::string input_tensor_name = argv[3];  // ģ��������ڵ������   //����ڵ�����  ��ȥpbģ����ƥ��,��������load.py�鿴�ڵ���Ϣ
	std::string output_tensor_name = argv[4];  // ģ��������ڵ������   //����ڵ������  ��ȥpbģ����ƥ��,��������load.py�鿴�ڵ���Ϣ

	//vsafety/data:0    ����ڵ�����
	//vsafety/fc2_joint8/fc2_joint8:0    ����ڵ�����
	//safety_ns.pb��ģ������data    ����ڵ�
	//safety_ns.pb��ģ�����prob    ����ڵ�
	//ʵ��ģ������ڵ�����"input"
	//ʵ��ģ������ڵ�����"MobilenetV1/Predictions/Reshape_1"

 
	//����ͼ��
	Mat srcImage=imread(image_path,CV_LOAD_IMAGE_COLOR);   //����ͼƬ, ͨ����Ϊ3
	if(srcImage.empty())
	{
		printf("can not load image \n");
		return -1;
	}
    std::cout<<"ͼƬά��:"<<srcImage.size()<<std::endl;

	Mat dstImage;

	//�ߴ����
	resize(srcImage,dstImage,Size(224,224),0,0,INTER_LINEAR);   // �����ά
    std::cout<<"��ͼƬά��:"<<dstImage.size()<<std::endl;
	double data[224][224][3];
	//�����ͨ������,����ģ�ͼ���������4ͨ��(224,224,3,1)����(1,224,224,3)
	for(int row = 0; row < dstImage.rows; row++)
	{
		for(int col = 0; col < dstImage.cols; col++)
		{
			data[row][col][0]=dstImage.at<Vec3b>(row, col)[0]-123;    // ��������ֵ,ʹ�ñ�׼��
			data[row][col][1]=dstImage.at<Vec3b>(row, col)[1]-117;    // ��������ֵ,ʹ�ñ�׼��
			data[row][col][2]=dstImage.at<Vec3b>(row, col)[2]-104;    // ��������ֵ,ʹ�ñ�׼��
		   	// std::cout<<"��:"<<r<<"��:"<<g<<"��:"<<b<<std::endl; 
		}
	 }

	

	//IplImage* imgClr = cvCreateImage(Size(224,224), IPL_DEPTH_8U, 3);
 
    	// �����µ�Session
    	Session* session;
    	Status status = NewSession(SessionOptions(), &session);
    	if (!status.ok()) {
        	std::cout << status.ToString() << "\n";
        	return 0;
    	}
 
    	// ����Ԥ��demo
    	tf_model::ANNModelLoader model;  // ��������ģ��Ԥ��
    	if (0 != model.load(session, model_path)) {
        	std::cout << "Error: Model Loading failed..." << std::endl;
        	return 0;
    	}

 
    	// ��������������ת����,�����ݰ󶨵���Ӧ������ڵ���
    	tf_model::ANNFeatureAdapter input_feat;

    	input_feat.assign(input_tensor_name,data,224,224,3);   //��vector������ֵ������ڵ�
 
    	// ����Ԥ����
    	double prediction[100];
    	if (0 != model.predict(session, input_feat, output_tensor_name, prediction)) {
        	std::cout << "WARNING: Prediction failed..." << std::endl;
    	}
    	std::cout << "Output Prediction Value:" << prediction[0] << std::endl;
 
    	return 0;
}
