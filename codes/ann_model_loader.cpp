#include <iostream>
#include <vector>
#include <map>
#include "ann_model_loader.h"
//#include <tensor_shape.h>
 
using namespace tensorflow;
 
namespace tf_model {
 
/**
 * ANNFeatureAdapter Implementation
 * */
    ANNFeatureAdapter::ANNFeatureAdapter() {
 
    }
 
    ANNFeatureAdapter::~ANNFeatureAdapter() {
 
    }
 
/*
 * @brief: Feature Adapter: convert 1-D double vector to Tensor, shape [1, ndim]
 * @param: std::string tname, tensor name;
 * @parma: std::vector<std::vector<std::vector<double>>>*, input vector;
 * */
    void ANNFeatureAdapter::assign(std::string tname,double data[224][224][3],int width,int height,int deep) {   //´«ÈëÈıÎ¬¶ÈÏòÁ¿
        //½«ÈıÎ¬¶ÈÏòÁ¿Êı¾İ¸³Öµµ½ÍøÂç½ÚµãÉÏ
        if (width == 0 || height==0 || deep==0) {
            std::cout << "WARNING: Input Vec size is 0 ..." << std::endl;
            return;
        }
        // Create New tensor and set value
        Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,width, height,deep})); // ´´Ôì½Úµã,Î¬¶ÈÎª224,224,3,1,Ôö¼ÓÒ»¸öÎ¬¶ÈÎªÁËÊÊÓ¦pbÄ£ĞÍ
        auto x_map = x.tensor<float, 4>();  //´´½¨4¸öÎ¬¶È¿Õ¼äµÄ½Úµã
	
        for (int i = 0; i < width; i++) {
          for (int ii = 0; ii < height; ii++) {
            for (int iii = 0; iii < deep; iii++) {
                x_map(0,i,ii,iii) = data[i][ii][iii];  // *(data + height*deep * i + ii * deep + iii);    // Õâ¾ä»°ÓĞ´ıÑéÖ¤
            }
          }	
        }
        // Append <tname, Tensor> to input
        input.push_back(std::pair<std::string, tensorflow::Tensor>(tname, x));   //½«ÍøÂç½ÚµãÃûºÍÊı¾İ½øĞĞ°ó¶¨
    }
 
/**
 * ANN Model Loader Implementation
 * */
    ANNModelLoader::ANNModelLoader() {
 
    }
 
    ANNModelLoader::~ANNModelLoader() {
 
    }
 
/**
 * @brief: load the graph and add to Session
 * @param: Session* session, add the graph to the session
 * @param: model_path absolute path to exported protobuf file *.pb
 * */
 
    int ANNModelLoader::load(tensorflow::Session* session, const std::string model_path) {
        //Read the pb file into the grapgdef member
        tensorflow::Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
        if (!status_load.ok()) {
            std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
            std::cout << status_load.ToString() << "\n";
            return -1;
        }
 
        // Add the graph to the session

        tensorflow::Status status_create = session->Create(graphdef);
        //status_load = session->Create(graphdef);
        if (!status_create.ok()) {
            std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
            return -1;
        }
        return 0;
    }
 
/**
 * @brief: Making new prediction
 * @param: Session* session
 * @param: FeatureAdapterBase, common interface of input feature
 * @param: std::string, output_node, tensorname of output node
 * @param: double, prediction values
 * */
 
    int ANNModelLoader::predict(tensorflow::Session* session, const FeatureAdapterBase& input_feature,
                                const std::string output_node, double* prediction) {
        // The session will initialize the outputs
        std::vector<tensorflow::Tensor> outputs;         //shape  [batch_size]
 
        // @input: vector<pair<string, tensor> >, feed_dict
        // @output_node: std::string, name of the output node op, defined in the protobuf file
        tensorflow::Status status = session->Run(input_feature.input, {output_node}, {}, &outputs);
        if (!status.ok()) {
            std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
            return -1;
        }





 
        //»ñÈ¡Êä³ö½á¹û
        std::cout << "Output tensor size:" << outputs.size() << std::endl;   //Êä³ö½á¹ûµÄÎ¬¶È
        for (std::size_t i = 0; i < outputs.size(); i++) {
            std::cout << outputs[i].DebugString();   //Êä³ö½á¹ûÃ¿¸ö·ÖÁ¿
        }
        std::cout << std::endl;
 
        Tensor t = outputs[0];                   // Fetch the first tensor
        int ndim = t.shape().dims();             // Get the dimension of the tensor
        auto tmap = t.tensor<float, 2>();        // Tensor Shape: [Ñù±¾¸öÊı, ·ÖÀà¸öÊı]
        int output_dim = t.shape().dim_size(1);  // Get the target_class_num from 1st dimension
        std::vector<double> tout;
 
        // È¡Êä³öæ¦????×î´óµÄÒ»¸ö·ÖÁ¿×÷Îª×îÖÕ·ÖÀà,¼ÆËãÀà±êºÅºÍ¸ÅÂÊ
        int output_class_id = -1;
        double output_prob = 0.0;
        for (int j = 0; j < output_dim; j++) {
            std::cout << "Class " << j << " prob:" << tmap(0, j) << "," << std::endl;
            if (tmap(0, j) >= output_prob) {
                output_class_id = j;
                output_prob = tmap(0, j);
            }
        }
 
        // ´òÓ¡ÈÕÖ¾
        std::cout << "Final class id: " << output_class_id << std::endl;
        std::cout << "Final value is: " << output_prob << std::endl;
 
        (*prediction) = output_prob;   // ·µ»ØÔ¤²âËùÊô·Öç??µÄ¸ÅÂÊ  
        return 0;
    }
 
}
