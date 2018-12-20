//#include </home/senius/tensorflow-r1.4/bazel-genfiles/tensorflow/cc/ops/io_ops.h>
//#include </home/senius/tensorflow-r1.4/bazel-genfiles/tensorflow/cc/ops/parsing_ops.h> 
//#include </home/senius/tensorflow-r1.4/bazel-genfiles/tensorflow/cc/ops/array_ops.h> 
//#include </home/senius/tensorflow-r1.4/bazel-genfiles/tensorflow/cc/ops/math_ops.h> 
//#include </home/senius/tensorflow-r1.4/bazel-genfiles/tensorflow/cc/ops/data_flow_ops.h> 
#include<iostream>
#include <opencv2/opencv.hpp>    // 这个放在前面 才能不被后面的错误使用
#include <opencv2/highgui/highgui.hpp>
#include <tensorflow/core/public/session.h> 
#include <tensorflow/core/protobuf/meta_graph.pb.h> 
#include <fstream> 
using namespace std; 
using namespace tensorflow; 
using namespace cv;
//using namespace tensorflow::ops; 
int main() { 


    //set up your input paths
    const string pathToGraph = "/home/jwei/tensorflow_c/codes/model/liner.pb";
    const string checkpointPath = "/home/jwei/tensorflow_c/codes/model/mylinermodel-900";
    //const string pbpath = "/home/jwei/tensorflow_c/codes/model/liner.pb";

    auto session = NewSession(SessionOptions());
    if(session == nullptr)
    {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

    //Read in the protobuf graph we exported
    //MetaGraphDef graph_def;
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(),pathToGraph,&graph_def);
    if(!status.ok())
    {
        throw runtime_error("Error reading graph definition from "+pathToGraph+": "+status.ToString());

    }

    //Add the graph to the session
    //status = session->Create(graph_def.graph_def());
    status = session->Create(graph_def);
    if(!status.ok())
    {
        throw runtime_error("Error creating graph: "+status.ToString());
    }

//    //Read weights from the saved checkpoint
//    Tensor checkpointPathTensor(DT_STRING,TensorShape());
//    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
//    status = session->Run({{graph_def.saver_def().filename_tensor_name(),checkpointPathTensor},},{},{graph_def.saver_def().restore_op_name()},nullptr);
//    if(!status.ok())
//    {
//        throw runtime_error("Error loading checkpoint from "+checkpointPath+": "+status.ToString());
//    }

    cout<<1<<endl;

    //const string filename = "/home/jwei/tensorflow_c/data/04t30t00.npy";
    const string filename = "/home/jwei/tensorflow_c/data/4.jpg";

//    //Read TXT data to array
//    float Array[1681*41];
//    ifstream is (filename);
//    for(int i = 0;i<1681*41;i++){
//        is >> Array[i];
//    }
//    is.close();

//    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,tensorflow::TensorShape({1,41,41,41,1}));

//    auto input_tensor_mapped = input_tensor.tensor<float,5>();

//    float *pdata = Array;

//    //copying the data into the corresponding tensor
//    for(int x = 0; x < 41; ++x)//depth
//    {
//        for(int y = 0; y < 41; ++y){
//            for(int z = 0; z < 41; ++z){
//                const float *source_value = pdata + x*1681+y*41+z;
//                //input_tensor_mapped(0,x,y,0) = *source_value;
//                input_tensor_mapped(0, x, y, z, 0) = 1;
//            }
//        }
//    }



//    //Read image data to array
//    float Array[224][224][3];
//    Mat inimage = imread(filename,CV_LOAD_IMAGE_COLOR);
//    if(inimage.empty())
//    {
//        throw runtime_error("can't load image.");
//    }

//    cout<<"size of image: "<<inimage.size()<<endl;
//    //imshow("inimage",inimage);

//    Mat outimage;

//    resize(inimage,outimage,Size(224,224),0,0,INTER_LINEAR);
//    cout<<"new size of image: "<<outimage.size()<<endl;
//    //imshow("outimage",outimage);

//    //waitKey();

//    cout<<"2"<<endl;

//    for(int row = 0; row<outimage.rows;row++)
//    {
//        for(int col = 0;col<outimage.cols;col++)
//        {
//            Array[row][col][0]=outimage.at<Vec3b>(row,col)[0]-123;
//            Array[row][col][1]=outimage.at<Vec3b>(row,col)[1]-117;
//            Array[row][col][2]=outimage.at<Vec3b>(row,col)[2]-104;
//        }
//    }

//    Tensor input_tensor(DT_FLOAT,TensorShape({1,224,224,3}));//1,width,height,depth
//    auto input_tensor_mapped = input_tensor.tensor<float,4>();
//    for (int i =0;i<224;i++)
//    {
//        for(int ii=0;ii<224;ii++)
//        {
//            for(int iii=0;iii<3;iii++)
//            {
//                input_tensor_mapped(0,i,ii,iii)=Array[i][ii][iii];
//            }
//        }
//    }
//    cout<<input_tensor.shape()<<endl;

//    cout<<"3"<<endl;

    Tensor input_tensor(DT_FLOAT,TensorShape({1,1,1,1}));
    auto input_tensor_mapped = input_tensor.tensor<float,4>();
    for(int i=1;i<3;i++)
    {
        input_tensor_mapped(0,0,0,0)=0.5;
    }
    cout<<input_tensor.shape()<<endl;



    std::vector<tensorflow::Tensor> finalOutput;
    std::string InputName = "inputs";//your input placeholder's name
    std::string OutputName = "outputs";//your output placeholder's name
    vector<std::pair<string,Tensor>> inputs;
    //inputs.push_back(std::make_pair(InputName,input_tensor));
    inputs.push_back(std::pair<std::string,tensorflow::Tensor>(InputName,input_tensor));
    cout<<inputs.size()<<endl;

    cout<<"4"<<endl;

    //Fill input tensor with your input data
    status = session->Run(inputs,{OutputName},{},&finalOutput);
    if(!status.ok())
    {
        throw runtime_error("prediction failed...");
    }

    cout<<"5"<<endl;

//    cout<<"output tensor size: "<<finalOutput.size()<<endl;//输出结果的维度
//    for(std::size_t i=0;i<finalOutput.size();i++)
//    {
//        cout<<finalOutput[i].DebugString();//输出结果每个分量
//    }
//    cout<<endl;


    auto output_y = finalOutput[0].scalar<float>();
    cout<<"6"<<endl;
    cout<<"output: "<<output_y()<<endl;

    cout<<"7"<<endl;


    return 0;
}

