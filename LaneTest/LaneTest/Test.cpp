#include<opencv2\opencv.hpp>
#include<iostream>
using namespace cv;
Mat toGray(Mat src);
Mat gaussian(Mat src,int kenerl_size);
Mat cannyFind(Mat src,int low_threshold,int high_threshold);
Mat RIO(Mat src);
int main(int argc,char **argv)
{
	/*
	Mat srcImage;
	Mat grayImage;
	Mat blurImage;
	Mat cannyImage;

	srcImage = imread("D:\\lane_Moment.jpg");
	//imshow("原图",srcImage);
	cvtColor(srcImage,grayImage,CV_BGR2GRAY);
	imshow("灰度图",grayImage);
	int kenerl_size = 5;
	GaussianBlur(grayImage,blurImage,Size(kenerl_size,kenerl_size),0);

	imshow("高斯滤波平滑化后",blurImage);

	int low_threshold = 50;
	int high_threshold = 150;

	Canny(blurImage,cannyImage,low_threshold,high_threshold);
	imshow("边缘检测",cannyImage);

	Size imgsize = cannyImage.size();

	Point rookpoint[1][3];
	rookpoint[0][0] = Point(0,imgsize.height-57);
	rookpoint[0][1] = Point(imgsize.width/2,imgsize.height/2+20);
	rookpoint[0][2] = Point(imgsize.width,imgsize.height-57);

	const Point* ppt[1] = {rookpoint[0]};
	int npt[] = {3};
	polylines(cannyImage,ppt,npt,1,1,Scalar(0,0,0),1,8,0);
	Mat mask_ann, imageRIO;
	cannyImage.copyTo(mask_ann);
	mask_ann.setTo(Scalar::all(0));
	fillPoly(mask_ann,ppt,npt,1,Scalar(255,255,255));
	cannyImage.copyTo(imageRIO, mask_ann);
	imshow("设置RIO",imageRIO);
	
	
	Mat houghImage;
	srcImage.copyTo(houghImage);
	vector<Vec4i> lines;
	HoughLinesP(imageRIO,lines,1,CV_PI/180,50,10,10);
	for( size_t i = 0; i < lines.size(); i++ )
	{
		Vec4i l = lines[i];
		line( houghImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186,88,255), 3, CV_AA);
	}
	imshow("霍夫变换",houghImage);
	std::cout<<lines.size()<<std::endl;*/



	VideoCapture capture;
	capture.open("H:\\project_video.mp4");
	//capture.open("D:\\lane.avi");
	if (!capture.isOpened())  
    {  
       std::cout << "Read video Failed !" << std::endl;  
       return 0;  
	}

	Mat frame;
	namedWindow("video test");

	while(1)
	{
		capture>>frame;
		Mat result;
		result = toGray(frame);
		result = gaussian(result,5);
		result = cannyFind(result,50,150);
		result = RIO(result);
		Mat houghImage;
		frame.copyTo(houghImage);
		vector<Vec4i> lines;
		HoughLinesP(result,lines,1,CV_PI/180,50,10,10);
		for( size_t i = 0; i < lines.size(); i++ )
		{
			Vec4i l = lines[i];
			line( houghImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186,88,255), 3, CV_AA);
		}
		imshow("video test",houghImage);
		if (waitKey(30) == 'q')  
        {  
           break;  
        }  
	}

	waitKey(0);

}

Mat toGray(Mat src)
{
	Mat result;
	cvtColor(src,result,CV_BGR2GRAY);
	return result;
}

Mat gaussian(Mat src,int kenerl_size)
{
	Mat result;
	GaussianBlur(src,result,Size(kenerl_size,kenerl_size),0);
	return result;
}

Mat cannyFind(Mat src,int low_threshold,int high_threshold)
{
	Mat result;
	Canny(src,result,low_threshold,high_threshold);
	return result;
}

Mat RIO(Mat src)
{
	Size imgsize = src.size();
	Point rookpoint[1][3];
	/*rookpoint[0][0] = Point(0,imgsize.height-65);
	rookpoint[0][1] = Point(imgsize.width/2,imgsize.height/2+40);
	rookpoint[0][2] = Point(imgsize.width,imgsize.height-65);*/
	
	rookpoint[0][0] = Point(0,imgsize.height-65);
	rookpoint[0][1] = Point(imgsize.width/2,imgsize.height/2+100);
	rookpoint[0][2] = Point(imgsize.width,imgsize.height-65);
	const Point* ppt[1] = {rookpoint[0]};
	int npt[] = {3};
	polylines(src,ppt,npt,1,1,Scalar(0,0,0),1,8,0);
	Mat mask_ann, imageRIO;
	src.copyTo(mask_ann);
	mask_ann.setTo(Scalar::all(0));
	fillPoly(mask_ann,ppt,npt,1,Scalar(255,255,255));
	src.copyTo(imageRIO, mask_ann);
	return imageRIO; 
}



