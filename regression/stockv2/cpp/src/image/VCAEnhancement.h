#pragma once

#define ENHANCEMENT_FUNCTION_SSR
#define ENHANCEMENT_FUNCTION_MSRCR

void SSR_Enhancement(Mat &src, Mat &dst);
void MSRCR_Enhancement(Mat& object_image);
void apiBilateralFilter(Mat &input_image, Mat &output_image, int bilateralFilter_size=24);



