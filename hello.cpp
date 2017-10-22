#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

void update_map(int rows, int cols, Mat& map_x, Mat& map_y) {
  const int mid_x = cols / 2;
  const int mid_y = rows / 2;
  int x0, y0;
  std::complex<double> z, z0;

  z.imag(-mid_y);
  for (int r = 0; r < rows; ++r, z.imag(z.imag() + 1)) {
    z.real(-mid_x);
    for (int c = 0; c < cols; ++c, z.real(z.real() + 1)) {
      z0 = std::pow(z, 2) / 100.0;

      x0 = (int) z0.real();
      y0 = (int) z0.imag();

      map_x.at<float>(r, c) = ((x0 + mid_x) % rows + rows) % rows;
      map_y.at<float>(r, c) = ((y0 + mid_y) % cols + cols) % cols;
    }
  }

  std::cout << map_x.at<float>(200, 500) << " " << map_y.at<float>(200, 500) << std::endl;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Incorrect usage" << std::endl;
    return -1;
  }

  Mat src = imread(argv[1], IMREAD_COLOR);

  if (!src.data) {
    std::cout << "No image data" << std::endl;
    return -1;
  }

  Mat dst, map_x, map_y;
  dst.create(src.size(), src.type());
  map_x.create(src.size(), CV_32FC1);
  map_y.create(src.size(), CV_32FC1);

  update_map(src.rows, src.cols, map_x, map_y);
  remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

  std::vector<int> compression_params = {IMWRITE_PNG_COMPRESSION, 9};
  imwrite("out.png", dst);

  return 0;
}
