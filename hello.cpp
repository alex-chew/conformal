#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

typedef Point2f Pixel;

void update_map(int rows, int cols, Mat& pixmap) {
  const std::complex<double> offset = std::complex<double>(
      cols / 2, rows / 2);
  const double zpow = 5;
  const double scale = (double) std::min(offset.real(), offset.imag());

  pixmap.forEach<Pixel>([&rows, &cols, &offset, &zpow, &scale](Pixel& p, const int *pos) -> void {
      std::complex<double> z = std::complex<double>(pos[1], pos[0]) - offset;
      z = std::pow(z / scale, zpow) * scale + offset;
      p.x = ((int) z.real() % cols + cols) % cols;
      p.y = ((int) z.imag() % rows + rows) % rows;
      });
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

  Mat dst;
  dst.create(src.size(), src.type());
  Mat pixmap;
  pixmap.create(src.size(), CV_32FC2);

  update_map(src.rows, src.cols, pixmap);
  remap(src, dst, pixmap, noArray(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

  std::vector<int> compression_params = {IMWRITE_PNG_COMPRESSION, 9};
  imwrite("out.png", dst);

  return 0;
}
