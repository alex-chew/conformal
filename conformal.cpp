#include <iomanip>
#include <iostream>

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

typedef Point2f Pixel;

/*
 * Returns a complex plane of the specified size, which has center at
 * approximately (0, 0) and is scaled down by the specified scale.
 */
arma::cx_mat base_plane(int rows, int cols, double scale) {
  const int mid_x = cols / 2;
  const int mid_y = rows / 2;

  arma::mat A = arma::repmat(
      arma::regspace<arma::rowvec>(-mid_x, -mid_x + cols - 1), rows, 1);
  arma::mat B = arma::repmat(
      arma::regspace<arma::colvec>(-mid_y, -mid_y + rows - 1), 1, cols);
  return arma::cx_mat(A, B) / scale;
}

inline arma::fmat matmod(arma::fmat a, int k) {
  return a - k * arma::floor(a / k);
}

inline arma::fmat posmod(arma::fmat a, int k) {
  return matmod(matmod(a, k) + k, k);
}

void image_plane(const arma::cx_fmat& base, const double scale,
    Mat& out_x, Mat& out_y) {
  const arma::cx_float offset =
    arma::cx_float(base.n_cols / 2, base.n_rows / 2);
  arma::cx_fmat img = base * scale + offset;
  std::cout << "  posmod" << std::endl;
  arma::fmat re = arma::conv_to<arma::fmat>::from(
      posmod(arma::real(img), base.n_cols)).t();
  arma::fmat im = arma::conv_to<arma::fmat>::from(
      posmod(arma::imag(img), base.n_rows)).t();

  std::cout << "  copy to Mat" << std::endl;
  Size sz(base.n_cols, base.n_rows);
  out_x = Mat(sz, CV_32FC1, const_cast<float *>(re.memptr())).clone();
  out_y = Mat(sz, CV_32FC1, const_cast<float *>(im.memptr())).clone();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Incorrect usage" << std::endl;
    return -1;
  }

  // Read source image
  Mat src = imread(argv[1], IMREAD_COLOR);
  if (!src.data) {
    std::cout << "No image data" << std::endl;
    return -1;
  }

  double scale = min(src.cols, src.rows) / 2.0;
  arma::cx_fmat base = arma::conv_to<arma::cx_fmat>::from(
      base_plane(src.rows, src.cols, scale));
  Mat map_x, map_y;

  Mat dst;
  dst.create(src.size(), src.type());

  VideoWriter vw("out.mp4", CV_FOURCC('H', '2', '6', '4'), 30, src.size());

  for (float i = 0.1; i < 2.0; i += 0.04) {
    std::cout << "Computing map for z^" << i << std::endl;
    image_plane(arma::pow(base, i), scale, map_x, map_y);
    std::cout << "Remapping image for z^" << i << std::endl;
    remap(src, dst, map_x, map_y,
        INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

    // Write dst
    std::cout << "Writing image for z^" << i << std::endl;
    vw.write(dst);
  }

  return 0;
}
