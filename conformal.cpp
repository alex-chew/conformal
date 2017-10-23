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
arma::cx_mat base_plane(const int rows, const int cols,
    const double scale,
    const arma::cx_float offset) {
  const int mid_x = offset.real();
  const int mid_y = offset.imag();

  arma::mat A = arma::repmat(
      arma::regspace<arma::rowvec>(-mid_x, -mid_x + cols - 1), rows, 1);
  arma::mat B = arma::repmat(
      arma::regspace<arma::colvec>(-mid_y, -mid_y + rows - 1), 1, cols);
  return arma::cx_mat(A, B) / scale;
}

inline arma::fmat posmod(const arma::fmat& a, const int k) {
  arma::fmat adk = a / k;
  arma::fmat floor_adk_dec = arma::floor(adk) - 1;
  return (a - k * floor_adk_dec) - k * arma::floor(adk - floor_adk_dec);
}

void image_plane(const arma::cx_fmat& base,
    const double scale,
    const arma::cx_float offset,
    Mat& out_x, Mat& out_y) {
  std::cout << "  norm" << std::endl;
  arma::fmat norm_re = arma::real(base) * scale + offset.real();
  arma::fmat norm_im = arma::imag(base) * scale + offset.imag();
  std::cout << "  mod" << std::endl;
  arma::fmat mod_re = posmod(norm_re, base.n_cols).t();
  arma::fmat mod_im = posmod(norm_im, base.n_rows).t();

  std::cout << "  copy" << std::endl;
  Size sz(base.n_cols, base.n_rows);
  out_x = Mat(sz, CV_32FC1, const_cast<float *>(mod_re.memptr())).clone();
  out_y = Mat(sz, CV_32FC1, const_cast<float *>(mod_im.memptr())).clone();
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

  const double scale = min(src.cols, src.rows) / 2.0;
  const arma::cx_float offset = arma::cx_float(src.cols / 2, src.rows / 2);

  arma::cx_fmat base = arma::conv_to<arma::cx_fmat>::from(
      base_plane(src.rows, src.cols, scale, offset));
  Mat map_x, map_y;

  Mat dst;
  dst.create(src.size(), src.type());

  VideoWriter vw("out.mp4", CV_FOURCC('H', '2', '6', '4'), 30, src.size());

  for (float i = 0.1; i < 2.0; i += 0.08) {
    std::cout << "Computing map for z^" << i << std::endl;
    image_plane(arma::pow(base, i), scale, offset, map_x, map_y);
    std::cout << "Remapping image for z^" << i << std::endl;
    remap(src, dst, map_x, map_y,
        INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

    // Write dst
    std::cout << "Writing image for z^" << i << std::endl;
    vw.write(dst);
  }

  return 0;
}
