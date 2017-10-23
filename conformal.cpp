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

inline arma::mat matmod(arma::mat a, int k) {
  return a - k * arma::floor(a / k);
}

inline arma::mat posmod(arma::mat a, int k) {
  return matmod(matmod(a, k) + k, k);
}

void image_plane(const arma::cx_mat& base, double scale,
    Mat& out_x, Mat& out_y) {
  const arma::cx_double offset =
    arma::cx_double(base.n_cols / 2, base.n_rows / 2);
  arma::cx_mat img = base * scale + offset;
  arma::fmat re = arma::conv_to<arma::fmat>::from(
      posmod(arma::real(img), base.n_cols)).t();
  arma::fmat im = arma::conv_to<arma::fmat>::from(
      posmod(arma::imag(img), base.n_rows)).t();

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
  arma::cx_mat base = base_plane(src.rows, src.cols, scale);
  arma::cx_mat sq = arma::pow(base, 1);
  Mat map_x, map_y;
  map_x.create(src.size(), CV_32FC1);
  map_y.create(src.size(), CV_32FC1);

  Mat dst;
  dst.create(src.size(), src.type());

  std::vector<int> compression_params = {IMWRITE_PNG_COMPRESSION, 9};

  for (int i = 1; i <= 10; ++i) {
    image_plane(arma::pow(base, i), scale, map_x, map_y);
    remap(src, dst, map_x, map_y,
        INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

    // Write dst
    std::ostringstream framename;
    framename << "out"
      << std::setfill('0') << std::setw(4) << i << ".png";
    imwrite(framename.str(), dst, compression_params);
  }

  return 0;
}
