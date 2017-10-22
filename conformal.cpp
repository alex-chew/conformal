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
  const double mid_x = cols / 2.0;
  const double mid_y = rows / 2.0;

  arma::mat A = arma::repmat(
      arma::regspace<arma::rowvec>(-mid_x, mid_x - 1), rows, 1);
  arma::mat B = arma::repmat(
      arma::regspace<arma::colvec>(-mid_y, mid_y - 1), 1, cols);
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
    arma::cx_double(base.n_cols, base.n_rows) / 2.0;
  arma::cx_mat img = base * scale + offset;
  arma::dmat re = posmod(arma::real(img), base.n_cols);
  arma::dmat im = posmod(arma::imag(img), base.n_rows);

  Size sz(base.n_cols, base.n_rows);
  out_x = Mat(sz, CV_32FC1, const_cast<double *>(re.memptr())).clone();
  out_y = Mat(sz, CV_32FC1, const_cast<double *>(im.memptr())).clone();
}

/*
 * Maps pixels of a (rows) by (cols) image by treating their components as a
 * complex number, and performing complex exponentiation to the (pow)th power.
 */
class PowerMap {
public:
  PowerMap(double pow, int rows, int cols) : pow (pow),
    rows (rows),
    cols (cols),
    offset (std::complex<double>(cols / 2, rows / 2)),
    scale (std::min(cols, rows) / 2.0) { }

  void operator ()(Pixel &p, const int *pos) const {
    std::complex<double> z =
      (std::complex<double>(pos[1], pos[0]) - offset) / scale;
    z = std::pow(z, pow) * scale + offset;
    p.x = ((int) z.real() % cols + cols) % cols;
    p.y = ((int) z.imag() % rows + rows) % rows;
  }

private:
  double pow;
  int rows;
  int cols;
  std::complex<double> offset;
  double scale;
};

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
  arma::cx_mat sq = arma::pow(base, 2);
  Mat map_x, map_y;
  map_x.create(src.size(), CV_32FC1);
  map_y.create(src.size(), CV_32FC1);
  image_plane(sq, scale, map_x, map_y);

  Mat dst;
  dst.create(src.size(), src.type());

  std::vector<int> compression_params = {IMWRITE_PNG_COMPRESSION, 9};

  remap(src, dst, map_x, map_y,
      INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

  // Write dst
  std::ostringstream framename;
  framename << "out"
    << std::setfill('0') << std::setw(4) << 0 << ".png";
  imwrite(framename.str(), dst, compression_params);
  return 0;
}
