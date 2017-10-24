#include <iostream>

#define ARMA_DONT_USE_WRAPPER
#include <armadillo>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

/*
 * Renders a sequence of mappings applied to an image, into a video.
 *
 * Usage:
 *    cv::Mat src = imread("src.png", cv::IMREAD_COLOR);
 *    Conformal con(src);
 *    const arma::cx_fmat& base = con->get_base();
 *
 *    arma::cx_fmat cubed = arma::pow(base, 3);
 *    for (float lambda = 0.0; lambda < 1.0; lambda += 0.10) {
 *      con->render((1 - lambda) * base + lambda * cubed);
 *    }
 */
class Conformal {
public:
  Conformal(const cv::Mat& src,
      const int fps = 30,
      const std::string& out_name = "out.mp4")
    : src (src)
    , scale (std::min(src.cols, src.rows) / 2.0)
    , offset (arma::cx_float(src.cols / 2, src.rows / 2))
    , sz (cv::Size(src.cols, src.rows))
  {
    // Compute the base plane, which has center at approximately (0, 0) and is
    // scaled down so that the shorter dimension has range [-1.0, 1.0).
    const int cols = src.cols;
    const int rows = src.rows;
    const int mid_x = offset.real();
    const int mid_y = offset.imag();
    arma::mat A = arma::repmat(
        arma::regspace<arma::rowvec>(-mid_x, -mid_x + cols - 1), rows, 1);
    arma::mat B = arma::repmat(
        arma::regspace<arma::colvec>(-mid_y, -mid_y + rows - 1), 1, cols);
    this->base = arma::conv_to<arma::cx_fmat>::from(
        arma::cx_mat(A, B) / scale);

    dst.create(src.size(), src.type());

    const int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    this->vw = cv::VideoWriter(out_name, fourcc, fps, src.size());
  }

  const arma::cx_fmat& get_base() {
    return this->base;
  }

  void render(const arma::cx_fmat& mapping) {
    this->set_mapping(mapping);
    remap(this->src, this->dst, this->map_x, this->map_y,
        cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    vw.write(this->dst);
  }

private:
  const cv::Mat src;
  cv::Mat map_x, map_y, dst;

  const double scale;
  const arma::cx_float offset;
  const cv::Size sz;
  arma::cx_fmat base;

  cv::VideoWriter vw;

  /*
   * Renormalizes the mapping to the image dimensions (reversing the
   * offset/scale of the base plane, and modulo the image dimensions), saving
   * the results as OpenCV Mat objects for use with cv::remap.
   */
  void set_mapping(arma::cx_fmat mapping) {
    arma::fmat norm_re = arma::real(mapping) * this->scale
      + this->offset.real();
    arma::fmat norm_im = arma::imag(mapping) * this->scale
      + this->offset.imag();
    arma::fmat mod_re = posmod(norm_re, this->base.n_cols).t();
    arma::fmat mod_im = posmod(norm_im, this->base.n_rows).t();

    this->map_x = cv::Mat(this->sz, CV_32FC1,
        const_cast<float *>(mod_re.memptr())).clone();
    this->map_y = cv::Mat(this->sz, CV_32FC1,
        const_cast<float *>(mod_im.memptr())).clone();
  }

  /*
   * Returns the element-wise positive modulus (a mod k).
   */
  static inline arma::fmat posmod(const arma::fmat& a, const int k) {
    arma::fmat adk = a / k;
    arma::fmat floor_adk_dec = arma::floor(adk) - 1;
    return (a - k * floor_adk_dec) - k * arma::floor(adk - floor_adk_dec);
  }
};

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Incorrect usage" << std::endl;
    return -1;
  }

  // Read source image
  cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (!src.data) {
    std::cout << "No image data" << std::endl;
    return -1;
  }

  Conformal con(src);
  const arma::cx_fmat& base = con.get_base();

  arma::cx_fmat mapped1 = arma::pow(base, 3);
  arma::cx_fmat mapped2 = arma::exp(base * 3);

  const int frames = 60;
  for (int frame = 0; frame < frames; ++frame) {
    std::cerr << "Rendering frame " << frame << std::endl;
    float lambda = (float) frame / (frames - 1);
    arma::cx_fmat mapping = (1 - lambda) * mapped1 + lambda * mapped2;
    con.render(mapping);
  }

  std::cout << frames << " frames written" << std::endl;
  return 0;
}
