#include <iostream>

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
 *    const arma::cx_fmat& base = con.get_base();
 *
 *    arma::cx_fmat cubed = arma::pow(base, 3);
 *    for (float lambda = 0.0; lambda < 1.0; lambda += 0.10) {
 *      con.render((1 - lambda) * base + lambda * cubed);
 *    }
 */
class Conformal {
public:
  Conformal(const cv::Mat& src,
      const int fps = 30,
      const std::string& out_name = "out.mp4")
    : src_ (src)
    , scale_ (std::min(src.cols, src.rows) / 2.0)
    , offset_ (arma::cx_float(src.cols / 2, src.rows / 2))
    , sz_ (cv::Size(src.cols, src.rows))
    , frames_ (0)
  {
    // Compute the base plane, which has center at approximately (0, 0) and is
    // scaled down so that the shorter dimension has range [-1.0, 1.0).
    const int cols = src_.cols;
    const int rows = src_.rows;
    const int mid_x = offset_.real();
    const int mid_y = offset_.imag();
    arma::mat A = arma::repmat(
        arma::regspace<arma::rowvec>(-mid_x, -mid_x + cols - 1), rows, 1);
    arma::mat B = arma::repmat(
        arma::regspace<arma::colvec>(-mid_y, -mid_y + rows - 1), 1, cols);
    base_ = arma::conv_to<arma::cx_fmat>::from(
        arma::cx_mat(A, B) / scale_);

    dst_.create(src_.size(), src_.type());

    const int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
    vw_ = cv::VideoWriter(out_name, fourcc, fps, src.size());
  }

  const arma::cx_fmat& get_base() const {
    return base_;
  }

  void render(const arma::cx_fmat& mapping) {
    set_mapping(mapping);
    remap(src_, dst_, map_x_, map_y_,
        cv::INTER_LINEAR, cv::BORDER_WRAP, cv::Scalar(0, 0, 0));
    vw_.write(dst_);
    ++frames_;
  }

  unsigned int frames_rendered() {
    return frames_;
  }

private:
  const cv::Mat src_;
  cv::Mat map_x_, map_y_, dst_;

  const double scale_;
  const arma::cx_float offset_;
  const cv::Size sz_;
  arma::cx_fmat base_;

  cv::VideoWriter vw_;
  unsigned int frames_;

  /*
   * Renormalizes the mapping to the image dimensions (reversing the
   * offset/scale of the base plane, and modulo the image dimensions), saving
   * the results as OpenCV Mat objects for use with cv::remap.
   */
  void set_mapping(arma::cx_fmat mapping) {
    arma::fmat norm_re = arma::real(mapping) * scale_ + offset_.real();
    arma::fmat norm_im = arma::imag(mapping) * scale_ + offset_.imag();
    arma::fmat mod_re = matmod(norm_re, base_.n_cols).t();
    arma::fmat mod_im = matmod(norm_im, base_.n_rows).t();

    map_x_ = cv::Mat(sz_, CV_32FC1,
        const_cast<float *>(mod_re.memptr())).clone();
    map_y_ = cv::Mat(sz_, CV_32FC1,
        const_cast<float *>(mod_im.memptr())).clone();
  }

  /*
   * Returns the element-wise modulus (a mod k).
   */
  static inline arma::fmat matmod(const arma::fmat& a, const int k) {
    return a - k * arma::floor(a / k);
  }
};

/*
 * Interpolation coefficients based on sin^2(x) from 0 to pi/2.
 */
template <typename ArmaVec>
ArmaVec eased_interp(unsigned int steps) {
  return arma::square(arma::sin(
        arma::linspace<ArmaVec>(0, arma::fdatum::pi / 2, steps)));
}

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

  // Actual conformal mappings
  std::vector<arma::cx_fmat> maps = {
    base,
    arma::pow(base, 2),
    arma::pow(base, -1),
    arma::exp(base * 3),
    arma::sin(base * 3)
  };

  // Frames spent on translation
  const unsigned int time_translate = 120;
  // Frames spent on transition
  const unsigned int time_transition = 60;

  // Translation constants based on r = 1 + cos(2*theta) from 0 to 2*pi
  const arma::vec theta_space =
    eased_interp<arma::vec>(time_translate) * 2 * arma::fdatum::pi;
  const arma::vec r_space = (1 - arma::cos(theta_space)) / 2;
  const arma::vec translate_x = r_space % arma::cos(theta_space);
  const arma::vec translate_y = r_space % arma::sin(theta_space);
  const arma::cx_fvec translate_c = arma::conv_to<arma::cx_fvec>::from(
      arma::cx_vec(translate_x, translate_y));

  // Interpolation coefficients
  const arma::fvec interp_c = eased_interp<arma::fvec>(time_transition);

  unsigned int m, frame;
  float lambda;
  arma::cx_fmat mapping;
  for (m = 0; m < maps.size(); ++m) {
    std::cerr << "Rendering mapping " << m;
    const auto& m_curr = maps[m];
    const auto& m_next = maps[(m + 1) % maps.size()];

    // Translate
    if (m > 0) {
      for (frame = 0; frame < time_translate; ++frame) {
        con.render(m_curr + translate_c[frame]);
        std::cerr << "-";
      }
    }

    // Transition
    for (frame = 0; frame < time_transition; ++frame) {
      lambda = interp_c[frame];
      mapping = (1 - lambda) * m_curr + lambda * m_next;
      con.render(mapping);
      std::cerr << ">";
    }

    std::cerr << std::endl;
  }

  std::cout << con.frames_rendered() << " frames rendered" << std::endl;
  return 0;
}
