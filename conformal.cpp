#include <iomanip>
#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

typedef Point2f Pixel;

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

  Mat pixmap;
  pixmap.create(src.size(), CV_32FC2);
  Mat dst;
  dst.create(src.size(), src.type());

  std::vector<int> compression_params = {IMWRITE_PNG_COMPRESSION, 9};

  for (int i = 0; i < 10; ++i) {
    // Compute conformal mapping
    pixmap.forEach<Pixel>(PowerMap(2.0 + i / 10.0, pixmap.rows, pixmap.cols));

    // Apply mapping to src into dst
    remap(src, dst, pixmap,
        noArray(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));

    // Write dst
    std::ostringstream framename;
    framename << "out"
      << std::setfill('0') << std::setw(4) << i << ".png";
    imwrite(framename.str(), dst, compression_params);
  }
  return 0;
}
