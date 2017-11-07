# conformal

Visualization of conformal mappings on the complex plane.

![Sample output](sample.gif)

Inspired by [this video by /u/ttoinou](https://www.youtube.com/watch?v=CMMrEDIFPZY).

Cat photo by Paul on Unsplash.

## Usage

```
$ ./conformal input.png
$ mpv --loop out.mp4
```

## Requirements

-   OpenCV (tested with v3.3.0)
-   Armadillo (tested with v8.200.1)
    -   LAPACK (tested with v3.6.0)
    -   OpenBLAS (tested with v0.2.18) or other BLAS equivalent

## Building

From the repository root:

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
