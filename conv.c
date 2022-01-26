#if defined(IM2COL_OPENBLAS) || defined(YACONV_OPENBLAS)
#include "cblas.h"
#else
#error "No valid convolution algorithm defined"
#endif
#include <stdlib.h>
#include <time.h>

#define START_TIMER()	clock_gettime(CLOCK_REALTIME, &timer_start);
#define STOP_TIMER(COUNTER) \
  clock_gettime(CLOCK_REALTIME, &timer_stop); \
  COUNTER += (timer_stop.tv_sec - timer_start.tv_sec) * 1e9; \
  COUNTER += timer_stop.tv_nsec - timer_start.tv_nsec;

float *alloc_random(int size);

void im2col_conv(float **images, int N, int H, int W, int C,
                 float *filter, int FH, int FW, int M,
                 float **outputs, int PH, int PW);

int main(int argc, char** argv) {
  // Init seed for array randomization
  srand(time(NULL));

  // Usage message
  if (argc < 10) {
    fprintf(stderr, "Usage: ./test_conv N H W C FH FW M PH PW\n");
    return -1;
  }

  // Get convolution parameters from CLI, compute OH and OW
  const int N = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int W = atoi(argv[3]);
  const int C = atoi(argv[4]);
  const int FH = atoi(argv[5]);
  const int FW = atoi(argv[6]);
  const int M = atoi(argv[7]);
  const int PH = atoi(argv[8]);
  const int PW = atoi(argv[9]);
  const int OH = H + 2 * PH - FH + 1;
  const int OW = W + 2 * PW - FW + 1;

  // Filter array
  float *filter = alloc_random(FH * FW * C * M);

  // Input arrays
  float **images = (float **)malloc(N * sizeof(float *));
  for (int i = 0; i < N; ++i)
    images[i] = alloc_random(H * W * C);

  // Output arrays
  float **outputs = (float **)malloc(N * sizeof(float *));
  int output_size = OH * OW * M;
#ifdef YACONV_OPENBLAS
  output_size += yaconv_extra_size(H, FH, PH, OW, M);
#endif
  for (int i = 0; i < N; ++i)
    outputs[i] = alloc_random(output_size);

  // Time variables
  struct timespec timer_start, timer_stop;
  double nap = 0.0;
  double total = 0.0;

  // Punch in
  START_TIMER();

  // Compute convolution
#if defined(IM2COL_OPENBLAS)
  im2col_conv(images, N, H, W, C, filter, FH, FW, M, outputs, PH, PW);
#elif defined(YACONV_OPENBLAS)
  yaconv(images, N, H, W, C, filter, FH, FW, M, outputs, PH, PW);
#endif

  // Punch out
  STOP_TIMER(total);

  // Print GFLOPS
  printf("%5.2f\n", 2.0 * N * M * FH * FW * C * OH * OW / total);

  // Free arrays
  free(filter);
  for (int i = 0; i < N; ++i) {
    free(images[i]);
    free(outputs[i]);
  }

  return 0;
}

float *alloc_random(int size) {

  // Try to allocate
  float *data = (float*)malloc(size * sizeof(float));

  // Handle potential malloc error
  if (data == NULL)
  {
    fprintf(stderr, "Some error in malloc!\n");
    exit(-1);
  }

  for (int i = 0; i < size; ++i)
    // data[i] = rand() % 256; // fill with random numbers from [0, 255]
    data[i] = i + 1; // fill with a sequence 1, 2, 3, ...

  return data;
}

// The following two functions are taken from
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp#L14-L55
// static_cast in is_a_ge_zero_and_a_lt_b was changed to C-style cast
#define is_a_ge_zero_and_a_lt_b(a, b) ((unsigned)a < (unsigned)b)

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            float *data_col) {

  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

void im2col_conv(float **images, int N, int H, int W, int C,
                 float *filter, int FH, int FW, int M,
                 float **outputs, int PH, int PW) {

  const int OH = H + 2 * PH - FH + 1;
  const int OW = W + 2 * PW - FW + 1;
  float *im2col_buf = alloc_random(OH * OW * FH * FW * C);
  float alpha = 1.0, beta = 0.0;

  for (int i = 0; i < N; ++i) {
    // im2col
    im2col(images[i], C, H, W, FH, FW, PH, PW, 1, 1, 1, 1, im2col_buf);

    // GEMM
    cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, M, OH * OW, FH * FW * C,
       alpha, filter, FH * FW * C, im2col_buf, OH * OW, beta, outputs[i], M);
  }

  free(im2col_buf);
}
