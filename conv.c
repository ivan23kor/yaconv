#if !defined(IM2COL) && !defined(YACONV)
#error "No convolution algorithm defined! [IM2COL/YACONV]"
#endif

#ifdef OPENBLAS
#include "cblas.h"
#elif defined(BLIS)
#include "blis/blis.h"
#else
#error "No BLAS defined! [OPENBLAS/BLIS]"
#endif

#include <stdlib.h>
#include <time.h>

#define START_TIMER()	clock_gettime(CLOCK_REALTIME, &timer_start);
#define STOP_TIMER(COUNTER) \
  clock_gettime(CLOCK_REALTIME, &timer_stop); \
  COUNTER += (timer_stop.tv_sec - timer_start.tv_sec) * 1e9; \
  COUNTER += timer_stop.tv_nsec - timer_start.tv_nsec;

float *alloc_and_init(int size);

void im2col_conv(float **images, int N, int H, int W, int C,
                 float *filter, int FH, int FW, int M,
                 float **outputs, int PH, int PW);

void print_array(float *array, int size);

void tranform_for_im2col(float *filter, float **images, int N,
                         int H, int W, int C, int FH, int FW, int M);

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
  float *filter = alloc_and_init(FH * FW * C * M);

  // Input arrays
  float **images = (float **)malloc(N * sizeof(float *));
  for (int i = 0; i < N; ++i)
    images[i] = alloc_and_init(H * W * C);

  // Output arrays
  float **outputs = (float **)malloc(N * sizeof(float *));
  int output_size = OH * OW * M;
#ifdef YACONV
  output_size += yaconv_extra_size(H, FH, PH, OW, M);
#endif
  for (int i = 0; i < N; ++i)
    outputs[i] = alloc_and_init(output_size);

  // Time variables
  struct timespec timer_start, timer_stop;
  double total = 0.0;

  // Punch in
  START_TIMER();

  // Compute convolution
#if defined(IM2COL)
  im2col_conv(images, N, H, W, C, filter, FH, FW, M, outputs, PH, PW);
#elif defined(YACONV)
  yaconv(images, N, H, W, C, filter, FH, FW, M, outputs, PH, PW);
#endif

  // Punch out
  STOP_TIMER(total);

  // Print GFLOPS
  printf("%5.2f\n", 2.0 * N * M * FH * FW * C * OH * OW / total);

  // For yaconv, compare with reference implementation (im2col)
#if defined(YACONV) && defined(CHECK)
  float **outputs_ref = (float **)malloc(N * sizeof(float *));
  for (int i = 0; i < N; ++i)
    outputs_ref[i] = alloc_and_init(OH * OW * M);

  // For im2col, change image layout HWC -> CHW and filter layout HWCM -> MCHW
  tranform_for_im2col(filter, images, N, H, W, C, FH, FW, M);
  im2col_conv(images, N, H, W, C, filter, FH, FW, M, outputs_ref, PH, PW);

#define MAX(a,b) (a > b ? a : b)
#define ABS(a) (a > 0 ? a : -a)
  float max_rel_diff = 0.0;
  int yaconv_before_off = yaconv_extra_size_before(FH, PH, OW, M);
  for (int i = 0; i < N; ++i) {
    float *output = outputs[i] + yaconv_before_off;
    for (int j = 0; j < OH * OW * M; ++j)
      max_rel_diff = MAX(max_rel_diff,
                         ABS(outputs_ref[i][j] - output[j]) / output[j]);
  }
  // printf("%f\n", max_rel_diff);
#endif

  // Free arrays
  free(filter);
  for (int i = 0; i < N; ++i) {
    free(images[i]);
    free(outputs[i]);
  }

#if defined(YACONV) && defined(CHECK)
  return max_rel_diff > 1e-3;
#else
  return 0;
#endif
}

float *alloc_and_init(int size) {

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
    // data[i] = 1; // fill with a constant 1

  return data;
}

void print_array(float *array, int size) {
  for (int i = 0; i < size; ++i)
    printf("%.0f ", array[i]);
  printf("\n");
}

void tranform_for_im2col(float *filter, float **images, int N,
                         int H, int W, int C, int FH, int FW, int M) {
  float *filter_hwcm = alloc_and_init(FH * FW * C * M);
  for (int fh = 0; fh < FH; ++fh)
    for (int fw = 0; fw < FW; ++fw)
      for (int c = 0; c < C; ++c)
        for (int m = 0; m < M; ++m)
          *(filter + m * C * FH * FW + c * FH * FW + fh * FW + fw) =
              *(filter_hwcm + fh * FW * C * M + fw * C * M + c * M + m);

  float *image_hwc = alloc_and_init(H * W * C);
  for (int i = 0; i < N; ++i)
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        for (int c = 0; c < C; ++c)
          *(images[i] + c * H * W + h * W + w) =
              *(image_hwc + h * W * C + w * C + c);

  free(filter_hwcm);
  free(image_hwc);
}

// The following two functions are taken from
// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp#L14-L55
// static_cast in is_a_ge_zero_and_a_lt_b was changed to C-style cast
#define is_a_ge_zero_and_a_lt_b(a, b) ((unsigned)a < (unsigned)b)

// This im2col works on NCHW->NMHW format, while yaconv works on NHWC->NHWM
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
  float *im2col_buf = alloc_and_init(OH * OW * FH * FW * C);
  float alpha = 1.0, beta = 0.0;

  for (int i = 0; i < N; ++i) {
    // im2col
    im2col(images[i], C, H, W, FH, FW, PH, PW, 1, 1, 1, 1, im2col_buf);

    // GEMM
    int m = M, k = FH * FW * C, n = OH * OW;
#ifdef OPENBLAS
    cblas_sgemm(CblasColMajor, CblasTrans, CblasTrans, m, n, k,
                alpha, filter, k,
                im2col_buf, n,
                beta, outputs[i], m);
#elif defined(BLIS)
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k,
              &alpha, filter, k, 1,
              im2col_buf, n, 1,
	            &beta, outputs[i], 1, m);
#endif
  }

  free(im2col_buf);
}
