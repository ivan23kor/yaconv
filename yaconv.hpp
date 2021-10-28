#pragma once

void yaconv(float *Image, float *Filter, float *Output, int C,
            int H, int W, int M, int KH, int FW,
            int SH, int SW, int PH, int PW);
