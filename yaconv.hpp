#pragma once

void yaconv(float *Image, float *Filter, float *Output, unsigned C,
            unsigned H, unsigned W, unsigned M, unsigned KH, unsigned FW,
            unsigned SH, unsigned SW, unsigned PH, unsigned PW);
