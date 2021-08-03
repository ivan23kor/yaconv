#ifndef __MEC_H_
#define __MEC_H_

float *mec(float *Input, unsigned C, unsigned H, unsigned W, unsigned KH,
           unsigned KW, unsigned PadH, unsigned PadW, unsigned StrideH,
           unsigned StrideW, unsigned DilH, unsigned DilW);

#endif // __MEC_H_
