// Input: H x W
// L: OW x H x KW
float *mec(float *Input, unsigned C, unsigned H, unsigned W, unsigned KH,
           unsigned KW, unsigned PadH, unsigned PadW, unsigned StrideH,
           unsigned StrideW, unsigned DilH, unsigned DilW) {

  const unsigned OH = (H - KH + 2 * PadH) / StrideH + 1;
  const unsigned OW = (W - KW + 2 * PadW) / StrideW + 1;

  auto *L = new float[OW * H * KW];

  for (unsigned w = 0; w < OW; ++w) {
    for (unsigned h = 0; h < H; ++h) {
      unsigned LIdx = (w * H + h) * KW;
      unsigned IIdx = h * W + w * StrideW;
      // L[w, h, 0:kw] = I[h, w * StrideW: w * StrideW + KW];
      for (unsigned kw = 0; kw < KW; ++kw)
        L[LIdx + kw] = Input[IIdx + kw];
    }
  }

  return L;
}
