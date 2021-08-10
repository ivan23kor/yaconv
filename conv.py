from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument('--C', type=int, default=1)
parser.add_argument('--H', type=int, default=6)
parser.add_argument('--W', type=int, default=6)
parser.add_argument('--M', type=int, default=1)
parser.add_argument('--KH', type=int, default=3)
parser.add_argument('--KW', type=int, default=3)
args = parser.parse_args()

for k, v in vars(args).items():
    exec('{} = {}'.format(k, v))
print(C, H, W, M, KH, KW)

OH = H - KH + 1
OW = W - KW + 1
Kernel = np.arange(1, M * C * KH * KW + 1).reshape(M, C, KH, KW)
Input = np.arange(1, C * H * W + 1).reshape(C, H, W)

Output = [[[0 for _ in range(OW)] for _ in range(OH)] for _ in range(M)]
for c in range(C):
    for m in range(M):
        for ho in range(OH):
            for wo in range(OW):
                for kh in range(KH):
                    for kw in range(KW):
                        Output[m][ho][wo] += Input[c][ho + kh][wo + kw] * Kernel[m][c][kh][kw]
print(Input)
print(Kernel)
print(np.array(Output).reshape(M, OH * OW))
