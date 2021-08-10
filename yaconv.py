import numpy as np

MR = NR = 3
KC, NC = 9, 9

H, W, KH, KW = 5, 5, 3, 3
OH, OW = H - KH + 1, W - KW + 1

def im2col(I):
    im2col_buffer = []
    for oh in range(OH):
        for ow in range(OW):
            patch = []
            for kh in range(KH):
                for kw in range(KW):
                    Idx = (oh + kh) * W + ow + kw
                    im2col_buffer.append(I[Idx])
    return np.array(im2col_buffer)

def pack_im2col_buffer_as_B(block):
    LDB = KH * KW
    pack = np.zeros(KC * NC)
    for jc in range(0, NC, NR):
        for k in range(KC):
            for jr in range(NR):
                From = jc + jr + k * LDB
                To = jc * KC + k * NR + jr
                pack[To] = block[From]
    return pack

def pack_image_as_B(I):
    pack = np.zeros(KC * NC)
    for jc in range(0, NC, NR):
        for k in range(KC):
            for jr in range(NR):
                To = k * NR + jc * KC + jr
                From =
                print(From, '->', To)


I = np.arange(1, H * W + 1)
print(I)
print(I.reshape(H, W), end='\n' * 5)

# Method 1: image -> im2col_buffer -> im2col buffer packed for gemm as matrix B
im2col_buffer = im2col(I)
print(im2col_buffer)
print(im2col_buffer.reshape(OH * OW, -1), end='\n' * 5)
pack1 = pack_im2col_buffer_as_B(im2col_buffer)
print(pack1)
print(pack1.reshape(-1, NR), end='\n' * 5)

# Method 2: image          ->         im2col buffer packed for gemm as matrix B
# pack2 = pack_image_as_B(I)
# print(pack2.reshape(-1, 4))
