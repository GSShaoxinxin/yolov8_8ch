'''
图像配准我选用了相位相关法
'''
import os
import shutil

import cv2
import numpy as np
import tifffile
def read_image_1ch(image_path):#兼容JPG和TIF格式，返回一通道灰度图
    image=None
    if image_path.endswith(".JPG"):
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    elif image_path.endswith(".TIF"):
        image = tifffile.imread(image_path)
    return image

def fft_oneband_to_jpg(jpg_path, band_index):
    filename = os.path.split(jpg_path)[1]
    # if outdir:
    #     jpg_out_path = os.path.join(outdir, filename)
    image1 = read_image_1ch(jpg_path)
    image2 = tifffile.imread(jpg_path[:-5]+str(band_index)+".TIF")

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # 计算傅里叶变换
    fft1 = np.fft.fft2(image1)
    fft2 = np.fft.fft2(image2)

    # 计算频率域的相位相关
    pro = fft1 * np.conj(fft2)
    cps = pro / (np.abs(pro) + 1e-8)
    cps_shift = np.fft.ifftshift(cps)

    cps_ifft = np.fft.ifft2(cps_shift)
    cps_ifft_abs = np.abs(cps_ifft)
    # 找到最大值位置
    y, x = np.unravel_index(np.argmax(cps_ifft_abs), cps_ifft_abs.shape)

    # 计算平移量
    height, width = image1.shape
    shift_x = x if x <= width / 2 else x - width
    shift_y = y if y <= height / 2 else y - height
    #
    # print("Shift in X:", shift_x)
    # print("Shift in Y:", shift_y)
    M = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    # # 对第二幅图像进行平移
    # if outdir:
    #
    #     result = cv2.warpAffine(image2_bank, M, (image2.shape[1], image2.shape[0]),
    #                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #     print("filename:", jpg_out_path)
    #     tifffile.imwrite(jpg_out_path[:-5] + str(band_index) + ".TIF", result)
    return M

