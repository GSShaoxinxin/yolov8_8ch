import os

import cv2

'''只执行校正功能，保留camera输出结果到uint16,不进行归一化
'''

import os
import shutil

import numpy as np
import tifffile as tiff
import cv2 as cv
import pyexiv2

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import re
import json
from registration import fft_oneband_to_jpg
from enum import Enum
from util import num_to_color, color_to_num, Align

#文件名称中序号代表不同波段
def normalize(im,num=65535):
    return im / num


def r_euclidean_distance(h,w, vcntr,ucntr ):#算各点到焦点距离
    ucntr = float(ucntr)
    vcntr = float(vcntr)
    center = np.array([vcntr,ucntr ])
    distances = np.linalg.norm(np.indices((h,w)) - center[:, None, None] + 0.5, axis=0)
    return distances

def vignetting(w,h, meta):#暗角补偿
    u_cntr = meta['Xmp.drone-dji.CalibratedOpticalCenterX']
    v_cntr = meta['Xmp.drone-dji.CalibratedOpticalCenterY']
    k_list = meta['Xmp.drone-dji.VignettingData'].split(",")
    k_list = [float(item) for item in k_list]
    r_matrix = r_euclidean_distance(h,w,v_cntr, u_cntr)
    #   DN3=DN2*(k5*r^6+k4*r^5+k3*r^4+k2*r^3+k1*r^2+k0*r+1)

    p = np.poly1d([k_list[5], k_list[4], k_list[3], k_list[2], k_list[1], k_list[0], 1])
    r_2 = p(r_matrix)
    # dn1 = np.multiply(dn, r_2)
    return r_2


def DistortionCorrection(img, meta):
    centerX = float(meta['Xmp.drone-dji.CalibratedOpticalCenterX'])
    centerY = float(meta['Xmp.drone-dji.CalibratedOpticalCenterY'])
    ls = re.split(r'[,;]', meta['Xmp.drone-dji.DewarpData'])
    ls = [float(item) for item in ls[1:]]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = ls  # _;fx,fy,cx,cy,k1,k2,p1,p2,k3
    # 相机参数
    # cameraMatrix = [(fx,0,centerX+cx),(0,fy,centerY+cy),(0,0,1)]
    cameraMatrix = np.array([[fx, 0, centerX + cx],
                             [0, fy, centerY + cy],
                             [0, 0, 1]])
    # 畸变参数
    distCoeffs = np.array([k1, k2, p1, p2, k3])


    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))#alpha =1或0 有不同的效果
    img_undistored = cv.undistort(img, cameraMatrix, distCoeffs,None,newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = img_undistored[y:y + h, x:x + w]
    # cv.imwrite('calibresult.png', dst)
    # cv.imwrite('11.png', img_undistored)
    return img_undistored
def camera(im, meta,type=['black','norm','vig','dist','iso','gd']):
    blacklevel = 4096
    gain = float(meta['Xmp.drone-dji.SensorGain'])
    epotime = float(meta['Xmp.drone-dji.ExposureTime']) / 1e6  # 化到以秒为单位
    # 黑电平校正
    #show_what(im,path="raw.jpg",title="raw")
    if 'black' in type:
        im = im - blacklevel

    # if im_path:
    #     #show_what(im,path=im_path[:-4]+"blacklevel.png",title="blacklevel")
    #     tiff.imwrite(im_path[:-4]+"_6blackDegree.TIF", im.astype(np.uint16))

    #show_what(im,path="norm.jpg",title="normalize")
    # 暗角校正
    if 'vig' in type:
        vig = vignetting(im.shape[1],im.shape[0], meta)#vig矩阵与原图无关
        # plt.imshow(vig)
        # plt.show()
        # cv.imwrite("vig.jpg", (vig*150).astype(np.uint8))
        im = np.multiply(im,vig)
        # tiff.imwrite(im_path[:-4]+"_7vig.TIF",im.astype(np.uint16))
    # 畸变校正
    if 'dist' in type:
        im = DistortionCorrection(im, meta)
        # tiff.imwrite(im_path[:-4]+"_8jibian1.TIF", im.astype(np.uint16))
    #show_what(im,path="blackDegree.jpg",title="DistortionCorrection")
    if 'norm' in type:#把归一化操作挪到暗角校正和畸变校正之后，没影响
        im = normalize(im)
    if 'iso' in type:#涉及到相机曝光,
        im = im / (gain * epotime)
    if 'gd' in type:#多光谱传感器的感光性能一致，即感度
        pcam = float(meta['Xmp.drone-dji.SensorGainAdjustment'])
        irradiance = float(meta['Xmp.Camera.Irradiance'])
        t = (pcam / irradiance)#t在万分之一左右0.0001
        # p_nir = #nir与其他感光的系数，被约掉了
        # im = im * (t*p_nir)
        im = im * (t)
        # print("pcam / irradiance:{}".format(t))
    # tiff.imwrite(im_path[:-4]+"_9gainepoch.TIF", im.astype(np.uint8))
    #show_what(im, path="gain.jpg",title="gain")
    print("carama:{},{}".format(np.min(im),np.max(im)))
    return im

def ref_one_group(tif_dir, jpgfilename,out_dir,args):
    jpg_path = os.path.join(tif_dir, jpgfilename)
    out_jpg_path = os.path.join(out_dir, jpgfilename)
    if args.get("is_dist_jpg"):#是否对JPG进行畸变校正
        if "dist" in args.get("type"):
            with pyexiv2.Image(jpg_path) as jpg_xmp:  # 使用with就会自动关闭
                jpg_xmp = jpg_xmp.read_xmp()
            jpg = cv.imread(jpg_path)
            newjpg = DistortionCorrection(jpg,jpg_xmp)
            cv.imwrite(out_jpg_path,newjpg)
            with pyexiv2.Image(out_jpg_path) as newjpg_xmp:#将元信息原封不动写入保留
                newjpg_xmp.modify_xmp(jpg_xmp)
    else:
        shutil.copy(jpg_path,out_jpg_path)
    ref_lists= {}#校正后数据存放在该数组
    xmp_lists= {}#源tif图像的元信息存留

    # for i in range(3):
    for i in [1,2,3,4,5]:#可为0,1,2
    # for i in [0,1,2]:#可为0,1,2
        # print("i:{}".format(i))
        filename =  jpgfilename[:-5] + str(i)+".TIF"
        p = os.path.join(tif_dir, filename)
        gray = tiff.imread(p)#读取某一波段的灰度图像
        with pyexiv2.Image(p) as img_xmp:  # 读取元信息 使用with就会自动关闭
            xmp = img_xmp.read_xmp()
        new_gray = camera(gray, xmp,  type=args.get("type"))#进行校正

        ref_lists[num_to_color(i)]=new_gray
        xmp_lists[num_to_color(i)]=xmp

    return ref_lists,xmp_lists
        # out_p = os.path.join(out_dir,filename)
        # tiff.imwrite(out_p,new_gray.astype(np.uint16))
        # if args.get("write_xmp"):
        #     with pyexiv2.Image(out_p) as newimg:
        #         newimg.modify_xmp(xmp)
def cmpt_registration_one_group(tif_dir, jpgfilename):
    M_lists= {}#平移变换矩阵存放
    for i in [1,2,3,4,5]:#可为0,1,2
        M = fft_oneband_to_jpg(os.path.join(tif_dir, jpgfilename),i)
        M_lists[num_to_color(i)]=M
    return M_lists

def main(tif_dir,out_dir,args):
    filenames = [item for item in os.listdir(tif_dir)  if (item.endswith('.JPG'))]#筛选所有JPG文件
    for filename in filenames:
        # 将各光谱图均向JPG彩色图对齐。用原图进行图像配准，计算平移量备用
        M_lists = cmpt_registration_one_group(tif_dir, filename)#
        #辐射校正
        ref_lists,xmp_lists = ref_one_group(tif_dir, filename,out_dir,args)
        # 平移多光谱图像，一致向JPG图像对齐
        blue_ref = ref_lists.get(num_to_color(1))
        blue_ref = Align(blue_ref,M_lists.get(num_to_color(1)))

        green_ref = ref_lists.get(num_to_color(2))
        green_ref = Align(green_ref,M_lists.get(num_to_color(2)))

        red_ref = ref_lists.get(num_to_color(3))
        red_ref = Align(red_ref,M_lists.get(num_to_color(3)))

        rededge_ref = ref_lists.get(num_to_color(4))
        rededge_ref = Align(rededge_ref,M_lists.get(num_to_color(4)))

        nir_ref = ref_lists.get(num_to_color(5))
        nir_ref = Align(nir_ref,M_lists.get(num_to_color(5)))
        #计算植被指数
        result_vi = eval(args.get("formula"))
        out_path= os.path.join(out_dir, filename[:-4] + "_vi.JPG")

        ############ 第一种可视化指数结果
        plt.imshow(result_vi,cmap='RdYlGn')
        plt.colorbar()
        f = plt.gcf()  # 获取当前图像
        f.savefig(out_path)
        f.clear()  # 释放内存
        # plt.show()

        # ############## 第二种根据vi的数值范围映射到0-255之间，保存为灰度图像，但这种粗略的可视化会丢失一定有效信息，尽量统计一下数据中集中的数据范围
        # # 例如ndvi一般分布在-1-1，如下归一化到 [0, 255]
        # normalized = ((result_vi + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        # # 保存图像
        # cv2.imwrite(out_path, normalized)

        print("处理完：{}".format(filename))

if __name__ == '__main__':

    args ={
        "tif_dir" : "./src_images",#源图像所在文件夹
        "out_dir" : "./out_images",
        "formula" :"(nir_ref-red_ref)/(nir_ref+red_ref)",#请严格以blue_ref,green_ref,red_ref,rededge_ref和nir_ref五个变量构造计算公式。
        #################################
        # "is_vis_ref":"./vis_refimages",#是否可视化校正后的ref

        # 执行校正步骤，包括'black'：暗角校正, 'vig'：暗角补偿，'dist'：畸变校正, 'iso'：增益校正, 'gd'：感度校正,
        # "type": ['black','norm','vig','dist','iso','gd'], #执行的校正步骤
        "type": ['black','norm','vig','iso','gd'], #畸变校正对图像外观改变较大，在允许一定位置差异的情况下，我觉得可省略，
        "is_dist_jpg": False,  # 是否同时将JPG进行畸变校正
    }
    if not os.path.exists(args.get("out_dir")):
        os.mkdir(args.get("out_dir"))

    main(args.get("tif_dir"),args.get("out_dir"),args)



