#大疆精灵4多光谱版无人机图像处理指南V1.0 2020.07 --代码实现
（`https://dl.djicdn.com/downloads/p4-multispectral/20200717/P4_Multispectral_Image_Processing_Guide_CHS.pdf`）
##使用指南
本代码将指南中原植被指数的计算过程进行解耦，以更灵活处理数据。以ndvi计算过程为例，原始图像nir,red进行如下步骤：
1. 以JPG真彩色图像为目标，将各光谱灰度图像向真彩色图像平移配准，得到各自的平移量dx,dy备用。
2. 计算`nir_ref`，计算`red_ref`
3. 利用步骤1的dx,dy对齐`nir_ref`,`red_ref`
4. 根据(`nir_ref`-`red_ref`)/(`nir_ref`+`red_ref`)公式计算得到ndvi
5. 提供两种ndvi的可视化方式：一是plt画彩色图、二是映射到0-255保存为灰度图。如果进行后续任务请使用未可视化的真实数据。

##代码细节点
1. 除了官方图像处理指南，还参考资料：https://www.cnblogs.com/ludwig1860/p/14965019.html
2. 本代码中校正包括：黑电平校正, 暗角补偿，畸变校正, 增益校正,感度校正。
3. TIF格式图像中附带丰富的元信息，这些信息包括GPS、飞行器、云台、相机的各种参数信息，可使用pyexiv2库读写。
4. requiremet库：必需库tifffile，pyexiv2



