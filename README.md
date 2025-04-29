# YOLOv8接收多通道数据进行训练
本文介绍实现向YOLOv8输入多通道数据（>3）进行训练的修改过程。<br>
阅读须知：<br>
本文修改过程可对照代码进行阅读<br>
本文的前置条件是你已了解并复现了yolov8在常规三通道RGB图像数据集上的训练<br>
本文基于yolov8进行修改，修改处均注释以`hcy`,可在项目内全局搜索进行快捷查看<br>
本文只是多通道方法的一种实现，还有一种方法是通过多个dataloader实现，这种实现方法不会牵扯到数据增强过程的修改，修改处可能更少，也是一种有效的方法<br>
如果直接使用本代码，只需要关注数据机构建和代码修改中的数据加载部分。

## 一、构建数据集：
本文多通道数据以大疆精灵4多光谱无人机采集到的可见光和多光谱图像数据为例,该无人机同时获取一张彩色图和5张单波段灰度图，名称是DJI_xxxx0.JPG,DJI_xxxx1.TIF,DJI_xxxx2.TIF,DJI_xxxx3.TIF,DJI_xxxx4.TIF,DJI_xxxx5.TIF,文件名中的后缀0.JPG、1.TIF、2.TIF、3.TIF、4.TIF和5.TIF分别代表RGB彩色图、蓝波段、绿波段、红波段、红边波段和近红外波段图像。因此，本文设计将多光谱TIF图像和JPG图像放在同一文件夹下（即yolo数据集的images文件夹下），根据JPG图像径得到对应的不同波段光谱图像。<br>
Warning：该无人机获得的多光谱图像有两个特点：一是多光谱图像存在空间不匹配问题，二是彩色JPG图像的值在0-255，多光谱TIF图像的值在0-65535，分布不同，需要做额外的处理，本文在此不进行介绍，以已经对齐的、值统一在0-255的图像数据为基础进行后续步骤<br>
本文设计多通道图像的加载逻辑是：以RGB图像为依据，通过文件名的对应直接找到其他光谱图像加载到模型中。<br>
### 1.数据集参考datasets_my_multi文件夹
```
├── `datasets_my_multi` 
│   ├── images
│   │   ├── DJI_xxxx10.JPG
│   │   ├── DJI_xxxx11.TIF
│   │   ├── DJI_xxxx12.TIF
│   │   ├── DJI_xxxx13.TIF
│   │   ├── DJI_xxxx14.TIF
│   │   ├── DJI_xxxx15.TIF
│   │   ├── DJI_xxxx20.JPG
│   │   ├── DJI_xxxx21.TIF
│   │   ├── DJI_xxxx22.TIF
│   │   ├── DJI_xxxx23.TIF
│   │   ├── DJI_xxxx24.TIF
│   │   ├── DJI_xxxx25.TIF
│   │   └──......
│   ├── labels
│   │   ├── DJI_xxxx10.txt
│   │   ├── DJI_xxxx20.txt
│   │   └──......
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
```
### 2.在ultralytics/cfg/datasets下创建数据集配置文件`datasets_my_multi.yaml`
关键点是加上ch:8
## 二、代码修改

### 1.创建my_train.py，该文件实现对yolov8.yaml的加载、训练和评估主函数，后续过程会以运行该文件作为测试。
### 2.向ultralytics/cfg/default.yaml，ultralytics/cfg/models/v8/yolov8.yaml两文件中均添加`ch: 8`参数
修改至此，运行`my_train.py`会出现的报错是：“期望8通道数据，但实际只有3通道”，因此下面我们进入数据集加载过程中的修改。
### 3.修改数据集加载过程中读取图像的代码，保证获取到多通道图像数据。
这部分内容根据自己的数据特点可以有不同实现方法。在原yolov8中，ultralytics/data/base.py文件`load_image`函数中用代码`im = cv2.imread(f) `读取到h×w×3的RGB彩色图像，而我们现在期望增加5张多光谱图像，读取到h×w×8的数据。因此我自定义getMultiImages函数根据彩色图的路径得到五张多光谱图像，并依次读取并堆叠成h×w×5数据，再和彩色图堆叠起来，最终实现读取了h×w×8数据。<br>
```
				im = cv2.imread(f)  # BGR
                ####hcy 增加代码
                mx = getMultiImages(f)  # 得到array
                im = np.concatenate((im, mx), axis=2)
                #####
```

修改至此，运行`my_train.py`会出现的报错是：“Invalid number of channels in input image:...”，原因是数据增强步骤的opencv函数都只接受3通道或者1通道的数据，8通道的数据输入不符合函数定义，下面对数据增强部分进行修改。
### 4.修改数据增强不能处理8通道数据的问题。
以下处理顺序是根据运行`my_train.py`出现错误的顺序进行，修改完一条后，运行`my_train.py`进行后一个修改处。<br>
（1）ultralytics/data/augment.py 文件def v8_transforms定义中注释掉RandomHSV。8通道数据不能转换到hsv颜色空间了，因此放弃使用这种增强方法。<br>
（2）ultralytics/utils/plotting.py文件def plot_images函数中进行mosaic增强时，原先创建新图像是3通道，修改为创建新图根据输入图的通道而定。<br>
```
    #hcy 代码修改
    # mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    mosaic = np.full((int(ns * h), int(ns * w), images.shape[1]), 255, dtype=np.uint8)  # init
```
（3）ultralytics/data/augment.py 中class LetterBox的cv2.copyMakeBorder使用出错，该方法只能处理3通道活1通道数据，因此更改为循环8次，每次处理1通道。<br>
```
        # hcy 修改代码
        # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=(114, 114, 114))  # add border
        tmp_img = []
        for i in range(img.shape[2]):
            tmp = np.expand_dims(cv2.copyMakeBorder(img[:, :, i], top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                    value=(114)), 2)  # add border
            if i == 0:
                tmp_img = tmp
            else:
                tmp_img = np.concatenate((tmp_img, tmp), axis=2)
        img = tmp_img
        # ↑↑↑↑↑↑↑↑↑↑
```
（4）warmup<br>
修改至此，运行，得到报错：“Given groups=1, weight of size [16, 8, 3, 3], expected input[1, 3, 640, 640] to have 8 channels, but got 3 channels instead”，原因是ultralytics/engine/validator.py文件中def `__call__`函数中warmup用了3通道，下面修改warmup时的数据根据ch通道变化.
```
            # hcy 修改代码
            # model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup
            for i in range(1000):
                model.warmup(imgsz=(1 if pt else self.args.batch, self.args.ch, imgsz, imgsz))  # warmup
```
(5)画图中修改<br>
修改至此，运行后报错：“TypeError: Cannot handle this data type: (1, 1, 8), |u1”，原因是ultralytics\utils\plotting.py"中 class Annotator使用的PIL库不能处理8通道。该类处理了一下将标注框画到图片进行输出`val_batch0_pred.jpg`等的，表示了一些关于数据集的信息，因此在其init函数中直接使用3通道即可，不会影响网络的输入。
```
  def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        # hcy 增加代码
        im = im[:, :, :3].copy()
```
至此，`my_train.py`就跑通了，train和val都可以正常运行了
### 5.推理过程的8通道实现<br>
（1）创建`my_pred.py`,其中一个参数是模型pt文件路径，一个是txt文件，里面包含了推理的文件名称，这里我们用数据集中test.txt
```
if __name__ == '__main__':
    model = YOLO(r""D:\xxxx\v8_multi_channel\runs\detect\train4\weights\best.pt")
    path = r"D:\Shao\Files\appdata\PycharmProject\v8\v8_ch\datasets_my_multi\test.txt"
    model.predict(source=path, ch=8, save=True, conf=0.5, save_txt=True, save_conf=True)
```
(2)修改warmup<br>
(a)新建`pt_ch`文件夹,将原先的yolov8n.pt挪到该文件夹下，然后用`修改初始化pt文件通道.py`文件将pt文件的第一层从3通道扩充为8通道的，并命名为`yolov8n_8.pt`,将该文件复制到v8_multi_channel下，和ultralytics同级目录，然后再重命名为`yolov8n.pt`。这里我不想修改代码中pt文件的名称，扩充过程又会用到YOLO模型加载过程，需要原来的yolov8n，比较耦合，所以用这种比较麻烦的方式。<br>
(b)ultralytics/engine/predictor.py文件stream_inference函数中修改代码为8通道<br>
```
            #hcy 修改代码 self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            # 在推理时用了.pt 所以要用8
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs,8, *self.imgsz))
```

（3）修改数据加载<br>
ultralytics/data/loaders.py文件class LoadImages类`__next__`函数
```
 			# Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            #↓↓↓↓↓↓ hcy 添加代码
            mx = getMultiImages(path)  # 得到array
            if len(mx) > 0:
                im0 = np.concatenate((im0, mx), axis=2)
            #↑↑↑↑↑↑↑
```

# 其他注意
yolov8加载图像过程会将读取到的bgr图像在进入网络之前翻转为rgb






