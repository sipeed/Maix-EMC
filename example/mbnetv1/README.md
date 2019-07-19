# Maix-EMC Convert  MobileNetV1 Model
This example introduces the way to convert TL's mobilenet model to K210's kmodel, and run it on the MaixPy environment.

## Model network modify
`mobilenetv1.py` is modified from tensorlayer's file:

```
/usr/local/lib/python3.6/dist-packages/tensorlayer/models/mobilenetv1.py
```
we modified two things:
1. TF's default padding is right-bottom, K210 only support  zero around padding, 
so we need manually add ZeroPad2d Layer before conv layer which stride=(2,2)
**Note:** after this adjustment, we should retrain the network, while I tested the result, orignal weights data still have same result of top5, and top1 probability changed less than 3%, so we skip the retrain step at the moment.
2. original TL's model doesn't support alpha param(default 1.0), we add it on. 
It is because K210 only have ~5MB ram for a model in C environment, ~3.5MB ram in micropython environment.  
1.0 version cost 4.2MB memory, not suit for micropython, so we choose 0.75 version, it takes about 2.7MB.

## Get model  weight file
We use the weights data from **keras**.
Save_keras_weight.py will save keras mobilenet weights into npz file, store in 'params.'
You can choose your alpha value to get the different model size.

|  alpha   |  size(MB)   |  Top-1 Accuracy | Top-5 Accuracy |
| --- | --- | --- | ---|
|  1.0   |  4.24   | 70.9 | 89.9 |
|  0.75   |  2.59   | 68.4 | 88.2 |
|  0.5   |  1.34   | 63.3 | 84.9 |
|  0.25   |  0.47   | 49.8 | 74.2 |

## Convert TL model to K210's Kmodel
First you need put some imagenet pictures into **mbnetv1_dataset** folder, it is used for **quantization**.
Then you just need run mbnetv1_to_kmodel.py to get mobilenet's kmodel.
In this file, you will see the step:
1. build the mobilenet model in TL: 
```
mobilenetv1 = MobileNetV1(pretrained=True, alpha=0.75)
```
2. save the model into kmodel:
```
emc.save_kmodel(mobilenetv1, kmodel_name, './mbnetv1_dataset', dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3, sm_flag=True)
```
API:
```
save_kmodel(network, filepath, dataset_dir, dataset_func='img_0_1', quant_func='minmax', quant_bit=8, version=3, sm_flag=False)
```

|  Parameter   |  Intro |
| --- | --- | 
|network|  TL model network|
|filepath|  output kmodel's filepath|
|dataset_dir|  the path you store pictures for quantization|
|dataset_func| the dataset loader function, convert image to 0~1, or -1~1, or 0~255|
|quant_func|   quantization functions,  choose from minmax, kld|
|quant_bit|    default 8bit|
|version| kmodel version, default 3|
|sm_flag|  add softmax layer at the end.|

## Run kmodel in MaixPy
We have several kinds of k210 boards, called Maix boards, and we port convenient micropython environment called **MaixPy**.
[https://github.com/sipeed/MaixPy](https://github.com/sipeed/MaixPy)
[https://maixpy.sipeed.com/](https://maixpy.sipeed.com/)    

### Burn MaixPy Firmware
MobileNet kmodel cost about 2.7MB RAM, the “full” MaixPy can’t fit it in, we need the minimal version of MaixPy (strip most openmv function and misc functions)
**MaixPy Firmware Download **from(please use >0.3.2 version or master version):
[http://dl.sipeed.com/MAIX/MaixPy/release/](http://dl.sipeed.com/MAIX/MaixPy/release/)
The **firmware download tool**:
[https://github.com/sipeed/kflash\_gui](https://github.com/sipeed/kflash_gui)

### Burn kmodel
And you can burn the **kmodel** you have generate before:  mbnetv1.kfpkg.
kfpkg is the file zip kmodel and flash-list.json, used by kflash_gui.
flash-list.json indicate the kmodel address in a flash, choose position in 0x200000~0xa00000  (2~10MB offset)
We default use 0x200000.

### Add label text
In addition, we need label list to identify number to 1000 class name.
Put labels.txt into MicroSD or Flash's filesystem.

### Run MobileNet on MaixPy!
Open the terminal of Serial port (use minicom or somthing),  press Ctrl+E goto paste mode, and Press Ctrl+D to run the code.
Or use our MaixPy IDE:
[http://dl.sipeed.com/MAIX/MaixPy/ide/](http://dl.sipeed.com/MAIX/MaixPy/ide/)

~~~
import sensor, image, lcd, time
import KPU as kpu
lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))
sensor.set_vflip(1)
sensor.run(1)
lcd.clear()
lcd.draw_string(100,96,"MobileNet Demo")
lcd.draw_string(100,112,"Loading labels...")
f=open('labels.txt','r')
labels=f.readlines()
f.close()
task = kpu.load(0x200000) 
clock = time.clock()
while(True):
    img = sensor.snapshot()
    clock.tick()
    fmap = kpu.forward(task, img)
    fps=clock.fps()
    plist=fmap[:]
    pmax=max(plist)    
    max_index=plist.index(pmax)    
    a = lcd.display(img, oft=(0,0))
    lcd.draw_string(0, 224, "%.2f:%s                            "%(pmax, labels[max_index].strip()))
    print(fps)
a = kpu.deinit(task)

~~~

![image](https://bbs.sipeed.com/uploads/default/optimized/1X/a9329fc053909faca7f34d2500f4f5ec2d576e50_2_668x500.jpeg)  


Video:
[https://www.bilibili.com/video/av46664014](https://www.bilibili.com/video/av46664014)

We can see it identify husky picture correctly~  
And we can see fps in the serial terminal is about 26fps.  
You can make it faster by boost CPU and KPU freq.  
