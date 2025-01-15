## <div align="center">READ-detect-基于YOLOv5-master(我也不知道这是哪个版本...)</div>
首先，我证明了收集的图片少没啥用。。。最后都没什么结果还是要多点图片才有用。。。下次改进吧。

### 1.首先是按照群文件的指引，先弄一个虚拟环境，然后通过命令激活它。
```bash
conda activate yolov5
```

### 2.继续按照群文件的指引，弄好自己的数据集，把这个数据集放在train.py所在文件夹下，数据集下属有两个分支，分别是images和labels，两个分支下面又各有两个同名分支train和val。顾名思义，images下放图，labels下放文本，train下放用来训练的，val下放要用来验证的（我废话有点多sorry）
我的话是训练了四个标签（dog,cat,rabbit,hamster）但是总归只找了二十多张图片，所以准确率惨不忍睹。。。所以最后验证那个结果也只能说没有办法。不过除了耗时久之外这个还是挺有意思。
我是在data下创建了一个train-.yaml的代码
训练的命令行是：
```bash
python train.py --data data/train-1.yaml
```
<div>最后是存储在了..\yolov5-master1\runs\train\exp2下面<div>

### 3.最后是推理过程
命令行如下
```bash
python detect.py --weights runs/train/exp2/weights/best.pt
```
我在data下放了一个test文件夹，里面是测试图集，然后改了一下detect.py里的source，所以就没加后缀了
然后最后就是得到那个结果，怎么说不出所料。。。怎么都识别不出猫，连仓鼠都被认成狗了。。。没话讲，惨痛教训TAT
最后输出是在runs\detect\exp2


## <div align="center">READ-detect-1</div>
这个代码大部分是参照第七次培训的示例代码做出来的，然后结合了detect.py原本的一些内容，终于把这个东西做出来了。因为我倾向于只显示识别的对象及外框和中心点，所以没有保留课上提到的帧率部分，最后是在看课和查找detect.py的手忙脚乱的修改中结束那堆让我抓狂的报错。
最终运行如下代码
```bash
python detect-1.py
```
## <div align="center">训练过程</div>
### 1. 复制原本的detect.py代码到新建的detect-1.py文件中

### 2.目的是在仅保留detect.py从相机中获取视频流能力的同时实现detect.py的类和对象的封装

### 3.好吧其实没有什么可以写的步骤，我只是一直在抓狂罢了哈哈哈哈哈，首先是从教学视频里一点点把代码抠出来，然后因为需求不完全一样，导致一开始疯狂报错，搞得我焦头烂额，求助了KIMI和CHATGPT，但是好笑的是，最后拯救我的居然是原本的detect.py文件，我靠查找硬查出来了缺失的数据。
## <div align="left">最深的感受——常回头看看啊！！！