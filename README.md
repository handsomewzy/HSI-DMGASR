# 项目名称
This is the raw source code of the paper 'Enhancing Hyperspectral Images via Diffusion Model and Group-Autoencoder Super-Resolution Network' Our code is based on SR3, SSPSR GELIN
代码主要分为两个阶段，阶段1训练GAE，阶段2联合训练Diffusion model。在子文件AE.py训练GAE，训练完成后，配置好config文件，导入正确的训练集路径后，训练扩散模型，运行sr_gae.py。

## 安装
安装项目所需的依赖库：

```
pip install -r requirements.txt
```

## 使用说明
### 训练GAE
配置好数据集路径信息后，运行如下代码：

```
python AE.py
```

### 训练扩散模型Diffusion model
配置好数据集路径信息，GAE加载预训练模型。（具体其余参数配置在config文件中，位置在：EHSI-DMGESR/config/sr_sr3_16_128.json，直接运行如下代码即可：

```
python sr_gae.py
```
在sr_gae.py中可以更改是训练还是推理。

## 参数具体配置
配置文件config参考格式在：EHSI-DMGESR/config/sr_sr3_16_128.json，数据集的具体读取在对应的 AE.py 与 sr_gae.py 文件中进行配置。
具体的，前后实验中有两种数据处理方式：
1. 使用TrainsetFromFolder函数，参考MCNet中的数据处理方式，使用matlab本地处理数据集后，直接进行读取。
2. 使用HSTrainingData，HSTestData函数，加载数据后线上处理数据，灵活性更强。
具体的细节与使用方法参考两个函数的定义。

## 说明
本项目主要基于SR3，SSPSR，MCNet等代码开发。
