# 1.项目
基于ChatGLM和PaddleSpeech实现语音对话机器人。


# 2.环境准备

本项目的环境是基于nvidia显卡构建的。

## 2.1 安装anconda
下载安装anconda环境。

创建python 3.9的虚拟环境venv。ChatGLM的python环境要求是3.7版本以上，PaddleSpeech的python环境要求是3.7以上，但不要超过3.9，所以虚拟环境选择python3.9版本。
```
conda create --name cp python=3.9
```

使用anaconda中的powershell prompt输入下面的命令。

激活cp这个虚拟环境：
```
conda activate cp
```

## 2.2 结合CUDA的版本选择可以安装的pytorch的版本

安装gpu版本的pyTorch需要CUDA支持，安装CUDA的内容请查看

### 2.2.1 查看NVIDIA先看的CUDA版本
打开NVIDIA的控制面板，在系统信息中查看，是否有NVCUDA64.DLL文件，该文件是CUDA的驱动。

注意驱动的版本号，如果版本号为11.4.177，后面安装CUDA开发工具时，不能高于这个版本。

### 2.2.2 注册NVIDIA开发者账号

网站地址：
https://developer.nvidia.cn/zh-cn

### 2.2.3 下载并安装CUDA Toolkit
下载地址：
https://developer.nvidia.cn/cuda-downloads

如果显卡的驱动为11.8.x，那么CUDA Toolkit的版本不能高于11.8。

CUDA Toolkit 11.8的下载地址：
https://developer.download.nvidia.cn/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe

默认的安装位置为
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```

### 2.2.4 下载并安装cuDNN
cuDNN需要和CUDA版本保持一致。
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.8.1/local_installers/11.8/cudnn-windows-x86_64-8.8.1.3_cuda11-archive.zip

下载完成后，解压zip包，将解压的内容覆盖CUDA Toolkit的安装目录
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```
## 2.3 安装GPU版本的PyTorch
结合CUDA的版本选择可以安装的pytorch的版本
```
# CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

# 3.版本的检测
查看驱动的版本
```
 nvidia-smi
```
查看CUDA开发工具的版本
```
nvcc -V
```
查看虚拟环境中的pyTorch是否支持CUDA，进入python命令行：
```
python
```
输入下面的脚本查询当前虚拟环境的pyTorch是否支持CUDA。
```
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.__version__)
```

# 4.安装ChatGLM
下载chatglm代码库

```
git clone https://github.com/THUDM/ChatGLM-6B.git
```
在项目文件夹下，执行下面的命令安装依赖包
```
cd ChatGLM-6B
pip install -r .\requirements.txt
```
启动ChatGLM模型

```
 python .\web_demo.py
```
启动后会从hugging face下载模型，一共12G，模型非常大，下载特别耗时。可以网上搜索具体的解决方法。

执行成功后，跳出网页 http://127.0.0.1:7860/ 可以和ChatGLM对话，说明ChatGLM安装成功了。

# 4.安装PaddleSpeech
下载PaddleSpeech代码库

```
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
```

安装PaddlePaddle，在官网查询安装版本，
https://www.paddlepaddle.org.cn/
计算平台CUDA11.7对于飞桨版本2.4

conda的安装方式：
```
conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

在项目文件夹下，执行下面的命令安装pytest-runner

```
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在项目文件夹下，执行下面命令
```
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

文字转语音测试：
```
# 文字转语音输出   
paddlespeech tts --input "今天天气不好，我们暂时就先不约了，等什么时候有时间了，我给你打电话" --output opoint.wav
```
语音生成成功说明安装完成。


# 5.安装项目依赖
在项目文件夹下运行下面的命令，安装python库。
```
pip install -r .\requirements.txt
```
安装完成后，运行下面的命令启动程序：
```
python ./webui.py
```
在页面的文本框中输入文本，页面返回ChatGLM反馈的问题以及PaddleSpeech生成的语音。语音模型使用的是fastspeech2_mix支持中英文混合。

![](https://github.com/zhuchuangang/chat_speech/blob/main/imgs/page.png)


![](https://github.com/zhuchuangang/chat_speech/blob/main/imgs/chat.png)