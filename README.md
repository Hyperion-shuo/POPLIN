# POPLIN 实验
## 安装
### gpu服务器账号注册
   在40、44、53号机上注册账号密码，53号机的注册流程参照实验室gpu服务器使用流程， [GPU服务使用指南](http://172.31.246.45:8080/docs/GPU%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)，并阅读其中GPU测试平台使用方法一节，完成VSCode 客户端安装，免密登录等配置。**40、44号机是九教强化学习组的新机器，账号注册我会帮忙，主要实验应在40、44号机完成，53号机的注册为练手使用**

  其中各服务器IP地址为：

  40号，211.71.76.32

  43号，211.71.76.81

  44号，211.71.76.95

  53号，172.31.246.53

  服务器需要使用校园网账号联网，联网的脚本是

    curl -d "callback=dr1583040563683&DDDDD=你的校园网账号&upass=你的校园网密码&0MKKey=123456&R1=0&R3=0&R6=0&para=00&v6ip=&_=1583040239097" http://10.10.43.3/drcom/login

  断网的脚本是

    curl http://10.10.43.3/drcom/logout?callback=dr1583041023742&_=1583041015901

### github 协作
   1.将注册github的账号发给我，我来设置成协作者

   2.git clone 这个仓库

   3.可以创建一个自己的分支用于改动或熟悉代码，熟悉代码之前先不要改动主分支。

   4.关于github的版本控制、分布式存储、多人协作等，可以参考[git教程](https://www.liaoxuefeng.com/wiki/896043488029600)，我们需要了解版本控制与回退，分支创建与合并，多人协作推送/下载不同版本。

### mujoco安装
   安装mujoco ，直接用复制我文件夹下的.mujoco 文件，131，150，200三个版本都有,同时包含key，我们的实验主要使用的是**mujoo131版本**。  ```cp -r /data/ShenShuo/.mujoco /data/你的账户名```  
   mj150pro需要配置环境变量

### python虚拟环境配置

   1.在40、44号机上复制anaconda，由我来完成，在注册40、44号机账号时一并完成。

   2.新建虚拟环境 ```conda create -n poplin python=3.6```

   3.激活虚拟环境 ```conda activate poplin```

   4.用pip 安装 requirements.txt 文件中的包 ```pip install requirements.txt```

   5.使用conda 安装tensorflow-gpu已经对应的cudatoolkit ``` conda install tensorflow-gpu==1.14.0```,会自动匹配cuda、cudnn、和cudatoolkit的版本。如果没有cudatoolkit，则把后面这句也执行一下```conda install cudatoolkit ```

### 运行代码，测试
  该代码的入口文件在 scipts.py 中的 mbexp.py 中，cd 到demo_scripts这个目录下，执行``` bash PETS.sh``` 结果会保存在 POPLIN/log/中。

## 实验计划
### 验证：

  同样的动作序列和初始状态，在model和GT中看偏差大小，画出state偏差和horizon图，计算平均预测误差大小。(**已完成**）  
  &nbsp;&nbsp;&nbsp;&nbsp;后续需要将数据存储下来而非直接画图，同时要画出方差  
  &nbsp;&nbsp;&nbsp;&nbsp;动作序列应该安装策略生成，而非随机生成

  同一个环境下，调整horizon，看效果好坏。（**已完成**）  
  &nbsp;&nbsp;&nbsp;&nbsp;在cheetah上完成  
  &nbsp;&nbsp;&nbsp;&nbsp;后续需要在更多环境下测试

  同一个horizon下，找一个长horizon，调大pop_size，看效果变化。（**正在运行实验中**）  
  &nbsp;&nbsp;&nbsp;&nbsp;目前来看pop_size对于效果的增加是有限的

  PETS的其他传播方式对效果的影响 主要是TS，试一种即可（**下一步**）

### 改进：

  增加衰减系数gamma, 其他参数不变，看效果。（**下一步**）

  增加critic，其他参数不变，看效果。

  在CEM中计算动作序列价值时，减去model 预测的方差，看效果。

### 代码阅读：

  能否用GT模型规划，POLO和AOP都用了GT模型，阅读AOP的代码看如何使用，以及如何为shooting方法增加Critic [AOP](https://github.com/kzl/aop)

  看MBPO中环境的搭建，也是用xml + py文件的方式，看gym环境的使用方法 [MBPO](https://github.com/JannerM/mbpo)

  查看RAVE中对reward、value调整不确定性的实现 [RAVE](https://github.com/PaddlePaddle/PARL/tree/develop/examples/NeurIPS2019-Learn-to-Move-Challenge)

## 代码架构

  ```POPLIN```主文件下的几个文件中：  
  ```demo_scripts```和```my_scripts``` 是一些执行代码的脚本，```img```是原作者论文里的实验结果和图片。
  
  ```play```和```scripts```里是我自己写的一些测试用例或者学api的例子，其中实验的入口函数在```scripts```中的```mbexp.py```文件中。 ```mbexp.py```、```mbexp1.py```、```mbexp2.py```是我为了方便指定使用那块gpu设定的，分别对应服务器上的gpu 0、1、2块。  
  
   ```log```是我们在脚本中指定的存储实验结果的位置。关于运行实验的更多参数和log文件中存储的结构，可以看```README_ORIGIN``` 这个文件，是poplin这个库原始的readme.

   ```mbbl```这个文件主要存储的是mujoco和gym的环境文件，我们的实验中使用的env都是在这个文件中定义的，因为我们的model不学习reward，需要一个函数给定reward，所以不能直接用gym的环境。  暂时还没对比过这些环境和直接调用gym中对应的标准环境如gym.make("Ant-V2")这种对应环境的区别


  ```dmbrl```是主要部分的代码，我们主要也是在这里的```misc```文件下修改  
  &emsp;```env```这个文件包含了三个PETS中的实验环境，不重要。    
  &emsp;```config```这个文件是每个环境对应的参数表，包括环境相关的输入输出维度，任务长度等，模型相关的的神经网络结构、轨迹展开方式等，优化相关的优化方法选择、待选动作序列数等  
  &emsp;```modeling```这个文件包含了PETS中提到的两种环境模型，确定性集成模型和随机性集成模型。包括了神经网络结构，确定性\不确定性神经网络的输出定义，集成数量，轨迹传播方式等等。  
  &emsp;```controllers```这个文件是定义MPC控制流程，是实验最主要的流程。  
  &emsp;```misc```这个文件是最主要部分  
  &emsp;其中```optimizer```中定义了POPLINA、POPLINP、CEM的优化方法，在tensorflow中的计算图  
  &emsp;```optimizer/policy_network```中定义了tensorflow的一些基础组件和策略网络，**每种策略网络都对应POPLIN中的一种训练模式**，还定义了策略网络的训练。


## 分工
  ```dmrl```中的代码按照结构主要分为三块  
  一块是**model部分**如何定义集成的神经网络，如何定义与训练概率模型，如何用模型预测轨迹。  
  一块是**policy部分**，包括如何用tf建立个性化的神经网络，如何在神经网络的参数空间上加扰动，如何训练策略网络。  
  一块是**MPC部分**，MPC控制加CEM优化部分同时加上实验的主函数和参数表，，我们也按照这三块分工。

  **model部分：**  
  论文阅读：[PETS](https://arxiv.org/abs/1805.12114)，熟悉PETS中的概率集成模型，熟悉PETS中的轨迹展开方法  
  代码阅读：  
  ```dmbrl/modeling/layers/FC```部分,主要内容是建立集成神经网络中的一层。  
  ```dmbrl/modeling/models/BNN```部分，主要内容是根据FC中的全连接层建立model，如何训练model，还包括如何预测下一个状态，如何预测一条轨迹。  
  ```dmbrl/controller/MPC```部分345行 _compile_POPLINP_cost到底：主要内容是给定一个动作序列，如何使用model预测轨迹和reward。
  ```dmbrl/config/gym_cheetah```部分nn_constructor函数：看实验中的mdoel是如何传参并建立的。

  **policy部分：**  
  论文阅读：[POPLIN](https://arxiv.org/pdf/1906.08649.pdf),熟悉论文中POPLIN-A，POPLIN-P的优化方法，知道CEM的优化过程，CEM参考CS285 第10讲。  
  代码阅读：  
   ```dmbrl/optimizers/policy_network/whitening_util```部分, 存储state、action等信息，用于归一化，用tensorflow定义了归一化的计算option与variable  
  ```dmbrl/optimizers/policy_network/tf_networks```MLP、W_MLP，WZ_MLP部分。 定义了如何个性化的创建全连接网络，包括初始化、激活函数、神经元数、层数等。  
  ```dmbrl/optimizers/policy_network/base_policy & BC_A_policy & BC_WA_policy & BC_WD_policy```部分。base_policy是基类，其余的策略都集成base_policy，每种策略都对应着上一部分的一种神经网络，也对应着一种POPLIN训练方法

  **MPC+CEM部分：**
  论文阅读：[PETS](https://arxiv.org/abs/1805.12114) + [POPLIN](https://arxiv.org/pdf/1906.08649.pdf)  
  代码阅读：  
  ```dmbrl/controller/MPC```部分，定义了优化器（POPLIN—A、POPLIN-P、CEM）和TF计算图，定义了模型与策略训练的数据流与存储，定义了reward的计算方式（POPLIN—A、POPLIN-P）  
  ```dmbrl/config/gym_cheetah```部分对于每一个环境，设定了一个对应的参数表，这个gym_cheetah文件对应的就是gym_cheetah中这个环境的默认参数表  
  ```dmbrl/misc/MBExp```部分，根据测定的epoch、train_iter等参数调用agent和policy采样与训练，完成一轮实验  
  ```dmbrl/misc/Agent```部分，给定策略和rollout长度等设置，agent根据这些在环境中采样一条轨迹  
  ```dmbrl/scripts/mbexp```部分，读取命令行的参数，调用对应环境的config并用命令行的参数覆盖对应的参数，建立用参数表建立MPC类，将MPC类传入MBExp开始一次实验  

## 部分实验结果
  **待添加**








  