## Attention is what you need, but not enough
该作业分成两个板块：

### 1. Attention is what you need
该部分完成三个任务： 

     1. 模型比较：使用DeepCTR库在 MovieLens 20M Dataset 数据集上对比 Wide&Deep, DeepFM, DIN模型的效果
     2. 搭建DIN模型：不使用DeepCTR库，从零开始搭建DIN模型，生成数据完成简单的CTR预测
     3. 消融实验：去掉DIN模型的attention机制，改为avg pooling，验证attention机制的作用

运行步骤见[这里](Attention is you need/README.md)

### 2. But attention is not enough
    该部分在阿里巴巴数据集上对比了DIN和DIEN的效果，验证DIEN模型能有效缓解DIN模型target attention机制的不足

运行步骤见[这里](But not enough/README.md)


### 注意事项
#### 1. 关于gpu
第一部分 Attention is you need 代码中进行采样训练，可以不使用gpu运行        

第二部分为了更好地体现din和dien对大规模数据的真实效果，用全部样本进行训练，以验证din，请使用gpu

#### 2. 关于断点
由于作者偷懒（bushi），将数据处理、特征处理、训练、预测合为一个文件，因此虽然也可直接运行，但更建议在每一重要节点打断点运行
