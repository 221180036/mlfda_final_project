# Attention is you need
该部分完成三个任务： 

     1. 模型比较：使用DeepCTR库在 MovieLens 20M Dataset 数据集上对比 Wide&Deep, DeepFM, DIN模型的效果
     2. 搭建DIN模型：不使用DeepCTR库，从零开始搭建DIN模型，生成数据完成简单的CTR预测
     3. 消融实验：去掉DIN模型的attention机制，改为avg pooling，验证attention机制的作用
任务二和任务三已合并为一个文件


### 安装deepctr包
    pip install --no-warn-conflicts -q deepctr

### 数据集下载
     第一个任务数据集使用MovieLens 20M Dataset
https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset
    
     第二、三个任务不需下载数据集，py文件中已生成数据

### 运行
#### 1. 模型比较：
运行compare.py文件，注意修改数据读取路径        

MovieLens数据集共2000w+条，compare.py文件默认使用最新的5w条，若要修改，请在preprocess_movielens函数中修改         

示例：
    
    # 这里可以设置数据量的大小，movielens数据集共两千多万条，这里我们默认取最新的5w条
    print("⏳ 按时间排序并选取最新的5w条记录...")
    ratings = ratings.sort_values('timestamp', ascending=False).head(50000)
    ratings = ratings.reset_index(drop=True)

我们默认使用10条用户行为序列，但如果数据量足够，我们建议使用50条，若要修改，请在generate_samples函数中修改

示例：

            # 默认负样本至少有1个，可自行修改
            if len(hist) == 0 and row['label'] == 0:
                continue

            # 默认使用10条用户行为序列，可自行修改
            hist = hist[-10:]
            hist = [0] * (10 - len(hist)) + hist

     
#### 2. 搭建模型与消融实验：
    运行ablation_study.py文件

