## Attention is not enough
该部分对比在阿里巴巴数据集上对比了DIN和DIEN的效果，验证DIEN模型能有效缓解DIN模型target attention机制的不足

### 数据集下载
https://tianchi.aliyun.com/dataset/dataDetail?dataId=56       
 
     下载完成后，将数据集文件 train.csv、test.csv 和 embedding_count.csv 放置在 ./data 目录下

### 运行
     运行run_model.py文件，注意修改路径

### 查看训练日志:
     训练过程中的日志信息会保存在 ./train_log 目录下。你可以使用 TensorBoard 来可视化训练过程