import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import WDL, DeepFM, DIN
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
import tensorflow as tf
from tqdm import tqdm
import sys


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error configuring GPU memory growth: {e}")


class TqdmCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        print("\n训练开始...")
        self.epoch_bar = tqdm(total=self.epochs, desc="Epoch", position=0, leave=True)

    def on_epoch_begin(self, epoch, logs=None):
        if self.batch_bar:
            self.batch_bar.close()
        self.current_epoch = epoch
        self.batch_bar = None

    def on_batch_begin(self, batch, logs=None):
        if not self.batch_bar:
            total_batches = self.params['steps'] if self.params['steps'] else None
            self.batch_bar = tqdm(
                total=total_batches,
                desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
                position=1,
                leave=False
            )
        self.batch_bar.update(1)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss', float('nan'))
        auc = logs.get('auc', float('nan'))
        self.batch_bar.set_postfix({
            'loss': f"{loss:.4f}",
            'auc': f"{auc:.4f}"
        })

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get('val_loss', float('nan'))
        val_auc = logs.get('val_auc', float('nan'))
        self.epoch_bar.set_postfix({
            'val_loss': f"{val_loss:.4f}",
            'val_auc': f"{val_auc:.4f}"
        })
        self.epoch_bar.update(1)

    def on_train_end(self, logs=None):
        if self.batch_bar:
            self.batch_bar.close()
        self.epoch_bar.close()
        print("训练完成！")


def preprocess_movielens(data_path='./achive'):
    print("\n🔄 开始数据预处理...")

    print("📖 1/7: 读取数据...")
    ratings = pd.read_csv(f'{data_path}/rating.csv')

    #这里可以设置数据量的大小，movielens数据集共两千多万条，这里我们默认取最新的5w条
    print("⏳ 按时间排序并选取最新的5w条记录...")
    ratings = ratings.sort_values('timestamp', ascending=False).head(50000)
    ratings = ratings.reset_index(drop=True)

    print(f"已选取最新的 {len(ratings)} 条记录")
    movies = pd.read_csv(f'{data_path}/movie.csv')

    print("🔀 2/7: 合并电影类别信息...")
    ratings = ratings.merge(movies[['movieId', 'genres']], on='movieId', how='left')


    print("🔍 3/7: 构造正负样本...")
    ratings['label'] = (ratings['rating'] >= 4).astype(int)
    ratings = ratings[['userId', 'movieId', 'genres', 'label']]


    print("✂️ 划分训练集和测试集...")
    train, test = train_test_split(ratings, test_size=0.2, random_state=RANDOM_SEED)


    print("🔢 5/7: 离散特征编码...")
    for feat in tqdm(['userId', 'movieId'], desc="编码进度"):
        le = LabelEncoder()
        train[feat] = le.fit_transform(train[feat])


        test[feat] = test[feat].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        test[feat] = test[feat].replace(-1, 0)  # 将未见过的值映射为0


    print("📊 6/7: 处理多值类别特征...")


    first_genres = []
    for s in tqdm(train['genres'].str.split('|'), desc="提取类别进度", total=len(train)):
        if s and len(s) > 0:
            first_genres.append(s[0])
        else:
            first_genres.append('')

    unique_first_genres = list(set(first_genres))
    genre_encoder = {g: i + 1 for i, g in enumerate(unique_first_genres)}


    def encode_first_genre(x):
        if isinstance(x, str):
            genres = x.split('|')
            if genres and len(genres) > 0:
                return genre_encoder.get(genres[0], 0)  # 返回第一个类别编码，不存在则为0
        return 0

    train['genres'] = [encode_first_genre(x) for x in tqdm(train['genres'], desc="编码训练集类别进度")]
    test['genres'] = [encode_first_genre(x) for x in tqdm(test['genres'], desc="编码测试集类别进度")]


    print("🔢 7/7: 确保所有特征都是数值类型...")
    for col in ['userId', 'movieId', 'label', 'genres']:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        test[col] = pd.to_numeric(test[col], errors='coerce')


    train = train.fillna(0)
    test = test.fillna(0)


    print("🕒 8/8: 构建用户历史行为序列...")
    user_history = train[['userId', 'movieId', 'label']].sort_values('userId')
    user_item_dict = user_history[user_history['label'] == 1].groupby('userId')['movieId'].apply(list).to_dict()


    print("🎯 9/9: 生成带历史序列的样本...")

    def generate_samples(df):
        samples = []
        for idx, row in tqdm(df.iterrows(), desc="生成样本进度", total=len(df)):
            user_id = row['userId']
            hist = user_item_dict.get(user_id, [])

            # 默认负样本至少有1个，可自行修改
            if len(hist) == 0 and row['label'] == 0:
                continue

            # 默认使用10条用户行为序列，可自行修改
            hist = hist[-10:]
            hist = [0] * (10 - len(hist)) + hist

            samples.append({
                'userId': user_id,
                'movieId': row['movieId'],
                'genres': row['genres'],
                'hist_movieId': hist,
                'label': row['label']
            })
        return pd.DataFrame(samples)

    train = generate_samples(train)
    test = generate_samples(test)

    print("✅ 数据预处理完成！")
    return train, test, unique_first_genres



def get_feature_columns(train, test, unique_genres):
    embedding_dim = 8

    user_feature = SparseFeat(
        'userId',
        vocabulary_size=train['userId'].max() + 1,
        embedding_dim=embedding_dim
    )

    movie_feature = SparseFeat(
        'movieId',
        vocabulary_size=train['movieId'].max() + 1,
        embedding_dim=embedding_dim
    )

    genre_feature = SparseFeat(
        'genres',
        vocabulary_size=train['genres'].max() + 1,
        embedding_dim=embedding_dim  # 修改为与其他特征相同的维度
    )


    history_feature = VarLenSparseFeat(
        SparseFeat('hist_movieId', vocabulary_size=train['movieId'].max() + 1, embedding_dim=embedding_dim),
        maxlen=10,
        combiner='mean'
    )

    sparse_features = [user_feature, movie_feature, genre_feature]
    sequence_features = [history_feature]
    wide_features = [user_feature, movie_feature, genre_feature]
    deep_features = sparse_features + sequence_features

    return wide_features, deep_features, sparse_features, sequence_features



def train_and_evaluate(model_name, feature_columns, train, test, history_feature=None):
    print("\n🔍 验证输入数据...")


    for col in ['userId', 'movieId', 'genres']:
        print(f"列名: {col}")
        print(f"  类型: {train[col].dtype}")
        print(f"  训练集最小值: {train[col].min()}")
        print(f"  训练集最大值: {train[col].max()}")
        print(f"  测试集最小值: {test[col].min()}")
        print(f"  测试集最大值: {test[col].max()}")
        print(f"  样本值: {train[col].iloc[0]}")


    if 'hist_movieId' in train:
        hist_lengths = [len(h) for h in train['hist_movieId']]
        print(f"历史序列长度 - 最小: {min(hist_lengths)}, 最大: {max(hist_lengths)}, 平均: {np.mean(hist_lengths):.2f}")


    train_model_input = {name: train[name] for name in get_feature_names(feature_columns)}
    test_model_input = {name: test[name] for name in get_feature_names(feature_columns)}

    if 'hist_movieId' in train_model_input:
        train_model_input['hist_movieId'] = np.array(train_model_input['hist_movieId'].tolist())
        test_model_input['hist_movieId'] = np.array(test_model_input['hist_movieId'].tolist())

    print("\n样本输入:")
    for name, value in list(train_model_input.items())[:3]:
        print(f"  {name}: {value.shape if hasattr(value, 'shape') else len(value)}")

    y_train = train['label'].values
    y_test = test['label'].values

    if model_name == 'WideDeep':
        model = WDL(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns)
    elif model_name == 'DeepFM':
        model = DeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns)
    elif model_name == 'DIN':
        sparse_features = [feat for feat in feature_columns if isinstance(feat, SparseFeat)]
        sequence_features = [feat for feat in feature_columns if isinstance(feat, VarLenSparseFeat)]

        model = DIN(
            dnn_feature_columns=sparse_features + sequence_features,
            history_feature_list=['movieId'],
            dnn_use_bn=True,
            dnn_hidden_units=(200, 80),
            dnn_activation='dice',
            att_hidden_size=(80, 40),
            att_activation='dice',
            l2_reg_dnn=0.0,
            l2_reg_embedding=1e-6,
            dnn_dropout=0.5,
            seed=RANDOM_SEED,
            task='binary'
        )
    else:
        raise ValueError("Model not supported")

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )


    epochs = 10
    history = model.fit(
        train_model_input, y_train,
        batch_size=256,
        epochs=epochs,
        validation_data=(test_model_input, y_test),
        verbose=0,
        callbacks=[TqdmCallback(epochs=epochs)]
    )

    test_auc = model.evaluate(test_model_input, y_test, verbose=0)[1]
    print(f"✅ {model_name} Test AUC: {test_auc:.4f}")
    return model, history


#强烈建议这里一步步打断点操作
if __name__ == "__main__":
    # 1. 数据预处理
    train, test, unique_genres = preprocess_movielens()

    # 2. 特征工程
    print("\n🔄 开始特征工程...")
    wide_features, deep_features, sparse_features, sequence_features = get_feature_columns(
        train, test, unique_genres
    )
    history_feature = sequence_features[0]
    print("✅ 特征工程完成！")

    # 3. 定义要训练的模型列表
    models = ['WideDeep', 'DeepFM', 'DIN']

    # 4. 遍历训练模型（带总体进度条）
    print("\n🚀 开始模型训练...")
    for model_name in tqdm(models, desc="总体进度"):
        print(f"\n🔄 Training {model_name}...")
        if model_name == 'DIN':
            _, _ = train_and_evaluate(
                model_name,
                feature_columns=sparse_features + sequence_features,
                train=train,
                test=test,
                history_feature=history_feature
            )
        else:
            if model_name == 'WideDeep':
                feature_columns = wide_features + deep_features
            else:
                feature_columns = sparse_features
            _, _ = train_and_evaluate(model_name, feature_columns, train, test)

    print("\n🎉 所有模型训练完成！")