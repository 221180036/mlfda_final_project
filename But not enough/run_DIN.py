import tensorflow as tf
import pandas as pd
from DIN_DIEN import DIN
import data_tools as data_reader
import utils
import os

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        return 0
    except:
        return 1
model_name = "din"
def is_in_notebook():
    import sys
    return 'ipykernel' in sys.modules
def clear_output():
    """
    clear output for both jupyter notebook and the console
    """
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
    if is_in_notebook():
        from IPython.display import clear_output as clear
        clear()
train_data, test_data, embedding_count = data_reader.get_data()

embedding_features_list = data_reader.get_embedding_features_list()
user_behavior_features = data_reader.get_user_behavior_features()
embedding_count_dict = data_reader.get_embedding_count_dict(embedding_features_list, embedding_count)
embedding_dim_dict = data_reader.get_embedding_dim_dict(embedding_features_list)
import time
stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
mkdir("./train_log/" + model_name)
log_path = "./train_log/" + model_name + "/%s" % stamp
train_summary_writer = tf.summary.create_file_writer(log_path)
tf.summary.trace_on(graph=True, profiler=True)
loss_file_name = utils.get_file_name()
mkdir("./loss/" + model_name + "/")
utils.make_train_loss_dir(loss_file_name, cols = ["train_final_loss"], model = model_name)
utils.make_test_loss_dir(loss_file_name, cols = ["test_final_loss"], model = model_name)
model = DIN(
    embedding_count_dict,
    embedding_dim_dict,
    embedding_features_list,
    user_behavior_features,
    activation="dice"
)

min_batch = 0
batch = 100
optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
loss_metric = tf.keras.metrics.Sum()
auc_metric = tf.keras.metrics.AUC()
alpha = 1
epochs = 3
label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show, min_batch, clk_length, show_length = data_reader.get_batch_data(train_data, min_batch, batch = batch)
def get_train_data(label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show):
    user_profile_dict = {
        "cms_segid": cms_segid,
        "cms_group": cms_group,
        "gender": gender,
        "age": age,
        "pvalue": pvalue,
        "shopping": shopping,
        "occupation": occupation,
        "user_class_level": user_class_level
    }
    user_profile_list = ["cms_segid", "cms_group", "gender", "age", "pvalue", "shopping", "occupation", "user_class_level"]
    user_behavior_list = ["brand", "cate"]
    click_behavior_dict = {
        "brand": hist_brand_behavior_clk,
        "cate": hist_cate_behavior_clk
    }
    noclick_behavior_dict = {
        "brand": hist_brand_behavior_show,
        "cate": hist_cate_behavior_show
    }
    target_item_dict = {
        "brand": target_cate,
        "cate": target_brand
    }
    return user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict
user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data(label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show)
def train_one_step(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, clk_length):
        with tf.GradientTape() as tape:
            output, logit = model(
                user_profile_dict,
                user_profile_list,
                click_behavior_dict,
                target_item_dict,
                user_behavior_list,
                clk_length
            )
            final_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=tf.cast(label, dtype=tf.float32)))
            print("final_loss=" + str(final_loss))
        gradient = tape.gradient(final_loss, model.trainable_variables)
        clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
        optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))
        loss_metric(final_loss)
        return final_loss.numpy()
def get_test_loss(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, clk_length):
    output, logit = model(
        user_profile_dict,
        user_profile_list,
        click_behavior_dict,
        target_item_dict,
        user_behavior_list,
        clk_length
    )
    final_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=tf.cast(label, dtype=tf.float32)))
    print("final_loss=" + str(final_loss))
    return final_loss.numpy()
get_test_loss(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, clk_length)


def get_loss_fig(train_loss, test_loss):
    loss_list = ["final_loss"]
    color_list = ["r", "b"]
    plt.figure()
    cnt = 0
    for k in loss_list:
     loss = train_loss[k]
     step = list(np.arange(len(loss)))
     plt.plot(step, loss, color_list[cnt] + "-", label="train_" + k, linestyle="--")
     cnt += 1
    cnt = 0
    for k in loss_list:
     loss = test_loss[k]
     step = list(np.arange(len(loss)))
     plt.plot(step, loss, color_list[cnt], label="test_" + k)
     cnt += 1
    plt.title("Loss")
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    clear_output()
    mkdir("./loss/" + model_name)
    plt.savefig("./loss/" + model_name + "/loss.png")
    clear_output()
    plt.show()


def record_test_loss(test_loss, test_data, step):
 label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show, clk_length, show_length = data_reader.get_test_data(
  test_data)
 user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data(
  label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level,
  hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show)
 final_loss = get_test_loss(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict,
                            noclick_behavior_dict, user_behavior_list, label, clk_length)
 loss_dict = dict()
 loss_dict["final_loss"] = str(final_loss)
 utils.add_loss(loss_dict, loss_file_name, cols=["final_loss"], level="test", model=model_name)
 test_loss["final_loss"].append(float(final_loss))
 with train_summary_writer.as_default():
  tf.summary.scalar("test_final_loss epoch: " + str(epoch), final_loss, step=step)


mkdir("./checkpoint/" + model_name)
checkpoint_path = "./checkpoint/" + model_name + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
train_loss = {"final_loss": []}
test_loss = {"final_loss": []}
for epoch in range(epochs):
 for i in range(int(len(train_data) / batch)):
  label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show, min_batch, clk_length, show_length = data_reader.get_batch_data(
   train_data, min_batch, batch=batch)
  record_test_loss(test_loss, test_data, i)
  user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data(
   label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level,
   hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show)
  final_loss = train_one_step(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict,
                              noclick_behavior_dict, user_behavior_list, label, clk_length)
  # Record_loss12
  loss_dict = dict()
  loss_dict["final_loss"] = str(final_loss)
  utils.add_loss(loss_dict, loss_file_name, cols=["final_loss"], level="train", model=model_name)
  train_loss["final_loss"].append(float(final_loss))
  get_loss_fig(train_loss, test_loss)
  tf.summary.trace_on(graph=True, profiler=True)
  with train_summary_writer.as_default():
   tf.summary.scalar("train_final_loss epoch: " + str(epoch), final_loss, step=i)
   tf.summary.trace_export(
    name="DIN",
    step=i,
    profiler_outdir=log_path)
 model.save_weights(checkpoint_path.format(epoch=epoch))
utils.mkdir("./loss/" + "din" + "/")

last_model =  DIN(
    embedding_count_dict,
    embedding_dim_dict,
    embedding_features_list,
    user_behavior_features,
    activation="dice"
)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
last_model.load_weights(latest)
model= last_model
label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show, min_batch, clk_length, show_length = data_reader.get_batch_data(train_data, min_batch, batch = batch)
user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data(label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show)
final_loss = get_test_loss(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, clk_length)
final_loss


def convert_tensor(data):
 return tf.convert_to_tensor(data)


def get_normal_data(data, col):
 return data[col].values


def get_sequence_data(data, col):
 rst = []
 max_length = 0
 for i in data[col].values:
  temp = len(list(map(eval, i[1:-1].split(","))))
  if temp > max_length:
   max_length = temp

 for i in data[col].values:
  temp = list(map(eval, i[1:-1].split(",")))
  padding = np.zeros(max_length - len(temp))
  rst.append(list(np.append(np.array(temp), padding)))
 return rst


def get_length(data, col):
 rst = []
 for i in data[col].values:
  temp = len(list(map(eval, i[1:-1].split(","))))
  rst.append(temp)
 return rst


def get_evaluate_data(data):
 batch_data = data
 click = get_normal_data(batch_data, "guide_dien_final_train_data.clk")
 target_cate = get_normal_data(batch_data, "guide_dien_final_train_data.cate_id")
 target_brand = get_normal_data(batch_data, "guide_dien_final_train_data.brand")
 cms_segid = get_normal_data(batch_data, "guide_dien_final_train_data.cms_segid")
 cms_group = get_normal_data(batch_data, "guide_dien_final_train_data.cms_group_id")
 gender = get_normal_data(batch_data, "guide_dien_final_train_data.final_gender_code")
 age = get_normal_data(batch_data, "guide_dien_final_train_data.age_level")
 pvalue = get_normal_data(batch_data, "guide_dien_final_train_data.pvalue_level")
 shopping = get_normal_data(batch_data, "guide_dien_final_train_data.shopping_level")
 occupation = get_normal_data(batch_data, "guide_dien_final_train_data.occupation")
 user_class_level = get_normal_data(batch_data, "guide_dien_final_train_data.new_user_class_level")
 hist_brand_behavior_clk = get_sequence_data(batch_data, "guide_dien_final_train_data.click_brand")
 hist_cate_behavior_clk = get_sequence_data(batch_data, "guide_dien_final_train_data.click_cate")
 hist_brand_behavior_show = get_sequence_data(batch_data, "guide_dien_final_train_data.show_brand")
 hist_cate_behavior_show = get_sequence_data(batch_data, "guide_dien_final_train_data.show_cate")
 clk_length = get_length(batch_data, "guide_dien_final_train_data.click_brand")
 show_length = get_length(batch_data, "guide_dien_final_train_data.show_brand")
 return tf.one_hot(click, 2), convert_tensor(target_cate), convert_tensor(target_brand), convert_tensor(
  cms_segid), convert_tensor(cms_group), convert_tensor(gender), convert_tensor(age), convert_tensor(
  pvalue), convert_tensor(shopping), convert_tensor(occupation), convert_tensor(user_class_level), convert_tensor(
  hist_brand_behavior_clk), convert_tensor(hist_cate_behavior_clk), convert_tensor(
  hist_brand_behavior_show), convert_tensor(hist_cate_behavior_show), clk_length, show_length


label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show, clk_length, show_length = get_evaluate_data(
 test_data)
user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data(
 label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level,
 hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show)
output, logit = model(
 user_profile_dict,
 user_profile_list,
 click_behavior_dict,
 target_item_dict,
 user_behavior_list,
 clk_length
)

train_label = train_data["guide_dien_final_train_data.clk"].values
positive_num = len(train_label[train_label == 1])
negative_num = len(train_label[train_label == 0])
print("[训练集]正例:负例=%d : %d" % (positive_num, negative_num))
test_label = test_data["guide_dien_final_train_data.clk"].values
positive_num = len(test_label[test_label == 1])
negative_num = len(test_label[test_label == 0])
print("[测试集]正例:负例=%d : %d" % (positive_num, negative_num))

y_true = label.numpy()[:,-1]
y_score = output.numpy()[:,-1]

threshold = 0.01
y_pre = y_score.copy()
y_pre[y_pre > threshold] = 1
y_pre[y_pre <= threshold] = 0
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import sklearn.metrics as sm
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
print(accuracy_score(y_true, y_pre))

y_pre[y_pre == 1]
m = sm.confusion_matrix(y_true, y_pre)
print('混淆矩阵为：', m, sep='\n')
r = sm.classification_report(y_true, y_pre)
print('分类报告为：', r, sep='\n')

from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_true,y_score)
auc_score


def plot_roc(labels, predict_prob):
 false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
 roc_auc = auc(false_positive_rate, true_positive_rate)
 plt.title('ROC')
 plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
 plt.legend(loc='lower right')
 plt.plot([0, 1], [0, 1], 'r--')
 plt.ylabel('TPR')
 plt.xlabel('FPR')
 plt.show()
plot_roc(y_true, y_score)

train_loss_data = pd.read_csv("请改为你前面生成的train_loss路径")
test_loss_data = pd.read_csv("请改为你前面生成的test_loss路径")


def get_loss_fig_aux(train_loss_data, test_loss_data):
 train_loss = {
  "final_loss": list(train_loss_data["train_" + "final_loss"].values)
 }
 test_loss = {
  "final_loss": list(test_loss_data["test_" + "final_loss"].values)
 }
 get_loss_fig(train_loss, test_loss)


get_loss_fig_aux(train_loss_data, test_loss_data)


