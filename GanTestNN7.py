import datetime
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import generate_data as gd

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# writer = tf.summary.create_file_writer(log_dir)
# tf.summary.trace_on(graph=True, profiler=True)


# ======================================================


# data_seed = np.random.randint(1,10000)
# np.random.seed(data_seed)
# useless, all_test_data = gd.eg6(500000, N_test=10, feature_num=10,
#                            stable_ratio=0.4, rho=0.7, quantile=0.6, r=0.99)
# test_sample_size = 1000

def model_inputs(p):
    # real data
    inputs_real = tf.placeholder(tf.float32, [None, p], name='input_real')
    # fake data
    inputs_fake = tf.placeholder(tf.float32, [None, p], name='input_fake')
    # learning rate
    learning_rate = tf.placeholder(tf.float32, name='lr')

    return inputs_real, inputs_fake, learning_rate


def discriminator(samples, reuse=False):
    # samples is a placeholder
    # output the value between 0 and 1, indicating real(0) or fake(1)
    p = samples.shape.as_list()[-1]
    with tf.variable_scope('discriminator', reuse=reuse):
        d_z1 = tf.layers.dense(samples, 2 * p)
        d_a1 = tf.nn.relu(d_z1)
        d_z2 = tf.layers.dense(d_a1, 1)
        d_out = tf.nn.sigmoid(d_z2)

    return d_out, d_z2


def generator(z, reuse=False):
    # z a placeholder, has the same length with sample size
    # generate the weight for real samples
    p = z.shape.as_list()[-1]
    with tf.variable_scope('generator', reuse=reuse):
        g_z1 = tf.layers.dense(z, 2 * p)
        g_a1 = tf.nn.relu(g_z1)
        g_z2 = tf.layers.dense(g_a1, 1)
        g_out = tf.nn.sigmoid(g_z2)

    return g_out


def model_loss(input_real, input_fake, z):
    w = generator(z, reuse=False)
    w = tf.reshape(w, shape=(1, -1))
    d_model_fake, d_logits_fake = discriminator(input_fake, reuse=False)

    d_model_real, d_logits_real = discriminator(input_real, reuse=True)

    smooth = 0.1
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.ones_like(d_logits_fake) * (
                                                                                 1 - smooth)))

    fenzi = tf.reduce_sum(tf.multiply(w, tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                                 labels=tf.zeros_like(d_logits_real))))
    fenmu = tf.reduce_sum(w)
    d_loss_real = fenzi / fenmu

    d_loss = d_loss_fake + d_loss_real

    fenzi_g = tf.reduce_sum(tf.multiply(w, tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                                   labels=tf.ones_like(d_logits_real))))
    fenmu_g = tf.reduce_sum(w)
    g_loss = fenzi_g / fenmu_g

    return d_loss, g_loss, w


def model_opt(d_loss, g_loss, learning_rate):
    # 定义优化器
    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt


def generate_fake_data(sample):
    sample_fake = deepcopy(sample)
    for i in range(sample.shape[1]):
        np.random.shuffle(sample_fake[:, i])

    return sample_fake


def train(sample, sample_fake, epoch_count, learning_rate):
    # sample is a np.ndarray
    n = sample.shape[0]
    p = sample.shape[1]
    losses_d = []
    losses_g = []
    w_s = []

    input_real, input_fake, lr = model_inputs(p)
    z = tf.placeholder(tf.float32, [None, p])
    d_loss, g_loss, w = model_loss(input_real, input_fake, z)

    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate=lr)

    steps = 0
    current_z = sample
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            steps += 1

            _ = sess.run(d_opt, feed_dict={input_real: sample,
                                           input_fake: sample_fake,
                                           z: current_z,
                                           lr: learning_rate})

            _ = sess.run(g_opt, feed_dict={input_real: sample,
                                           z: current_z,
                                           lr: learning_rate})

            if steps % 1000 == 0:
                train_loss_d = d_loss.eval({input_real: sample,
                                            input_fake: sample_fake,
                                            z: current_z
                                            })

                train_loss_g = g_loss.eval({input_real: sample,
                                            z: current_z
                                            })
                train_w = w.eval({z: current_z})
                train_w = train_w / np.sum(train_w)

                losses_d.append(train_loss_d)
                losses_g.append(train_loss_g)
                w_s.append(train_w)

                print("Epoch {}/{}...".format(epoch_i + 1, epoch_count),
                      "Discriminator Loss: {:.4f}...".format(train_loss_d),
                      "Generator Loss: {:.4f}...".format(train_loss_g))
    return losses_d, losses_g, w_s


data_seed = np.random.randint(1, 10000)
np.random.seed(data_seed)
useless, all_data = gd.eg6(500000, N_test=10, feature_num=10,
                           stable_ratio=0.4, rho=0.7, quantile=0.6, r=0.99)
sample_size = 1000
epochs = 30000
bias_rate = 0.5
lr = 1e-4
data_train = gd.concate_biased_unbiased(all_data, size=sample_size, r_bias=bias_rate)

X = np.concatenate((data_train['stable'], data_train['noise']), axis=1)
X_fake = generate_fake_data(X)

X1 = pd.DataFrame(X)
X1_fake = pd.DataFrame(X_fake)
corX = X1.corr()
cor_fake = X1_fake.corr()

tf.reset_default_graph()
with tf.Graph().as_default():
    loss_d, loss_g, w_s = train(X, X_fake, epoch_count=epochs, learning_rate=lr)

for i in range(len(w_s)):
    if i % 5 == 0:
        plt.figure()
        plt.plot(w_s[i].reshape(-1))
        plt.title('{}th weights'.format(i))
        plt.show()

for i in range(len(w_s)):
    print("biased total weight:{}".format(np.sum(w_s[i][0, :int(sample_size * bias_rate)])),
          "unbiased total weight:{}".format(np.sum(w_s[i][0, int(sample_size * bias_rate):])))

## test data
data_seed = np.random.randint(1, 10000)
np.random.seed(data_seed)
useless, all_data_test = gd.eg6(500000, N_test=10, feature_num=10,
                                stable_ratio=0.4, rho=0.7, quantile=0.6, r=0.99)

test_sample_size = 1000

X_train = X
y_train = data_train['Y']

prd_method = 'LR'  # LR=linear regression
if prd_method == 'LR':
    from sklearn.linear_model import LinearRegression

    reg_baseline = LinearRegression()
    reg_common = LinearRegression()

    reg_weighted = []
    for reg_iter in range(len(w_s)):
        temp = LinearRegression()
        reg_weighted.append(temp)

    # ====== fitting ======
    # without weight
    reg_baseline.fit(X_train[int(sample_size * bias_rate):, :], y_train[int(sample_size * bias_rate):, :])
    reg_common.fit(X_train, y_train)
    # with weight
    for i in range(len(w_s)):
        w_prd = w_s[i].reshape(-1)
        reg_weighted[i].fit(X_train, y_train, w_prd)

    # test set performance
    mse_records = []
    x_range = []
    y_range = []
    test_bias_rate_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for test_br in test_bias_rate_range:
        test_data = gd.concate_biased_unbiased(all_data_test, size=test_sample_size, r_bias=test_br)
        X_test = np.concatenate((test_data['stable'], test_data['noise']), axis=1)
        y_test = test_data['Y'].reshape(-1, 1)
        x_range.append(np.std(X_test, axis=0))
        y_range.append(np.std(y_test, axis=0))
        mse_dict = {}
        # baseline, without weight
        y_baseline_prd = reg_baseline.predict(X_test)
        mse_baseline = np.sum(np.square(y_test - y_baseline_prd)) / y_test.shape[0]
        print("baseline performance:\nMSE:{:.4f}".format(mse_baseline))
        mse_dict['baseline'] = mse_baseline
        # common, without weight
        y_common_prd = reg_common.predict(X_test)
        mse_common = np.sum(np.square(y_test - y_common_prd)) / y_test.shape[0]
        print("normal performance:\nMSE:{:.4f}".format(mse_common))
        mse_dict['common'] = mse_common
        # with weight
        for i in range(len(reg_weighted)):
            w_prd = w_s[i].reshape(-1)
            reg_weighted[i].fit(X_train, y_train, w_prd)
            y_weight_prd = reg_weighted[i].predict(X_test)
            mse_weighted = np.sum(np.square(y_test - y_weight_prd)) / y_test.shape[0]
            print("reweighted performance of {}:\nMSE:{:.4f}".format(i, mse_weighted))
            mse_dict['{}iter_mse'.format(i)] = mse_weighted

        mse_records.append(mse_dict)
    mse_records_df = pd.DataFrame(mse_records)
    mse_records_df.index = test_bias_rate_range
