import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import time

start_time = time.time()
print("Start time--- %s seconds ---" % (start_time))

##The path to the training folders and validation folder for images and labels
img_tr_path = "/data/home/kardee/HW6_Dataset/image/train/"
img_val_path = "/data/home/kardee/HW6_Dataset/image/valid/"
img_tst_path = "/data/home/kardee/HW6_Dataset/image/test/"

lable_tr_path = "/data/home/kardee/HW6_Dataset/label/train/"
lable_val_path = "/data/home/kardee/HW6_Dataset/label/valid/"
lable_tst_path = "/data/home/kardee/HW6_Dataset/label/test/"


def read_tr_data():
    # Function to read the Training Data
    d_tr = np.empty([199, 352, 1216, 3])
    lb_tr = np.empty([199, 352, 1216])

    img_file = os.listdir(img_tr_path)
    label_file = os.listdir(lable_tr_path)
    for i in range(199):
        d_tr[i] = cv2.imread(img_tr_path + img_file[i])

    for i in range(199):
        with open(lable_tr_path + label_file[i], "rb") as f:
            lb_tr[i] = pickle.load(f, encoding="bytes")
        f.close()
    return d_tr, lb_tr


def read_val_data():
    # function to read the validation data
    d_v = np.empty([45, 352, 1216, 3])
    lb_v = np.empty([45, 352, 1216])

    im_file = os.listdir(img_val_path)
    lab_file = os.listdir(lable_val_path)
    for i in range(45):
        d_v[i] = cv2.imread(img_val_path + im_file[i])

    for i in range(45):
        with open(lable_val_path + lab_file[i], "rb") as f:
            lb_v[i] = pickle.load(f, encoding="bytes")
        f.close()
    return d_v, lb_v


def read_test_data():
    # function to read the test data
    d_tst = np.empty([45, 352, 1216, 3])
    lb_tst = np.empty([45, 352, 1216])

    im_file = os.listdir(img_tst_path)
    lab_file = os.listdir(lable_tst_path)
    for i in range(45):
        d_tst[i] = cv2.imread(img_tst_path + im_file[i])

    for i in range(45):
        with open(lable_tst_path + lab_file[i], "rb") as f:
            lb_tst[i] = pickle.load(f, encoding="bytes")
        f.close()
    return d_tst, lb_tst


X_train, Y_train = read_tr_data()
X_val, Y_val = read_val_data()
X_test, Y_test = read_test_data()

##Convert the data to type int
X_train = X_train.astype(int)
X_val = X_val.astype(int)
Y_train = Y_train.astype(int)
Y_val = Y_val.astype(int)

##INPUTS
X = tf.placeholder(tf.float32, shape=(None, 352, 1216, 3), name='input_x')
Y = tf.placeholder(tf.float32, shape=(None, 352, 1216), name='output_y')


# MODEL
def FCN32s_model(X):
    # X is the feature vector
    # Convolutional layers
    conv1 = tf.layers.conv2d(inputs=X, filters=64, kernel_size=[3, 3],
                             strides=1, padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3]
                             , strides=1, padding="same", activation=tf.nn.relu)
    # Pooling layer
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2]
                                    , strides=2, padding="same")

    # Convolutional layers
    conv3 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3]
                             , strides=1, padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3]
                             , strides=1, padding="same", activation=tf.nn.relu)
    # Pooling layer
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2]
                                    , strides=2, padding="same")

    # Convolutional layers
    conv5 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3]
                             , strides=1, padding="same", activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(inputs=conv5, filters=256, kernel_size=[3, 3]
                             , strides=1, padding="same", activation=tf.nn.relu)
    conv7 = tf.layers.conv2d(inputs=conv6, filters=256, kernel_size=[3, 3]
                             , strides=1, padding="same", activation=tf.nn.relu)
    # Pooling layer
    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2]
                                    , strides=2, padding="same")

    # Convolutional layers
    conv8 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3]
                             , strides=1, padding="same", activation=tf.nn.relu)
    conv9 = tf.layers.conv2d(inputs=conv8, filters=512, kernel_size=[3, 3]
                             , strides=1, padding="same", activation=tf.nn.relu)
    conv10 = tf.layers.conv2d(inputs=conv9, filters=512, kernel_size=[3, 3]
                              , strides=1, padding="same", activation=tf.nn.relu)

    # Pooling layer
    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2],
                                    strides=2, padding="same")

    # Convolutional layers
    conv11 = tf.layers.conv2d(inputs=pool4, filters=512, kernel_size=[3, 3]
                              , strides=1, padding="same", activation=tf.nn.relu)
    conv12 = tf.layers.conv2d(inputs=conv11, filters=512, kernel_size=[3, 3]
                              , strides=1, padding="same", activation=tf.nn.relu)
    conv13 = tf.layers.conv2d(inputs=conv12, filters=512, kernel_size=[3, 3]
                              , strides=1, padding="same", activation=tf.nn.relu)
    # Pooling layer
    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2]
                                    , strides=2, padding="same")

    ##The FC layer of VGG is replaced by convolution layer in FCN32s
    conv14 = tf.layers.conv2d(inputs=pool5, filters=4096, kernel_size=[7, 7]
                              , strides=1, padding="same", activation=tf.nn.relu)

    conv15 = tf.layers.conv2d(inputs=conv14, filters=4096, kernel_size=[1, 1]
                              , strides=1, padding="same", activation=tf.nn.relu)
    conv16 = tf.layers.conv2d(inputs=conv15, filters=1, kernel_size=[1, 1]
                              , strides=1, padding="same", activation=tf.nn.relu)

    ##Deconvolution Layer
    decon1 = tf.layers.conv2d_transpose(inputs=conv16, filters=1, kernel_size=[64, 64]
                                        , strides=32, padding="same")
    decon1 = decon1[:, :, :, 0]  ## as the output was (352,1216,1)

    return decon1


logit = FCN32s_model(X)
learningrate = 0.001
momentum = 0.99

##Masking

mask = (Y >= 0)
log_m = tf.boolean_mask(logit, mask)
Y_m = tf.boolean_mask(Y, mask)

# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=log_m, labels=Y_m))
optimizer = tf.train.MomentumOptimizer(learning_rate=learningrate, momentum=momentum)
train_op = optimizer.minimize(loss)


##Evaluation Metric
def IOU_eval(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    return TP, FP, FN


saver = tf.train.Saver()

EPOCH = 20
BATCH_SIZE = 1

# Variable Initialize
init = tf.global_variables_initializer()

ls_tr = np.array([])
iu_tr = np.array([])

ls_val = np.array([])
iu_val = np.array([])

##TRAINING
with tf.Session() as sess:
    # Run initializer
    sess.run(init)
    # number of training examples
    num_examples = X_train.shape[0]
    val_num_ex = X_val.shape[0]
    print("Training...")
    print()
    for i in range(EPOCH):
        loss_tr = 0
        IOU_tr = 0
        TP = 0
        FP = 0
        FN = 0
        for j in range(num_examples):
            mini_X = np.empty(shape=(1, 352, 1216, 3))
            mini_Y = np.empty(shape=(1, 352, 1216))
            # select randomly one sample that will form our minibatch
            idx = np.random.randint(0, 199)  # one index out of the 199 images
            mini_X[0] = X_train[idx]
            mini_Y[0] = Y_train[idx]

            op = sess.run(train_op, feed_dict={X: mini_X, Y: mini_Y})
            pred_label = sess.run(logit, feed_dict={X: mini_X})
            pred_label[pred_label >= 0] = 1
            pred_label[pred_label < 0] = 0

            TP_b, FP_b, FN_b = IOU_eval(mini_Y, pred_label)
            TP += TP_b
            FP += FP_b
            FN += FN_b

            loss_batch = sess.run(loss, feed_dict={X: mini_X, Y: mini_Y})
            loss_tr += loss_batch

        loss_tr = loss_tr / num_examples
        IOU_tr = TP / (TP + FP + FN)

        print("EPOCH " + str(i + 1) + ", Training Loss "
              + "{:.4f}".format(loss_tr) + ", IOU(pixel-level) for training "
              + "{:.4f}".format(IOU_tr))
        print()
        ls_tr = np.append(ls_tr, loss_tr)
        iu_tr = np.append(iu_tr, IOU_tr)

        loss_val = 0
        IOU_val = 0
        TP_val = 0
        FP_val = 0
        FN_val = 0

        for m in range(val_num_ex):
            val_mini_x = np.empty(shape=(1, 352, 1216, 3))
            val_mini_y = np.empty(shape=(1, 352, 1216))

            val_mini_x[0] = X_val[m]
            val_mini_y[0] = Y_val[m]
            loss_v = sess.run(loss, feed_dict={X: val_mini_x, Y: val_mini_y})
            prd_v = sess.run(logit, feed_dict={X: val_mini_x})

            prd_v[prd_v >= 0] = 1
            prd_v[prd_v < 0] = 0

            TP_v, FP_v, FN_v = IOU_eval(val_mini_y, prd_v)
            loss_val += loss_v
            TP_val += TP_v
            FP_val += FP_v
            FN_val += FN_v

        loss_val = loss_val / val_num_ex
        IOU_val = TP_val / (TP_val + FP_val + FN_val)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Loss = {:.4f}".format(loss_val) +
              ", Validation IOU = {:.4f}".format(IOU_val))
        print()
        ls_val = np.append(ls_val, loss_val)
        iu_val = np.append(iu_val, IOU_val)

    saver.save(sess, '/data/home/kardee/train_model.ckpt')
    print("Model saved")

    fig = plt.figure()
    t = np.arange(1, EPOCH + 1, 1)

    plt.title("Loss for Training Data")
    plt.plot(t, ls_tr, 'r')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.savefig('Loss_Training_curve.png')

    fig1 = plt.figure()
    plt.title("IOU for Training Data")
    plt.plot(t, iu_tr, 'b')
    plt.xlabel("Number of Epochs")
    plt.ylabel("IOU")
    plt.savefig('IOU_Training_curve.png')

    fig2 = plt.figure()

    plt.title("Loss for Validation Data")
    plt.plot(t, ls_val, 'r')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.savefig('Loss_Valid_curve.png')

    fig3 = plt.figure()
    plt.title("IOU for Validation Data")
    plt.plot(t, iu_val, 'b')
    plt.xlabel("Number of Epochs")
    plt.ylabel("IOU")
    plt.savefig('IOU_Valid_curve.png')

# Testing
with tf.Session() as sess:
    saver.restore(sess, '/data/home/kardee/train_model.ckpt')
    loss_tst = 0
    IOU_tst = 0
    tst_num_ex = X_test.shape[0]
    TP_tst = 0
    FP_tst = 0
    FN_tst = 0
    for j in range(tst_num_ex):
        tst_mini_x = np.empty(shape=(1, 352, 1216, 3))
        tst_mini_y = np.empty(shape=(1, 352, 1216))

        tst_mini_x[0] = X_test[j]
        tst_mini_y[0] = Y_test[j]
        loss_tes = sess.run(loss, feed_dict={X: tst_mini_x, Y: tst_mini_y})
        prd_tes = sess.run(logit, feed_dict={X: tst_mini_x})
        prd_tes[prd_tes >= 0] = 1
        prd_tes[prd_tes < 0] = 0

        TP_t, FP_t, FN_t = IOU_eval(tst_mini_y, prd_tes)
        loss_tst += loss_tes

        TP_tst += TP_t
        FP_tst += FP_t
        FN_tst += FN_t
        if j == 1 or j == 5 or j == 9 or j == 40 or j == 11 or j == 2:
            np.save('/data/home/kardee/' + 'predicted' + str(j) + '.npy', prd_tes)
            np.save('/data/home/kardee/' + 'actual' + str(j) + '.npy', tst_mini_y)
    loss_tst = loss_tst / tst_num_ex
    IOU_tst = TP_tst / (TP_tst + FP_tst + FN_tst)
    print("Test Loss " + "{:.4f}".format(loss_tst) +
          ", IOU(pixel-level) for Test Data  " + "{:.4f}".format(IOU_tst))
    print()
