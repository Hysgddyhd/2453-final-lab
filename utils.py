from PIL import Image
import numpy as np
import scipy.misc as misc
import scipy.io as sio
import os
import pickle
from scipy.stats import truncnorm


def truncated_noise_sample(batch_size=1, dim_z=128, truncation=2., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


def read_cifar(data, labels, batch_size):
    rand_select = np.random.randint(0, 50000, [batch_size])
    batch = data[rand_select]
    batch_labels = labels[rand_select]
    return batch, batch_labels


def read_catdog(data, labels, batch_size):
    rand_select = np.random.randint(0, 11078, [batch_size])
    batch = data[rand_select]
    batch_labels = labels[rand_select]
    return batch, batch_labels


def read_imagenet(data, labels, batch_size):
    nums_data = labels.shape[0]
    rand_select = np.random.randint(0, nums_data, [batch_size])
    batch = data[rand_select]
    batch_labels = labels[rand_select]
    return batch, batch_labels


def read_face(data, batch_size):
    rand_select = np.random.randint(0, 13233, [batch_size])
    batch = data[rand_select]


    return batch, 0


# os.listdir("./dataset")
def random_batch_(path, batch_size, shape, c_nums):
    folder_names = os.listdir(path)
    rand_select = np.random.randint(0, folder_names.__len__())
    if not c_nums == folder_names.__len__():
        print("Error: c_nums must match the number of the folders")
        return
    y = np.zeros([1, c_nums])
    y[0, rand_select] = 1
    path = path + folder_names[rand_select] + "//"
    data = sio.loadmat(path + "dataset.mat")["data"]
    rand_select = np.random.randint(0, np.size(data, 0), [batch_size])
    batch = data[rand_select]
    return batch, y


def random_batch(path, batch_size, shape, c_nums):
    folder_names = os.listdir(path)
    rand_select = np.random.randint(0, folder_names.__len__())
    if not c_nums == folder_names.__len__():
        print("Error: c_nums must match the number of the folders")
        return
    y = np.zeros([1, 1])
    y[0, 0] = rand_select
    path = path + folder_names[rand_select] + "//"
    file_names = os.listdir(path)
    rand_select = np.random.randint(0, file_names.__len__(), [batch_size])
    batch = np.zeros([batch_size, shape[0], shape[1], shape[2]])
    for i in range(batch_size):
        img = np.array(Image.open(path + file_names[rand_select[i]]).resize([shape[0], shape[1]]))[:, :, :3]
        if img.shape.__len__() == 2:
            img = np.dstack((img, img, img))
        batch[i, :, :, :] = img
    return batch, y


def random_face_batch(path, batch_size):
    filenames_young = os.listdir(path+"0//")
    filenames_cats = os.listdir(path+"1//")
    rand_gender = np.random.randint(0, 2)
    batch = np.zeros([batch_size, 64, 64, 3])
    Y = np.zeros([1, 2])
    if rand_gender == 0:#young
        rand_samples = np.random.randint(0, filenames_young.__len__(), [batch_size])
        c = 0
        for i in rand_samples:
            img = np.array(Image.open(path+"0//"+filenames_young[i]))
            center_h = img.shape[0] // 2
            center_w = img.shape[1] // 2
            # batch[c, :, :, :] = misc.imresize(img[center_h - 70:center_h + 70, center_w - 70:center_w + 70, :], [64, 64])
            batch[c, :, :, :] = misc.imresize(img, [64, 64])
            c += 1
        Y[0, 0] = 1
    else:
        rand_samples = np.random.randint(0, filenames_cats.__len__(), [batch_size])
        c = 0
        for i in rand_samples:
            img = np.array(Image.open(path + "1//" + filenames_cats[i]))
            batch[c, :, :, :] = misc.imresize(img, [64, 64])
            c += 1
        Y[0, 1] = 1
    return batch, Y


def random_batch_(path, batch_size, shape):
    filenames = os.listdir(path)
    rand_samples = np.random.randint(0, filenames.__len__(), [batch_size])
    batch = np.zeros([batch_size, shape[0], shape[1], shape[2]])
    c = 0
    y = np.zeros([batch_size, 2])
    for idx in rand_samples:
        if (filenames[idx])[:3] == "cat":
            y[c, 0] = 1
        else:
            y[c, 1] = 1
        try:
            batch[c, :, :, :] = misc.imresize(crop(np.array(Image.open(path + filenames[idx]))), [shape[0], shape[1]])
        except:
            img = crop(np.array(Image.open(path + filenames[0])))
            batch[c, :, :, :] = misc.imresize(img, [shape[0], shape[1]])
        c += 1
    return batch, y


def crop(img):
    h = img.shape[0]
    w = img.shape[1]
    if h < w:
        x = 0
        y = np.random.randint(0, w - h + 1)
        length = h
    elif h > w:
        x = np.random.randint(0, h - w + 1)
        y = 0
        length = w
    else:
        x = 0
        y = 0
        length = h
    return img[x:x+length, y:y+length, :]


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def to_img(src_path, dst_path):
    filenames = os.listdir(src_path)
    for filename in filenames:
        data = unpickle(src_path + filename)
        imgs = data["data"]
        labels = data["labels"]
        for i in range(np.size(imgs, 0)):
            img = np.transpose(np.reshape(imgs[i, :], [3, 64, 64]), [1, 2, 0])
            if not os.path.exists(dst_path+str(labels[i])):
                os.mkdir(dst_path+str(labels[i]))
            Image.fromarray(np.uint8(img)).save(dst_path + str(labels[i]) + "//" + filename + "_" + str(labels[i]) + "_" + str(i) + ".jpg")
        print(filename)
