# update: 8.14.2017
import os
import pickle

import hickle
import numpy as np
import tensorflow as tf
from natsort import natsorted
from scipy import ndimage
from keras.preprocessing import image

import cv2
from core.utils import *
from core.vggnet import Vgg19

run_on_server = 1

if run_on_server:
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))


def comp(x, y):
    x_num = int(x[:-4])
    y_num = int(y[:-4])
    if x_num > y_num:
        return 1
    if x_num < y_num:
        return -1
    if x_num == y_num:
        return 0


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'

    class K(object):
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0

    return K


def main():
    PATH = os.getcwd()
    vgg_model_path = PATH + '/data/imagenet-vgg-verydeep-19.mat'
    data_dir = '../Dataset/data/tobii/'
    num_of_image_per_video = 17
    type = ['test']
    # type = ['train', 'val', 'test']
    # TIME = str(datetime.now())
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for each in type:
            # settle down the paths
            if each == 'train':
                session = '0409-b'
            elif each == 'val':
                session = '0409-c'
            elif each == 'test':
                session = '0409-e'

            img_dir = '%s/frames/' % (data_dir + session)
            label_dir = '%s/label_all.txt' % (data_dir + session)
            path = PATH + '/data/data_set/' + each + '/'

            # # generate images_path
            images_list = natsorted([
                img_dir + file for file in os.listdir(img_dir)
                if file.endswith('.jpg')
            ])
            # cur_images_path = [vf + '/' + image for image in images_list]
            step = int(float(len(images_list)) / float(num_of_image_per_video))
            print(step)
            all_feats = np.ndarray([step, num_of_image_per_video, 196, 512],
                                   dtype=np.float32)

            # read images and extract features
            for i in range(step):
                print('Processing No.' + str(i + 1) + '/%d batch..' % step)
                cur_images_path = images_list[i * 17:i * 17 + 17]
                image_batch = []
                for img_file in cur_images_path:
                    img = image.load_img(img_file, target_size=[224, 224])
                    x = image.img_to_array(img)
                    image_batch.append(x)

                image_batch = np.array(image_batch).astype(np.float32)
                feats = sess.run(
                    vggnet.features, feed_dict={vggnet.images: image_batch})

                all_feats[i, :] = feats

            label = []
            with open(label_dir, 'r') as f:
                for line in open(label_dir):
                    line = f.readline().strip().split(',')
                    label.append(line[1])
            label_reshape = np.array(label)
            label_reshape = label_reshape[:step * 17].reshape(step, 17)
            filenames_new = np.array(list(range(step)))
            train_data = {
                'features': all_feats,
                'labels': label_reshape,
                'new_filename': filenames_new
            }
            # use hickle to save huge feature vectors
            with open(each + '_data_vgg' + '.pkl', 'wb') as f:
                pickle.dump(train_data, f)


main()
