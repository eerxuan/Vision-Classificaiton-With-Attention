import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt


def add_attention(src, attention, color_start, color_end):
    h = src.shape[0]
    w = src.shape[1]
    attention = attention.reshape(14, 14)
    grid_size = 224 / 14

    # 归一化
    # for i in range(14):
    #     for j in range(14):
    #         if attention[i][j] != 0:
    #             attention[i][j] = -np.log10(attention[i][j])
    att_min, att_max = attention.min(), attention.max()
    # for i in range(14):
    #     for j in range(14):
    #         if attention[i][j] != 0:
    #             attention[i][j] = att_max - attention[i][j] + 5
    attention = (attention - att_min) / (att_max - att_min)

    # 创建一幅与原图片一样大小的透明图片
    grad_img = np.ndarray(src.shape, dtype=np.uint8)

    # # opencv 默认采用 BGR 格式而非 RGB 格式
    # g_b = float(color_end[0] - color_start[0]) / h
    # g_g = float(color_end[1] - color_start[1]) / h
    # g_r = float(color_end[2] - color_start[2]) / h

    for i in range(h):
        for j in range(w):
            m = int(i // grid_size)
            n = int(j // grid_size)
            grad_img[i, j, 0] = attention[m][n] * (
                color_end[0] - color_start[0])
            grad_img[i, j, 1] = attention[m][n] * (
                color_end[1] - color_start[1])
            grad_img[i, j, 2] = attention[m][n] * (
                color_end[2] - color_start[2])

    return grad_img


if __name__ == '__main__':
    idx_of_batch = 2
    # idx_of_video = 14

    label_dir = '../../data/data/tobii/0409-e/label_all.txt'
    data_path = './data/data_set/test/model_test_result/'

    with open('labels_to_idx.pkl', 'rb') as handle:
        labels_to_idx = pickle.load(handle)
    print(labels_to_idx)

    label = []
    with open(label_dir, 'r') as f:
        for line in open(label_dir):
            line = f.readline().strip().split(',')
            label.append(line[1])

    for idx_of_video in range(15):
        with open(data_path + 'video_id_test_%d.pkl' % idx_of_batch,
                  'rb') as handle:
            video_id_test = pickle.load(handle)

        with open(data_path + 'gen_idxs_test_%d.pkl' % idx_of_batch,
                  'rb') as handle:
            gen_idxs_test = pickle.load(handle)
            gen_idxs_test = np.array(gen_idxs_test)
            # gen_idxs_test = np.transpose(gen_idxs_test)

        with open(data_path + 'alpha_test_%d.pkl' % idx_of_batch,
                  'rb') as handle:
            alpha_test = pickle.load(handle)

        # print(gen_idxs_test.shape)
        # print(video_id_test.shape)
        # print(alpha_test.shape)

        frame_path = "../../data/data/tobii/0409-e/frames/"

        plt.figure(num='video%d' % idx_of_video, figsize=(15, 7))

        for idx_of_frame in range(0, 17):

            img = cv2.imread(frame_path + '%05d.jpg' %
                             (2340 + video_id_test[idx_of_video] * 17 * 3 +
                              idx_of_frame * 3))
            predict_label = gen_idxs_test[idx_of_video][idx_of_frame]
            if predict_label == 0:
                predict_label = 'brick'
            elif predict_label == 1:
                predict_label = 'grass'
            elif predict_label == 2:
                predict_label = 'stair'
            elif predict_label == 3:
                predict_label = 'tile'

            # print('predicted label: %d' % predict_label)

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            att_of_frame = alpha_test[idx_of_video][idx_of_frame]
            grad_img = add_attention(img, att_of_frame, (0, 0, 0),
                                     (0, 255, 255))
            blend = cv2.addWeighted(img, 1.0, grad_img, 0.6, 0.0)

            blend = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
            plt.subplot(3, 6, idx_of_frame + 1)
            plt.title('%d-%s-%s' % (idx_of_frame,
                                    (label[video_id_test[idx_of_video] * 17 +
                                           idx_of_frame]), predict_label))
            plt.imshow(blend)
            plt.axis('off')

        plt.savefig('./data/data_set/visualize/video%d-%d' % (idx_of_batch,
                                                              idx_of_video))
        # plt.show()
