# update: 8.14.2017
from keras.backend.tensorflow_backend import set_session
from tensorflow import ConfigProto, Session
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_pickle
import numpy as np
import pickle

current_path = './data/'

run_on_server = 1
train = 0
validate = 0
test = 1
train_for_test = 0

if run_on_server:
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(Session(config=config))


def main():
    # train_data = load_data(current_path + 'data_set/', 'test')
    # length = len(train_data['video_ids'])
    # train_data['features'] = train_data['features'][:int(0.7 * length)]
    # train_data['labels'] = train_data['labels'][:int(0.7 * length)]
    # train_data['video_ids'] = train_data['video_ids'][:int(0.7 * length)]
    # train_data['video_filenames'] = train_data['video_filenames'][:int(0.7 * length)]
    with open('train_data_vgg.pkl', 'rb') as handle:
        train_data = pickle.load(handle)
    # length = len(train_data['new_filename'])
    train_data['features'] = train_data['features']
    train_data['labels'] = train_data['labels']
    train_data['video_ids'] = train_data['new_filename']
    # train_data['video_filenames'] = train_data['video_filenames'][:int(0.7 * length)]
    if train_for_test == 1:
        with open('val_data_vgg.pkl', 'rb') as handle:
            val_data = pickle.load(handle)
        # length = len(train_data['new_filename'])
        train_data['features'] = np.concatenate(
            (train_data['features'], val_data['features']), axis=0)
        train_data['labels'] = np.concatenate(
            (train_data['labels'], val_data['labels']), axis=0)
        train_data['video_ids'] = np.concatenate(
            (train_data['new_filename'], val_data['new_filename']), axis=0)
    # train_data = {}

    data = {'train_data': train_data}
    # label_to_idx = load_pickle(current_path + 'data_set/label_to_idx.pkl')
    label_to_idx = load_pickle('labels_to_idx.pkl')
    num_images_per_video = 17

    model = CaptionGenerator(
        label_to_idx=label_to_idx,
        # dim_feature=[49, 1280],
        dim_feature=[196, 512],
        dim_hidden=1024,
        n_time_step=num_images_per_video,
        ctx2out=True,
        alpha_c=1.0,
        selector=True,
        dropout=False)

    solver = CaptioningSolver(
        model,
        data,
        n_epochs=300,
        batch_size=15,
        update_rule='adam',
        learning_rate=0.0006,
        print_every=3,
        save_every=10,
        pretrained_model=None,
        model_path=current_path + 'model/lstm/',
        test_model=current_path + 'model/lstm/model-310',
        log_path=current_path + 'log/',
        data_path=current_path + '/data_set/',
        test_result_save_path=current_path +
        'data_set/test/model_test_result/',
        models_val_disp=current_path + 'model/lstm/models_accuracy_val.txt')

    if train == 1:
        solver.train()
    if validate == 1:
        solver.all_model_val()
    if test == 1:
        solver.test()

    if train_for_test == 1:
        solver.train()


if __name__ == "__main__":
    main()
