import collections
import pickle

import keras.backend as K
import numpy as np
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.backend import repeat_elements
from keras.layers import (LSTM, Activation, Dense, Dropout, Input, Lambda,
                          TimeDistributed, concatenate)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects
from tensorflow.contrib import distributions

import cv2


def create_bayesian_mlp_model(n_classes, fea_dim):
    input_tensor = Input(shape=(fea_dim, ))
    x = Dropout(0.1)(input_tensor)
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5))(x)
    x = Dropout(0.1)(x)
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5))(x)
    x = Dropout(0.1)(x)
    logits = Dense(
        n_classes,
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5),
        name='logits')(x)
    softmax_output = Activation('softmax', name='softmax_output')(logits)
    variance = Dense(1, activation='softplus', name='variance')(x)
    # variance = Dense(1, activation='linear', name='variance')(x)
    logits_variance = concatenate([logits, variance], name='logits_variance')
    model = Model(
        inputs=input_tensor,
        outputs=[logits_variance, variance, softmax_output])
    return model


def create_traditional_mlp(n_classes, fea_dim, pred_num=1):
    input_tensor = Input(shape=(fea_dim, ))
    x = Dropout(0.05)(input_tensor)
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5))(x)
    x = Dropout(0.05)(x)
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5))(x)
    x = Dropout(0.05)(x)
    pred = Dense(
        n_classes,
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5),
        activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=pred)
    return model


def create_bayesian_lstm_model(look_back,
                               n_classes,
                               batch_size=1,
                               fea_dim=1,
                               pred_num=1,
                               stateful=False):
    if stateful:
        input_tensor = Input(batch_shape=(batch_size, look_back, fea_dim))
    else:
        input_tensor = Input(shape=(look_back, fea_dim))
    x = TimeDistributed(
        Dense(
            128,
            activation='relu',
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5)))(input_tensor)
    x = Dropout(0.05)(x)
    x = LSTM(
        128,
        dropout=0.05,
        recurrent_dropout=0.05,
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5),
        stateful=stateful)(x)
    x = Dropout(0.05)(x)
    logits = Dense(
        n_classes,
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5),
        name='logits')(x)
    softmax_output = Activation('softmax', name='softmax_output')(logits)
    variance = Dense(1, activation='softplus', name='variance')(x)
    logits_variance = concatenate([logits, variance], name='logits_variance')
    model = Model(
        inputs=input_tensor,
        outputs=[logits_variance, variance, softmax_output])
    return model


def create_traditional_lstm(look_back,
                            n_classes,
                            batch_size=1,
                            fea_dim=1,
                            pred_num=1,
                            stateful=False):
    if stateful:
        input_tensor = Input(batch_shape=(batch_size, look_back, fea_dim))
    else:
        input_tensor = Input(shape=(look_back, fea_dim))
    x = TimeDistributed(
        Dense(
            128,
            activation='relu',
            kernel_regularizer=l2(1e-5),
            bias_regularizer=l2(1e-5)))(input_tensor)
    x = Dropout(0.05)(x)
    x = LSTM(
        128,
        dropout=0.05,
        recurrent_dropout=0.05,
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5),
        stateful=stateful)(x)
    x = Dropout(0.05)(x)
    pred = Dense(
        n_classes,
        kernel_regularizer=l2(1e-5),
        bias_regularizer=l2(1e-5),
        activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=pred)
    return model


def predictive_entropy(pred_T):
    softmax_T = K.softmax(pred_T[:, :, :-1])
    softmax_mean = K.mean(softmax_T, axis=1)
    model_entropy_var = -1 * K.sum(K.log(softmax_mean) * softmax_mean, axis=-1)
    data_vars = pred_T[:, :, -1]
    return [softmax_T, data_vars, model_entropy_var]


def bayesian_categorical_crossentropy(n_MC, n_classes):
    def bayesian_categorical_crossentropy_internal(true, pred_var):
        std = K.square(K.exp(pred_var[:, n_classes]))
        # std = K.sqrt(K.epsilon() + K.exp(pred_var[:, -1]))
        pred = pred_var[:, 0:n_classes]
        iterable = K.variable(np.ones(n_MC))
        dist = distributions.Normal(
            loc=K.zeros_like(std), scale=K.ones_like(std))
        monte_carlo_results = K.map_fn(
            gaussian_categorical_crossentropy(true, pred, dist, std,
                                              n_classes),
            iterable,
            name='monte_carlo_results')
        variance_loss = K.mean(monte_carlo_results, axis=0)
        return variance_loss

    return bayesian_categorical_crossentropy_internal


def gaussian_categorical_crossentropy(true, pred, dist, std, n_classes):
    def map_fn(i):
        std_samples = K.transpose(std * dist.sample(n_classes))
        print(std_samples.shape, pred.shape)
        distorted_loss = K.categorical_crossentropy(
            true, pred + std_samples, from_logits=True)
        # diff = undistorted_loss - distorted_loss
        return distorted_loss

    return map_fn


def output_l1_regularizer(true, pred):
    return K.sum(K.abs(pred))


class feature_extraction_engine:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.create_encoder_model()
        self.output_shape = self.model.output.shape[1:]

    def create_encoder_model(self):
        input_tensor = Input(shape=self.input_shape)
        base_model = MobileNetV2(
            include_top=False, pooling='avg', input_tensor=input_tensor)
        for layer in base_model.layers:
            layer.trainable = False
        output_tensor = base_model.get_layer('out_relu').output
        model = Model(inputs=input_tensor, outputs=output_tensor)
        model.summary()
        return model

    def run(self, img_path=None, img=None):
        if img is None and img_path is None:
            return None
        elif img is None:
            img = image.load_img(img_path, target_size=self.input_shape[:2])
            x = image.img_to_array(img)
        else:
            img = cv2.resize(
                img, self.input_shape[:2], interpolation=cv2.INTER_CUBIC)
            x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)

    def get_output_shape(self):
        return self.output_shape


class uncertainty_engine():
    def __init__(self, model_path, para_path):
        para = pickle.load(open(para_path, "rb"))
        self.n_MC = para['n_MC']
        self.n_classes = para['n_classes']
        get_custom_objects().update({
            "bayesian_categorical_crossentropy_internal":
            bayesian_categorical_crossentropy(self.n_MC, self.n_classes),
            "output_l1_regularizer":
            output_l1_regularizer
        })
        self.model = load_model(model_path)
        self.epistemic_model = self.build_epistemic_model(
            self.model, para['n_MC'], para['n_classes'])

    def dropout_sampling(self, X):
        epistemic_result = self.epistemic_model.predict(
            np.expand_dims(X, axis=1), batch_size=1)

        aleatoric_result = self.model.predict(X, batch_size=1)
        return epistemic_result, aleatoric_result

    def build_epistemic_model(self, model, n_MC, n_classes):
        inpt = Input(shape=((1, ) + model.input_shape[1:]))
        x = Lambda(lambda x: repeat_elements(x, n_MC, 1), name='repeat')(inpt)
        hacked_model = Model(inputs=model.inputs, outputs=model.outputs[0])
        pred_T = TimeDistributed(hacked_model, name='epistemic_mc')(x)
        results = Lambda(
            lambda x: predictive_entropy(x), name='pred_var')(pred_T)
        epistemic_model = Model(inputs=inpt, outputs=results)
        return epistemic_model

    def cal_bnn_var(self, softmax_T, n_classes):
        preds = np.argmax(softmax_T, axis=-1)
        batch_size = softmax_T.shape[0]
        n_MC = softmax_T.shape[1]
        model_ratio_val = np.zeros((batch_size, n_classes))
        for i in range(batch_size):
            preds = np.argmax(softmax_T[i, :, :], axis=-1)
            b = collections.Counter(preds)
            for key in b:
                model_ratio_val[i, key] = float(b[key]) / n_MC
        softmax_mean = np.mean(softmax_T, axis=1)
        model_entropy_var = -1 * np.sum(
            np.multiply(np.log(softmax_mean), softmax_mean), axis=-1)
        sum_c_mean_T = np.mean(
            np.sum(np.multiply(np.log(softmax_T), softmax_T), axis=-1), axis=1)
        model_mi_var = model_entropy_var + sum_c_mean_T

        return model_ratio_val.reshape(
            (batch_size, n_classes)), model_entropy_var.reshape(
                (batch_size, 1)), model_mi_var.reshape((batch_size,
                                                        1)), softmax_mean

    def run(self, X):
        results = self.dropout_sampling(X)
        epistemic_result = results[0]
        # aleatoric_result = results[1]
        softmax_T = epistemic_result[0]
        data_var = np.mean(epistemic_result[1]).reshape((1, 1))
        model_ratio_var, model_entropy_var, model_mi_var, softmax = self.cal_bnn_var(
            softmax_T, self.n_classes)
        model_entropy_var = model_entropy_var.reshape((1, 1))
        uncertainties = np.concatenate(
            (data_var, model_entropy_var, model_mi_var), axis=1)
        return uncertainties, softmax


class prob_calibration_engine():
    def __init__(self, model_path, para_path):
        self.model = load_model(model_path)
        para = pickle.load(open(para_path, "rb"))
        self.means = para['means']
        self.vars = para['vars']

    def run(self, X):
        post_X = np.divide(np.subtract(X, self.means), self.vars**0.5)
        return self.model.predict(post_X)


class pipeline():
    def __init__(self, bnn_model, bnn_para, cali_model, cali_para):
        K.set_learning_phase(0)
        self.Fea_Eng = feature_extraction_engine((224, 224, 3))
        self.ProbCali_Eng = prob_calibration_engine(cali_model, cali_para)
        K.set_learning_phase(1)
        self.Uncertain_Eng = uncertainty_engine(bnn_model, bnn_para)

    def run(self, image):
        fea = self.Fea_Eng.run(img_path=None, img=image)
        uncertainties, softmax = self.Uncertain_Eng.run(fea)
        prob = self.ProbCali_Eng.run(uncertainties)
        pred = np.squeeze(np.argmax(softmax, axis=-1))
        return pred, prob, softmax
