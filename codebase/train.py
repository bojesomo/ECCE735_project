import argparse
import json
from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset, ImageSegmentationGen
import os
import glob
import six
import matplotlib.pyplot as plt
from keras import optimizers
from keras import backend as K
import keras
import tensorflow as tf
from keras.losses import binary_crossentropy

from .data_utils.utils import create_annotation_df
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from keras.utils import multi_gpu_model


def loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    
def customLoss(yTrue,yPred):
    dice = K.mean(1-(2 * K.sum(yTrue * yPred))/(K.sum(yTrue + yPred)))
    return K.mean(dice + keras.losses.categorical_crossentropy(yTrue,yPred))


def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid".format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files, key=lambda f: int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          validation_split=0.0,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          validation_steps=200,
          optimizer_name='adadelta',
		  do_augment=False, 
		  classifier=None
          ):

    #from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    #if isinstance(model, six.string_types):
        # create the model from the name
    #    assert (n_classes is not None), "Please provide the n_classes"
    #    if (input_height is not None) and (input_width is not None):
    #        model = model_from_name[model](
    #            n_classes, input_height=input_height, input_width=input_width)
    #    else:
    #        model = model_from_name[model](n_classes)

    num_gpus = len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU'])

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        # assert val_images is not None
        # assert val_annotations is not None
        if validation_split == 0.0:
            assert val_images is not None
            assert val_annotations is not None

#loss="categorical_crossentropy",#
    if optimizer_name is not None:
        model.compile(loss="categorical_crossentropy",  # loss=lambda yTrue, yPred: customLoss(yTrue, yPred),
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)
    '''additions by Alabi Bojesomo'''
    #####################################################
    if validation_split != 0.0:
        annotation_path = os.path.dirname(train_annotations)
        annotations = os.path.join(annotation_path, 'annotations.csv')
        if not os.path.exists(annotations):
            create_annotation_df(annotation_path)
        anno_df = pd.read_csv(annotations, dtype=str)
        train_sampler, val_sampler = train_test_split(anno_df, test_size=validation_split, stratify=anno_df['class_id'],
                                                      random_state=12345)
        val_images = train_images
        val_annotations = train_annotations
    else:
        train_sampler = None
        val_sampler = None
    ######################################################
    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images, train_annotations, n_classes)
        assert verified
        if validate:
            if validation_split == 0.0:
                print("Verifying validation dataset")
                verified = verify_segmentation_dataset(val_images, val_annotations, n_classes)
                assert verified

    train_gen = ImageSegmentationGen(
        train_images, train_annotations,  batch_size*num_gpus,  n_classes,
        input_height, input_width, output_height, output_width, do_augment=do_augment, sampler=train_sampler)

    if validate:
        val_gen = ImageSegmentationGen(
            val_images, val_annotations, val_batch_size*num_gpus,
            n_classes, input_height, input_width, output_height, output_width, sampler=val_sampler)

    if num_gpus > 1:
        model = multi_gpu_model(model, gpus=num_gpus)
        if optimizer_name is not None:
            model.compile(loss="categorical_crossentropy",  # loss=lambda yTrue, yPred: customLoss(yTrue, yPred),
                          optimizer=optimizer_name,
                          metrics=['accuracy'])
    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch, epochs=1, workers=1)
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))		  
            
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch,
                                          validation_data=val_gen,
                                          validation_steps=validation_steps, epochs=1,
                                          workers=1)
           
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)
