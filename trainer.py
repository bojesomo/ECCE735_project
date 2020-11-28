import cv2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # cpu
from keras.models import load_model
import matplotlib.pyplot as plt
import argparse
import ipykernel
import pandas as pd
from sklearn.model_selection import train_test_split

from codebase.models.segnet import *
from codebase.models.pspnet import *
from codebase.models.unet import *
from codebase.models.fcn import *
from codebase.data_utils.utils import create_annotation_df

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='D:/KU_Works/Datasets/', help='location of datasets')
parser.add_argument('--dataset', type=str, default='OPIXray', help='dataset to use',
                    choices=['OPIXray', 'SIXray', 'GDXray', 'COMPASS-XP'])
parser.add_argument('--net-type', type=str, default='unet', help='the kind of network to use',
                    choices=['segnet', 'unet', 'pspnet', 'fcn_8', 'fcn_32'])
parser.add_argument('--base-model', type=str, default='densenet121', help='the kind of base (encoder) model to use',
                    choices=['resnet50', '', 'vgg', 'densenet121'])
args = parser.parse_args()

# data_root = '/home/farhanakram/Alabi/Datasets/'  # OPIXray/train/train_annotation/gt'
# dataset = 'OPIXray'

data_root = args.data_root
dataset = args.dataset
if args.base_model == 'none':
    args.base_model = ''

model_name = f'{(args.base_model + "_") if args.base_model else ""}{args.net_type}'
model_fcn = eval(model_name)

# annotation_path = os.path.join(data_root, dataset, 'train/train_annotation')
# annotations = os.path.join(annotation_path, 'annotations.csv')
# if not os.path.exists(annotations):
#     create_annotation_df(annotation_path)
# anno_df = pd.read_csv(annotations, dtype=str)
# train, test = train_test_split(anno_df, test_size=0.2, stratify=anno_df['class_id'], random_state=12345)
# # test.groupby('class_id').agg('count')

# model = segnet(n_classes=7,  input_height=576, input_width=768)
# model = vgg_segnet(n_classes=7,  input_height=576, input_width=768)
# model = resnet50_segnet(n_classes=7,  input_height=576, input_width=768)

model = model_fcn(n_classes=7,  input_height=576, input_width=768)
print(f'working with {model_name}')

model.train(
    train_images=os.path.join(data_root, dataset, 'scans/positives'),
    train_annotations=os.path.join(data_root, dataset, 'train/train_annotation/gt'),
    # val_images=f'scans/{dataset}_dataset/test/',
    # val_annotations=os.path.join(data_root, dataset, 'test/test_annotation/gt'),  # "trainingDataset/val_annotations/",
    validation_split=0.1,
    checkpoints_path=None, epochs=5, validate=True,
    batch_size=1, val_batch_size=1,
    verify_dataset=False,
    steps_per_epoch=512, validation_steps=200,
)
#
# model.train(
#     train_images =  "trainingDataset/train_images/",
#     train_annotations = "trainingDataset/train_annotations/",
# 	val_images =  "trainingDataset/val_images/",
#     val_annotations = "trainingDataset/val_annotations/",
#     checkpoints_path = None , epochs=15, validate=True
# )
#
# model.summary()

folder = os.path.join(data_root, dataset, 'scans/test/')  # "testingDataset/test_images/"
# for filename in os.listdir(folder):
#     out = model.predict_segmentation(inp=os.path.join(folder, filename),
#                                      # out_fname=os.path.join("testingDataset/segmentation_results/", filename)
#                                      )

result = model.evaluate_segmentation(inp_images_dir=folder,  # "testingDataset/test_images/",
                                     annotations_dir=os.path.join(data_root, dataset, 'test/test_annotation/gt'),
                                     )

print(result)
# folder = "testingDataset/test_images/"
# for filename in os.listdir(folder):
# 	out = model.predict_segmentation(inp=os.path.join(folder,filename),
# 	out_fname=os.path.join("testingDataset/segmentation_results/",filename))
#
# print(model.evaluate_segmentation( inp_images_dir="testingDataset/test_images/"  ,
# 	annotations_dir="testingDataset/test_annotations/" ) )


# # save model parameters used
# readme_file = os.path.join('Results.csv')
# opts_dict = vars(argparse.Namespace(**{'filename': args.args_in.model_name[:-3], 'num_params': num_params,
#                                        'val_mse': best_loss}, **vars(opts)))
# # opts_dict = vars(argparse.Namespace(**{'filename': model_name[:-3], 'num_params': num_params,
# #                                        'val_loss': best_loss}, **vars(opts)))
# opts_df = pd.DataFrame([opts_dict])
# if os.path.exists(readme_file):
#     opts_df.to_csv(readme_file, mode='a', index=False, header=False)
# else:
#     opts_df.to_csv(readme_file, mode='a', index=False)

# save model parameters used
readme_file = os.path.join('Results.csv')
opts_dict = vars(argparse.Namespace(**vars(args), **result))
opts_df = pd.DataFrame([opts_dict])
if os.path.exists(readme_file):
    opts_df.to_csv(readme_file, mode='a', index=False, header=False)
else:
    opts_df.to_csv(readme_file, mode='a', index=False)