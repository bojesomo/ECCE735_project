import numpy as np
import pandas as pd
import os


def create_annotation_df(annotations_path):
    files = [x for x in os.listdir(annotations_path) if x.endswith('.txt')]
    anno_df = pd.DataFrame(columns=['file_id', 'class_id'])
    for each_file in files:
        with open(os.path.join(annotations_path, each_file), 'r') as f:
            dd = f.read()
            dd = dd.replace('.jpg', '')
            anno_df = anno_df.append(dict(zip(anno_df.columns, dd.split()[:2])), ignore_index=True)
    anno_df.to_csv(os.path.join(annotations_path, 'annotations.csv'), index=False)
