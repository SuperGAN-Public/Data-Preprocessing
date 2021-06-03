#!/usr/bin/env python3

import pandas as pd
import numpy as np
import h5py
import os


# Constants used for vectorization
SENSOR_INDICES = {
    'M01' : 0, 'M02' : 1, 'M03' : 2, 'M04' : 3, 'M05' : 4,
    'M06' : 5, 'M07' : 6, 'M08' : 7, 'M09' : 8, 'M10' : 9,
    'M11' : 10, 'M12' : 11, 'M13' : 12, 'M14' : 13, 'M15' : 14,
    'M16' : 15, 'M17' : 16, 'M18' : 17, 'M19' : 18, 'M20' : 19, 
    'M21' : 20, 'M22' : 21, 'M23' : 22, 'M24' : 23, 'M25' : 24,
    'M26' : 25, 'I01' : 26, 'I02' : 27, 'I03' : 28, 'I04' : 29,
    'I05' : 30, 'I06' : 31, 'I07' : 32, 'I08' : 33, 'D01' : 34,
    'E01' : 35, 'AD1-A' : 36, 'AD1-B' : 37, 'AD1-C' : 38, 'asterisk' : 39,
}

HIGH_VALUES = ['ON', 'OPEN', 'PRESENT', 'START', 'START_INSTRUCT']
LOW_VALUES = ['OFF', 'CLOSE', 'ABSENT', 'END', 'STOP_INSTRUCT']
EPSILON = 0.0001


# Generates a vector of 40 floating point numbers from a single row
# from the dataset. Categorical values are given either a 1 or -1 value,
# while floating point values remain the same, modified by an epsilon.
# A value of 0 indicates this sensor was not used.
def vectorize(df_row):
    output = [0.0] * 40
    idx = SENSOR_INDICES[df_row[2]]
    val = df_row[3]

    if val in HIGH_VALUES:
        output[idx] = 1.0
    elif val in LOW_VALUES:
        output[idx] = -1.0
    else:
        num = float(val)
        if num < 0:
            output[idx] = num - EPSILON
        else:
            output[idx] = num + EPSILON

    return np.array(output)


# Extract the class label from the filename.
# If the file does not contain data to process, return None.
def get_file_label(filename):
    if filename[0] != 'p':
        return None
    else:
        return int(filename[5])


def get_features(filename):
    # Get dataframe
    df = pd.read_csv(filename, header=None, delimiter='\t')

    # Get a feature vector from every row
    vecs = []
    for _, row in df.iterrows():
        vecs.append(vectorize(row))

    # Create a sliding window over every sequence of 10 rows
    sliding_window = []
    for i in range(0, len(vecs) - 9):
        matrix = np.array(vecs[i : i+10])
        sliding_window.append(matrix)

    return sliding_window


if __name__ == '__main__':
    xs = []
    ys = []
    y_onehots = []

    for filename in os.listdir('adlnormal'):
        label = get_file_label(filename)
        if label is None:
            continue

        # Get features from the file in the form of sliding windows
        filepath = "adlnormal/%s" % filename
        features = get_features(filepath)
        xs.extend(features)

        # Append labels
        onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
        onehot[label - 1] = 1.0
        nwindows = len(features) # Number of sliding windows we generated
        ys.extend([label] * nwindows)
        y_onehots.extend([onehot] * nwindows)

    xs = np.array(xs)
    ys = np.array(ys, dtype=np.int32)
    y_onehots = np.array(y_onehots)

    with h5py.File('CASAS_adlnormal_dataset.h5', 'w') as f:
        f['X'] = xs
        f['y'] = ys
        f['y_onehot'] = y_onehots
