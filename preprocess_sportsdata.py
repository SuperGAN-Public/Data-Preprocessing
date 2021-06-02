#!/usr/bin/env python3

import pandas as pd
import numpy as np
import h5py

# Figures out the filename from a segment number,
# Since they're all 2 digits, with 1 digit numbers prefixed by 0.
def get_filename(segment_num):
    if segment_num < 10:
        return "s0%d.txt" % segment_num
    else:
        return "s%d.txt" % segment_num
        

# Processes a single segment file (e.g. sportsdata/data/a10/p1/s01.txt).
# Returns two arrays, one for each of the gyroscope and accelermeter data.
def process_file(path):
    # Get dataframe
    df = pd.read_csv(path, header=None)

    # Select accelerometer and gyroscope data
    gyr = np.array(df.loc[0:124, 27:29])
    acc = np.array(df.loc[0:124, 30:32])

    # Perform L2 norm
    gyr_l2 = np.linalg.norm(gyr)
    acc_l2 = np.linalg.norm(acc)

    # Return both sets, separated
    return (gyr / gyr_l2, acc / acc_l2)


# Processes every file for a particular activity (e.g. sportsdata/data/a10)
# Number is an integer representing which activity to process.
# Returns two arrays, one for each of the gyroscope and acceloermeter data.
def process_activity(number):
    gyr_data = []
    acc_data = []
    activity_dir = "sportsdata/data/a%d" % number

    for particpant in range(1, 9):
        partipant_dir = "%s/p%d" % (activity_dir, particpant)
        for segment in range(1, 61):
            filepath = "%s/%s" % (partipant_dir, get_filename(segment))
            gyr, acc = process_file(filepath)
            gyr_data.append(gyr)
            acc_data.append(acc)

    return (np.array(gyr_data), np.array(acc_data))


if __name__ == '__main__':
    gyr_x = []
    acc_x = []
    gyr_y = []
    acc_y = []
    gyr_y_onehot = []
    acc_y_onehot = []

    SEGMENTS = 60
    PARTICIPANTS = 8
    ACTIVITIES = 9

    SEGMENTS_PER_ACTIVITY = SEGMENTS * PARTICIPANTS
    TOTAL_SEGMENTS = SEGMENTS_PER_ACTIVITY * ACTIVITIES

    for activity in range(10, 19):
        # Generate appropriate label data
        label = activity - 10
        onehot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        onehot[label] = 1.0

        label_vec = [label] * SEGMENTS_PER_ACTIVITY
        gyr_y.extend(label_vec)
        acc_y.extend(label_vec)

        onehot_vec = np.tile(np.array(onehot), (SEGMENTS_PER_ACTIVITY, 1))
        gyr_y_onehot.append(onehot_vec)
        acc_y_onehot.append(onehot_vec)

        # Process all relevant files
        gyr, acc = process_activity(activity)
        gyr_x.append(gyr)
        acc_x.append(acc)

    # Format everything for exporting to H5 file
    gyr_x = np.reshape(np.array(gyr_x), (TOTAL_SEGMENTS, 125, 3))
    gyr_y = np.array(gyr_y, dtype=np.int32)
    gyr_y_onehot = np.reshape(np.array(gyr_y_onehot), (TOTAL_SEGMENTS, 9))

    acc_x = np.reshape(np.array(acc_x), (TOTAL_SEGMENTS, 125, 3))
    acc_y = np.array(gyr_y, dtype=np.int32)
    acc_y_onehot = np.reshape(np.array(acc_y_onehot), (TOTAL_SEGMENTS, 9))

    with h5py.File('sports_data_gyroscope.h5', 'w') as f:
        f['X'] = gyr_x
        f['y'] = gyr_y
        f['y_onehot'] = gyr_y_onehot

    with h5py.File('sports_data_accelerometer.h5', 'w') as f:
        f['X'] = acc_x
        f['y'] = acc_y
        f['y_onehot'] = acc_y_onehot
