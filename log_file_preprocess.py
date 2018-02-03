import pandas as pd


######################
# 
# This script reads driving log file using pandas. Then reorganize the data 
# from the log to provide model.py the list of images to train, containing
# center, left, right and their flipped images for further processing.
#
# Note: this script does not actually do these. The processing of images happens 
# inside the generator in model.py.
#
######################
def preprocess(correction=.2):
    df = pd.read_csv('{}/driving_log.csv'.format('./data_sdc/'),
            names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    sample_df = pd.DataFrame()

    # preprocess log file
    sample_df['img_filename'] = pd.concat([df.center, df.left, df.right], ignore_index=True)
    sample_df['steering'] = pd.concat([df.steering, df.steering + correction,
        df.steering - correction], ignore_index=True)
    sample_df.img_filename = sample_df.img_filename.apply(lambda x: x.split('/')[-1])
    sample_df['flip'] = 1

    # flip
    tmp = sample_df.copy()
    tmp.flip *= -1
    sample_df = sample_df.append(tmp, ignore_index=True)

    return sample_df
