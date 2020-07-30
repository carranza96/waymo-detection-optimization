from utils_tf_record.read_dataset_utils import read_and_parse_sharded_dataset
import os
import itertools
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from numpy.polynomial.polynomial import polyfit
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
import pyximport

# pyximport.install()
# from compute_overlap import compute_overlap

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.compat.v1.enable_eager_execution()

FILENAME_PATTERN = "data/camera_data_lidarGB/training/*"
OUTPUT_FILE = 'boxes_all.csv'

GENERATE_DATASET = False


def generate_dataset():
    dataset = read_and_parse_sharded_dataset(FILENAME_PATTERN)

    def generate(x):
        segment_id = x['image/source_id']
        image_camera_name = x['image/camera_name']
        timestamp_micros = x['image/frame_timestamp_micros']
        source_id = segment_id + "_".encode() + str(image_camera_name).encode() + "_".encode() + str(
            timestamp_micros).encode()
        image_height = tf.cast(x['image/height'], tf.float32)
        image_width = tf.cast(x['image/width'], tf.float32)

        labels = tf.cast(x['image/object/class/label'], tf.float32)
        xmax = tf.cast(x['image/object/bbox/xmax'], tf.float32) * image_width
        xmin = tf.cast(x['image/object/bbox/xmin'], tf.float32) * image_width
        ymax = tf.cast(x['image/object/bbox/ymax'], tf.float32) * image_height
        ymin = tf.cast(x['image/object/bbox/ymin'], tf.float32) * image_height

        boxes = tf.stack([labels, xmax, xmin, ymax, ymin], axis=1)

        def _generator(x):
            _features = {
                'source_id': source_id,
                'label': tf.slice(x, (0,), (1,)),
                'xmax': tf.slice(x, (1,), (1,)),
                'xmin': tf.slice(x, (2,), (1,)),
                'ymax': tf.slice(x, (3,), (1,)),
                'ymin': tf.slice(x, (4,), (1,)),
                'image_width': [image_width],
                'image_height': [image_height]
            }
            return _features

        boxes = tf.data.Dataset.from_tensor_slices(boxes).map(_generator)
        return boxes

    dataset = dataset.flat_map(generate)

    csv_headers = ','.join(
        ['source_id', 'label', 'xmax', 'xmin', 'ymax', 'ymin', 'image_width', 'image_height'], ) + '\n'
    with open(OUTPUT_FILE, "w") as text_file:
        text_file.write(csv_headers)

    def write_to_csv(input, index, sep):
        if index < 0:
            v = 'NO_ID'
        else:
            v = str(input[index])
        v = v + sep.decode()
        with open(OUTPUT_FILE, "a") as text_file:
            text_file.write(v)
        return v

    def reduce_to_csv(csv_lines, example):
        keys = [
            ('source_id', -1, ','),
            ('label', 0, ','),
            ('xmax', 0, ','),
            ('xmin', 0, ','),
            ('ymax', 0, ','),
            ('ymin', 0, ','),
            ('image_width', 0, ','),
            ('image_height', 0, '\n'),
        ]
        for i in range(len(keys)):
            k, index, sep = keys[i]
            _ = tf.numpy_function(write_to_csv, [example[k], index, sep], tf.string)

        # csv_lines = csv_lines + ','.join([str(tf.numpy_function(write_to_csv, [example[k]], tf.string)) if k is not 'source_id' else str(example[k]) for k in keys]) + '\n'
        return csv_lines

    csv_lines = dataset.reduce('', reduce_to_csv)


if GENERATE_DATASET:
    generate_dataset()

sample_frac = 1

print('Read csv')
df = pd.read_csv(OUTPUT_FILE)
print(df.shape[0], 'objects.')
# _ = df.pop('source_id')
print(df.head())

anchor_base = 256
aspect_ratios = [0.5, 1, 2]  # [0.3, 1, 1.5, 2] #[0.5, 1, 2]
scales = [0.25, 0.5, 1, 2]  # [0.005, 0.01, 0.5, 1, 2, 8, 13]#[0.25, 0.5, 1, 2]


def closer_to(data, keys):
    def _cl(x):
        return min(keys, key=lambda k: abs(k - x))

    return np.vectorize(_cl)(data)


# print('\nGenerate new attributes')
# df['xcenter'] = (df['xmax'] + df['xmin']) / 2
# df['ycenter'] = (df['ymax'] + df['ymin']) / 2
# df['width'] = df['xmax'] - df['xmin']
# df['height'] = df['ymax'] - df['ymin']
# df['aspect_ratio'] = df['width'] / df['height']
# df['aspect_ratio_closer'] = closer_to(df['aspect_ratio'], aspect_ratios)
df['scale'] = df['height'] * np.sqrt(df[
                                         'aspect_ratio']) / anchor_base  # df['height'] * np.sqrt(df['aspect_ratio_closer']) / anchor_base  # df['height'] * np.sqrt(df['aspect_ratio']) / anchor_base
# df['scale_closer'] = closer_to(df['scale'], scales)
#
# print('and remove old ones')
# _ = [df.pop(k) for k in ['xmax', 'xmin', 'ymax', 'ymin']]
#
# print("Remove useless anchors")
#
# print("")

# useless = df[(df['height'] < 16) | (df['width'] < 16)]
# print(len(useless), 'useless')
# df = df[(df['height'] > 5) & (df['width'] > 5)]

# df = df.sample(frac=sample_frac)
# print(df.shape[0], 'sample objects.')
#
# print(df.head())

colors = {1.: 'red', 2.: 'green', 3.: 'blue'}
boxlabels = {1.: 'VEHICLE', 2.: 'PEDESTRIAN', 3.: 'CYCLIST'}


def study_width_height_position():
    # fig, ax = plt.subplots(4, 1, figsize=(5, 10))
    # ax[0].hist(df.width, bins=800)
    # ax[1].hist(df.width, bins=125, range=(0, 250))
    # ax[2].hist(df.width, bins=25, range=(0, 25))
    # ax[3].hist(df.width, bins=5, range=(0, 5))
    # ax[0].set_title("Width histogram")
    # fig.show()
    # fig, ax = plt.subplots(4, 1, figsize=(5, 10))
    # ax[0].hist(df.height, bins=800)
    # ax[1].hist(df.height, bins=125, range=(0, 250))
    # ax[2].hist(df.height, bins=25, range=(0, 25))
    # ax[3].hist(df.height, bins=5, range=(0, 5))
    # ax[0].set_title('Height histogram')
    # fig.show()

    h = 1.54  # 1.65 Height of the camera from the ground
    H = np.mean([1280, 886])  # 375# Height of the image
    Hv = 1.6  # Average height of the vehicles in the real world
    max_var_v = 0.4  # 0.4 Max variation of the height of the vehicle
    f = 721.54 / 512  # f = Focal length
    p = 1 / 1280  # p = Size of each pixel
    f_p = 721.54  # f/p
    alpha = 2  # 2 Maximunm relative pitch angle between the camera and the ground plane

    # p: Size of each pixel
    # Hb: Height of the bounding box
    def v(Hb, H=H):
        Hb = np.asarray(Hb)
        return ((h - Hv / 2) / Hv) * Hb + H / 2

    def v_min(Hb, H=H):
        Hb = np.asarray(Hb)
        return ((h - (Hv + max_var_v) / 2) / (Hv + max_var_v)) * Hb - math.tan(math.radians(alpha)) * f_p + H / 2

    def v_max(Hb, H=H):
        Hb = np.asarray(Hb)
        return ((h - (Hv - max_var_v) / 2) / (Hv - max_var_v)) * Hb + math.tan(math.radians(alpha)) * f_p + H / 2

    # b, m = polyfit(df['height'], df['ycenter'], 1)
    # # plt.plot([0,1280], 1/2*np.asarray([0,1280]), linestyle='--', c='r', alpha=0.5)
    # # plt.plot([0,1280], -1/2*np.asarray([0,1280])+1280, linestyle='--', c='r', alpha=0.5)
    # fig = plt.scatter(x=df['height'], y=df['ycenter'], s=0.01)
    # plt.plot([0, 1000], b + m * np.asarray([0, 1000]), '-', c='grey', label='reg. line')
    # # plt.plot([0, 1200],v([0, 1200]), c='green', label='v')
    # # plt.plot([0, 1200],v_min([0, 1200]), c='orange', label='vmin')
    # # plt.plot([0, 1000],v_max([0, 1000]), c='orange', label='vmax')
    # plt.xlabel("Height")
    # plt.ylabel("Vertical Position")
    # plt.legend()
    # plt.show()

    b1, m1 = polyfit(df[df['image_height'] == 1280]['height'], df[df['image_height'] == 1280]['ycenter'], 1)
    fig = plt.figure()
    plt.scatter(x=df[df['image_height'] == 1280]['height'], y=df[df['image_height'] == 1280]['ycenter'], s=0.001)
    plt.plot([0, 1200], b1 + m1 * np.asarray([0, 1200]), '--', c='grey', label='reg. line')
    # plt.plot([0, 1200],v([0, 1200], H=1280), c='green', label='v')
    # plt.plot([0, 1200],v_min([0, 1200], H=1280), c='orange', label='vmin')
    # plt.plot([0, 1000],v_max([0, 1000], H=1280), c='orange', label='vmax')
    # print(plt.ylim(), plt.xlim())
    # plt.ylim(top=1300)
    # print(plt.ylim())
    plt.axis('tight')
    plt.xlabel("Object Height", fontsize=14)
    plt.ylabel("Vertical Position", fontsize=14)
    plt.title('Frontal cameras (1920 x 1280)', fontsize=16)
    plt.legend(loc='upper right')
    fig.savefig('frontal_camera.png', format='png', dpi=500, bbox_inches='tight')

    plt.show()
    # print((df['image_height'] == 1280).count())

    b, m = polyfit(df[df['image_height'] != 1280]['height'], df[df['image_height'] != 1280]['ycenter'], 1)
    # fig2 = plt.figure(figsize=(0.5, 0.5))
    fig2 = plt.figure()
    plt.scatter(x=df[df['image_height'] != 1280]['height'], y=df[df['image_height'] != 1280]['ycenter'], s=0.001)
    plt.plot([0, 800], b + m * np.asarray([0, 800]), '--', c='grey', label='reg. line')
    # plt.plot([0, 886],v([0, 886], H=886), c='green', label='v')
    # plt.plot([0, 886],v_min([0, 886], H=886), c='orange', label='vmin')
    # plt.plot([0, 886],v_max([0, 886], H=886), c='orange', label='vmax')
    plt.xlabel("Height", fontsize=14)
    plt.ylabel("Vertical Position", fontsize=14)
    plt.title('Lateral cameras (1920 x 886)', fontsize=16)
    plt.legend()
    plt.show()
    fig2.savefig('lateral_camera.png', format='png', dpi=500, bbox_inches='tight')

    b, m = polyfit(df[df['image_height'] != 1280]['width'], df[df['image_height'] != 1280]['ycenter'], 1)
    # fig2 = plt.figure(figsize=(0.5, 0.5))
    fig2 = plt.figure()
    plt.scatter(x=df[df['image_height'] != 1280]['width'], y=df[df['image_height'] != 1280]['ycenter'], s=0.001)
    plt.plot([0, 800], b + m * np.asarray([0, 800]), '-', c='grey', label='reg. line')
    # plt.plot([0, 886],v([0, 886], H=886), c='green', label='v')
    # plt.plot([0, 886],v_min([0, 886], H=886), c='orange', label='vmin')
    # plt.plot([0, 886],v_max([0, 886], H=886), c='orange', label='vmax')
    plt.xlabel("Width", fontsize=14)
    plt.ylabel("Vertical Position", fontsize=14)
    plt.title('Lateral cameras (1920 x 886)', fontsize=16)
    plt.show()

    b, m = polyfit(df[df['image_height'] == 1280]['width'], df[df['image_height'] == 1280]['ycenter'], 1)
    # fig2 = plt.figure(figsize=(0.5, 0.5))
    fig2 = plt.figure()
    plt.scatter(x=df[df['image_height'] == 1280]['width'], y=df[df['image_height'] == 1280]['ycenter'], s=0.001)
    plt.plot([0, 800], b + m * np.asarray([0, 800]), '-', c='grey', label='reg. line')
    # plt.plot([0, 886],v([0, 886], H=886), c='green', label='v')
    # plt.plot([0, 886],v_min([0, 886], H=886), c='orange', label='vmin')
    # plt.plot([0, 886],v_max([0, 886], H=886), c='orange', label='vmax')
    plt.xlabel("Width", fontsize=14)
    plt.ylabel("Vertical Position", fontsize=14)
    plt.title('Lateral cameras (1920 x 886)', fontsize=16)
    plt.show()

    x, y = df['height'] / df['image_height'], df['ycenter'] / df['image_height']
    b, m = polyfit(x, y, 1)
    fig = plt.scatter(x=x, y=y, s=0.01)
    plt.plot([0, 1], b + m * np.asarray([0, 1]), '-', c='grey', label='reg. line')
    plt.xlabel("Height (norm)")
    plt.ylabel("Vertical Position (norm)")
    plt.legend()
    plt.show()

    # fig = plt.scatter(x=df['height'], y=df['ycenter'], c=df['label'].apply(lambda x: colors[x]), marker='+', s=1)
    # plt.xlabel("Height")
    # plt.ylabel("Vertical Position")
    # plt.legend()
    # plt.show()

    fig = plt.scatter(x=df['width'], y=df['ycenter'], marker='+', s=1)
    plt.xlabel("Width")
    plt.ylabel("Vertical Position")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(5, 11))
    for i in range(3):
        _df = df[df['label'] == (i + 1.)]
        ax[i].scatter(x=_df['height'], y=_df['ycenter'], c=_df['label'].apply(lambda x: colors[x]), marker='+', s=1)
        ax[i].set_xlabel("Height")
        ax[i].set_ylabel("Vertical Position")
        ax[i].set_title(boxlabels[i + 1.])
    fig.suptitle('All images', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=.075)
    fig.show()

    fig, ax = plt.subplots(3, 1, figsize=(5, 11), sharex=True, sharey=True)
    for i in range(3):
        _df = df[df['label'] == (i + 1.)]
        _df = _df[_df['image_height'] == 1280]
        ax[i].scatter(x=_df['height'], y=_df['ycenter'], c=_df['label'].apply(lambda x: colors[x]), marker='+', s=1)
        ax[i].set_xlabel("Height")
        ax[i].set_ylabel("Vertical Position")
        ax[i].set_title(boxlabels[i + 1.])
    fig.suptitle('Images (1920 x 1280)', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=.075)
    fig.show()

    fig, ax = plt.subplots(3, 1, figsize=(5, 11), sharex=True, sharey=True)
    for i in range(3):
        _df = df[df['label'] == (i + 1.)]
        _df = _df[_df['image_height'] != 1280]
        ax[i].scatter(x=_df['height'], y=_df['ycenter'], c=_df['label'].apply(lambda x: colors[x]), marker='+', s=1)
        ax[i].set_xlabel("Height")
        ax[i].set_ylabel("Vertical Position")
        ax[i].set_title(boxlabels[i + 1.])
    fig.suptitle('Images (1920 x 886)', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=.075)
    fig.show()

    fig, ax = plt.subplots(2, 3, figsize=(12, 9))
    for i in range(3):
        l = boxlabels[i + 1]
        _df = df[df['label'] == (i + 1.)]
        ax[0, i].hist(_df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=colors[i + 1], density=True)
        ax[1, i].hist(_df.scale, bins=100, range=(0, 3), alpha=0.7, color=colors[i + 1], density=True)
        ax[0, i].set_title('Aspect ratio ({})'.format(l))
        ax[1, i].set_title('Scale ({})'.format(l))
    fig.tight_layout()
    fig.show()

    # corner = True
    #
    # fig = sns.pairplot(df, hue="label", corner=corner, markers='+', plot_kws=dict(s=2,linewidth=0.2))
    # fig.savefig("boxes_label.png")
    # plt.show()
    #
    #
    # _ = df.pop('label')
    #
    # fig = sns.pairplot(df, corner=corner, markers='+', plot_kws=dict(s=2,linewidth=0.2))
    # fig.savefig("boxes.png")
    # plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    ax[0].hist(df.aspect_ratio_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in aspect_ratios]).flatten(),
               color='red', density=True, rwidth=0.5, label=str(aspect_ratios))
    ax[0].hist(df.aspect_ratio, bins=100, range=(0, 6), density=True, alpha=0.7,
               label='Max {:.1f}\nMin: {:.3f}'.format(max(df.aspect_ratio), min(df.aspect_ratio)))
    ax[0].set_title("Aspect ratio")
    ax[0].legend()
    ax[1].hist(df.scale_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in scales]).flatten(), color='red',
               density=True, rwidth=0.5, label=str(scales))
    ax[1].hist(df.scale, bins=100, range=(0, 4), density=True, alpha=0.7,
               label='Max {:.3f}\nMin: {:.3f}'.format(max(df.scale), min(df.scale)))
    ax[1].set_title("Scale")
    ax[1].legend()
    fig.savefig("boxes_generator_parameters.png")
    fig.show()

    _df = df[df.image_height == 1280]
    _df = _df[_df.ycenter < 500]
    # _df = _df[_df.height<200]
    b, m = polyfit(_df['width'], _df['xcenter'], 1)
    fig = plt.scatter(x=_df['width'], y=_df['xcenter'], marker='+', s=1)
    plt.xlim((0, max(df.width)))
    plt.ylim((0, max(df.xcenter)))
    plt.xlabel("Width")
    plt.ylabel("Horizontal Position")
    plt.title('Boxes in the top of the images (1920 x 1280)')
    plt.legend()
    plt.show()

    fig = plt.scatter(x=_df['height'], y=_df['ycenter'], marker='+', s=1)
    plt.xlim((0, max(df.height)))
    plt.ylim((0, max(df.ycenter)))
    plt.xlabel("Height")
    plt.ylabel("Vertical Position")
    plt.title('Boxes in the top of the images (1920 x 1280)')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    ax[0].hist(_df.aspect_ratio_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in aspect_ratios]).flatten(),
               color='red', density=True, rwidth=0.5, label=str(aspect_ratios))
    ax[0].hist(_df.aspect_ratio, bins=100, range=(0, 6), density=True, alpha=0.7,
               label='Max {:.1f}\nMin: {:.3f}'.format(max(_df.aspect_ratio), min(_df.aspect_ratio)))
    ax[0].set_title("Aspect ratio")
    ax[0].legend()
    ax[1].hist(_df.scale_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in scales]).flatten(), color='red',
               density=True, rwidth=0.5, label=str(scales))
    ax[1].hist(_df.scale, bins=100, range=(0, 4), density=True, alpha=0.7,
               label='Max {:.3f}\nMin: {:.3f}'.format(max(_df.scale), min(_df.scale)))
    ax[1].set_title("Scale")
    ax[1].legend()
    fig.show()


# study_width_height_position()

def study_cluster(_df, n_clusters, silhoutte_n_clusters=list(range(2, 21)), save_fig=True, label=''):
    # label_array = np.reshape(_df['label'].array, (_df['label'].shape[0], -1))
    # one_hot = OneHotEncoder().fit_transform(X=label_array)
    X = MinMaxScaler().fit_transform(_df[['scale', 'aspect_ratio']])
    # X = MinMaxScaler().fit_transform(_df[['height', 'width']])

    plt.clf()
    # Elbow
    if silhoutte_n_clusters is not None and silhoutte_n_clusters != []:
        silhoutte_values = {}
        for k in tqdm(silhoutte_n_clusters):
            kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=8).fit(X)
            labels = kmeans.predict(X)
            silhoutte_values[k] = kmeans.inertia_  # [silhouette_score(X, labels)]
        plt.plot(list(silhoutte_values.keys()), list(silhoutte_values.values()), 'o-')
        plt.xlabel("number of clusters")
        plt.ylabel("SSE")
        plt.title(label)
        if save_fig:
            plt.savefig('elbow_' + label + '.png')
        plt.show()

    # Clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=8).fit(X)
    labels = kmeans.labels_

    # fig = plt.scatter(x=_df['xcenter'], y=_df['ycenter'], c=labels, marker='+', s=0.1)
    # ax = plt.gca()
    # ax.invert_yaxis()
    # legend1 = ax.legend(*fig.legend_elements(),
    #                     loc="best", title="Clusters")
    # ax.add_artist(legend1)
    # plt.show()

    ## He sacado la imagen con 2 clusters y sample 0.3, and point size 0.001

    # fig, ax = plt.subplots(1 , n_clusters + 1, figsize=(15, 2 * (n_clusters + 1)), sharex=True, sharey=True)
    # #fig, ax = plt.subplots(1, n_clusters + 1, sharex=True, sharey=True)
    # fig0 = ax[0].scatter(x=_df['xcenter'], y=_df['ycenter'], c=labels, s=0.001)
    # ax[0].invert_yaxis()
    # ax[0].set_title("All objects", fontsize=16)
    # ax[0].set_xlabel("Horizontal Position", fontsize=14)
    # ax[0].set_ylabel("Normalized Vertical Position", fontsize=14)
    # for k in range(n_clusters):
    #     if k==1:
    #         ax[k + 1].scatter(x=_df[labels == k]['xcenter'], y=_df[labels == k]['ycenter'], s=0.002)
    #     else:
    #         ax[k + 1].scatter(x=_df[labels == k]['xcenter'], y=_df[labels == k]['ycenter'], s=0.001)
    #     ax[k + 1].set_title('Cluster {}'.format(k), fontsize=16)
    #     ax[k+1].set_xlabel("Horizontal Position", fontsize=14)
    #
    # legend1 = ax[0].legend(*fig0.legend_elements(),
    #                      loc="best", title="Clusters")
    # ax[0].add_artist(legend1)
    # #plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #  #                   hspace=0, wspace=0)
    # #plt.margins(0, 0)
    # plt.show()
    # if save_fig:
    #     fig.savefig('clustering_' + label + '.png', dpi=500, bbox_inches='tight')

    df_cluster = pd.concat((_df, pd.DataFrame(labels, index=_df.index, columns=["cluster"])), axis=1)
    df_cluster = df_cluster[(df_cluster.aspect_ratio<6) & (df_cluster.scale<4)]
    fig = plt.figure()
    # ax.boxplot(_df['scale'],positions=labels)
    # ax = sns.catplot(x="cluster", y='aspect_ratio', data=df_cluster, kind='violin', palette="Blues")
    _df_cluster = df_cluster[['aspect_ratio', 'scale', 'cluster']]
    df_cluster = df_cluster.rename(columns={"aspect_ratio": "Aspect Ratio", "scale": "Scale Ratio"})

    _df_cluster = df_cluster.melt(var_name='Characteristic', value_vars=['Aspect Ratio', 'Scale Ratio'], value_name='Value')

    _df_cluster['cluster'] = df_cluster['cluster'].values.tolist() + df_cluster['cluster'].values.tolist()
    _df_cluster = _df_cluster.rename(columns={"cluster": "Cluster"})



    ax = sns.violinplot(y='Value', x='Characteristic', hue='Cluster', data=_df_cluster, split=True, palette='Blues',inner=None)
    # ax = sns.violinplot(y="aspect_ratio", data=df_cluster[df_cluster.aspect_ratio<6], hue="cluster",split=True)

    plt.title("Cluster distribution", fontsize=16)
    # plt.xlabel("Cluster", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.xlabel("Feature", fontsize=14)
    plt.show()
    print()
    fig.savefig('cluster_dist_all.png', dpi=500, bbox_inches='tight')

    # fig = plt.scatter(x=_df['height'], y=_df['ycenter'], s=0.001, c=labels)
    # plt.xlim((0, max(_df.height)))
    # plt.ylim((0, max(_df.ycenter)))
    # plt.xlabel("Height")
    # plt.ylabel("Vertical Position")
    # plt.title('')
    # ax = plt.gca()
    # ax.invert_yaxis()
    # legend1 = ax.legend(*fig.legend_elements(),
    #                     loc="best", title="Clusters")
    # ax.add_artist(legend1)
    # plt.title(label)
    # if save_fig:
    #     plt.savefig('VerticalVsHeight_' + label + '.png')
    # plt.show()

    fig, ax = plt.subplots(2, n_clusters, figsize=(5 * n_clusters, 6), sharey='col')
    for k in range(n_clusters):
        __df = _df[labels == k]
        # ax[0, k].hist(__df.aspect_ratio_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in aspect_ratios]).flatten(),
        #               color='red', density=True, rwidth=0.5, label=str(aspect_ratios))
        sns.distplot(__df.aspect_ratio, ax=ax[k, 0], hist_kws={"range": (0, 6)}, kde_kws={"clip": (0, 6)})
        # ax[0, k].hist(__df.aspect_ratio, bins=100, range=(0, 6), density=True, alpha=0.7,)
        # label='Max {:.1f}\nMin: {:.3f}'.format(max(__df.aspect_ratio), min(__df.aspect_ratio)))
        ax[k, 0].set_title("Aspect ratio (cluster {})".format(k))
        # ax[0, k].legend()
        # ax[1, k].hist(__df.scale_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in scales]).flatten(), color='red',
        #               density=True, rwidth=0.5, label=str(scales))
        sns.distplot(__df[_df.scale < 4].scale, ax=ax[k, 1], hist_kws={"range": (0, 4)}, kde_kws={"clip": (0, 4)})

        # ax[1, k].hist(__df.scale, bins=100, range=(0, 4), density=True, alpha=0.7,)
        # label='Max {:.3f}\nMin: {:.3f}'.format(max(__df.scale), min(__df.scale)))
        ax[k, 1].set_title("Scale ratio (cluster {})".format(k))
        # ax[1, k].legend()

    ax[0, 0].set_ylabel("Density", fontsize=10)
    ax[1, 0].set_ylabel("Density", fontsize=10)

    ax[0, 0].set_xlabel("")
    ax[0, 1].set_xlabel("")
    ax[1, 0].set_xlabel("")
    ax[1, 1].set_xlabel("")

    plt.show()
    if save_fig:
        fig.savefig('cluster_description_' + label + '.png', dpi=500, bbox_inches='tight')

    # Scale and aspect ratio
    fig, ax = plt.subplots(2, n_clusters, figsize=(5 * n_clusters, 6))
    for k in range(n_clusters):
        __df = _df[labels == k]
        # ax[0, k].hist(__df.aspect_ratio_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in aspect_ratios]).flatten(),
        #               color='red', density=True, rwidth=0.5, label=str(aspect_ratios))
        ax[0, k].hist(__df.aspect_ratio, bins=100, range=(0, 6), density=True, alpha=0.7,
                      label='Max {:.1f}\nMin: {:.3f}'.format(max(__df.aspect_ratio), min(__df.aspect_ratio)))
        ax[0, k].set_title("Aspect ratio (cluster {})".format(k))
        ax[0, k].legend()
        # ax[1, k].hist(__df.scale_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in scales]).flatten(), color='red',
        #               density=True, rwidth=0.5, label=str(scales))
        ax[1, k].hist(__df.scale, bins=100, range=(0, 4), density=True, alpha=0.7,
                      label='Max {:.3f}\nMin: {:.3f}'.format(max(__df.scale), min(__df.scale)))
        ax[1, k].set_title("Scale (cluster {})".format(k))
        ax[1, k].legend()
    plt.show()
    if save_fig:
        fig.savefig('cluster_description_' + label + '.png')




def region_study(_df, n_clusters):
    X = MinMaxScaler().fit_transform(_df[['scale', 'aspect_ratio']])
    n_cluster = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=8).fit(X)
    labels = kmeans.labels_

    df_cluster = pd.concat((_df, pd.DataFrame(labels, index=_df.index, columns=["cluster"])), axis=1)

    num_elements = [df_cluster[df_cluster['cluster'] == k].shape[0] for k in [0, 1]]
    regions_count = {}

    for k in range(n_clusters):
        count_dict = {}
        clust = df_cluster[df_cluster['cluster'] == k]
        keys_list = []

        for i, limit in enumerate(np.arange(0.05, 1, 0.05)):
            keys_list.append(limit)
            if i == 0:
                count_dict[limit] = clust[(clust['ycenter'] < limit)].shape[0]
            else:
                count_dict[limit] = clust[(clust['ycenter'] < limit) & (clust['ycenter'] > keys_list[i - 1])].shape[0]
        regions_count[k] = count_dict
    # fig, ax = plt.subplots(2, 2, figsize=(15, 2 * (n_clusters + 1)), sharex=True, sharey=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for k in range(n_clusters):
        if k == 1:
            ax[k][0].scatter(x=_df[labels == k]['xcenter'], y=_df[labels == k]['ycenter'], s=0.002)
        else:
            ax[k][0].scatter(x=_df[labels == k]['xcenter'], y=_df[labels == k]['ycenter'], s=0.001)
        ax[k][0].set_title('Cluster {}'.format(k), fontsize=16)
        if k == 1:
            ax[k][0].set_xlabel("Horizontal Position", fontsize=14)
        ax[k][0].set_ylabel("Normalized Vertical Position", fontsize=14)
        ax[k][0].invert_yaxis()
        ax[k][0].set_xlim((0, 1920))
        ax[k][0].set_xticks([250, 500, 750, 1000, 1250, 1500, 1750])

        if k == 0:
            for line in [0.2, 0.4, 0.7]:
                ax[k][0].axhline(y=line, color='r', linestyle='--')
            # ax[k][0].hlines([0.2, 0.4, 0.7], xmin=0, xmax=ax[k][0].get_xlim()[1], colors='r', linestyles='dashed')
        else:
            for line in [0.4]:
                ax[k][0].axhline(y=line, color='r', linestyle='--')
            # ax[k][0].hlines([0.4], xmin=0, xmax=ax[k][0].get_xlim()[1], colors='r',linestyles='dashed')
        # ax[k][0].hlines(np.arange(0.05, 1, 0.05),xmin=0, xmax=ax[k][0].get_xlim()[1], colors='r',linestyles='dashed')
        ax[k][0].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

        ax[k][1].plot([x for x in list(regions_count[k].keys())],
                      [x / num_elements[k] * 100 for x in list(regions_count[k].values())])
        ax[k][1].scatter([x for x in list(regions_count[k].keys())],
                         [x / num_elements[k] * 100 for x in list(regions_count[k].values())], s=5)

        # ax[k][1].bar([x for x in list(regions_count[k].keys())], [x / num_elements[k] * 100 for x in list(regions_count[k].values())], width=0.03)
        ax[k][1].set_title('Cluster {}'.format(k), fontsize=16)

        if k == 1:
            ax[k][1].set_xlabel("Normalized Height", fontsize=14)
        ax[k][1].set_ylabel("Percentage of elements", fontsize=14)
        ax[k][1].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        if k == 0:
            ax[k][1].vlines([0.2, 0.4, 0.7], ymin=0, ymax=ax[k][1].get_ylim()[1], colors='r', linestyles='dashed')
        else:
            ax[k][1].vlines([0.4], ymin=0, ymax=ax[k][1].get_ylim()[1], colors='r', linestyles='dashed')
            ax[k][1].set_yticks(list(range(0, 20, 2)))

        # ax[k][1].set_ylim((0, 20))

    #
    # legend1 = ax[0].legend(*fig0.legend_elements(),
    #                        loc="best", title="Clusters")
    # ax[0].add_artist(legend1)

    # dict_c0_num = {}
    # c = df_cluster[df_cluster['cluster'] == 0]
    # keys_list = []
    # for i, limit in enumerate(np.arange(0.05, 1, 0.05)):
    #     keys_list.append(limit)
    #     if i == 0:
    #         dict_c0_num[limit] = c[(c['ycenter'] < limit)].shape[0]
    #     else:
    #         dict_c0_num[limit] = c[(c['ycenter'] < limit) & (c['ycenter'] > keys_list[i - 1])].shape[0]
    #
    # sorted_c0 = sorted(dict_c0.items(), key=lambda x: x[1], reverse=True)
    #
    # plt.figure()
    # plt.title("Cluster 0")
    # plt.plot([x for x in list(dict_c0_num.keys())], list(dict_c0_num.values()))
    # plt.show()
    #
    # dict_c1_num = {}
    # c = df_cluster[df_cluster['cluster'] == 1]
    # keys_list = []
    # for i, limit in enumerate(np.arange(0.05, 1, 0.05)):
    #     keys_list.append(limit)
    #     if i == 0:
    #         dict_c1_num[limit] = c[(c['ycenter'] < limit)].shape[0]
    #     else:
    #         dict_c1_num[limit] = c[(c['ycenter'] < limit) & (c['ycenter'] > keys_list[i - 1])].shape[0]
    #
    # sorted_c1 = sorted(dict_c0.items(), key=lambda x: x[1], reverse=True)
    #
    # plt.figure()
    # plt.title("Cluster 1")
    # plt.plot([x for x in list(dict_c1_num.keys())], [x / c.shape[0] * 100 for x in list(dict_c1_num.values())])
    # plt.xlabel("Normalized Height")
    # plt.ylabel("Percentage of elements")
    # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # plt.ylim((0, 20))
    # plt.show()
    # fig.savefig('cluster1.png', dpi=500, bbox_inches='tight')
    #
    # plt.figure()
    # plt.bar([x for x in list(dict_c1_num.keys())], [x / c.shape[0] * 100 for x in list(dict_c1_num.values())],
    #         align='edge')
    # plt.xlabel("Normalized Height")
    # plt.ylabel("Percentage of elements")
    # plt.show()
    #
    # sorted_c0 = sorted(dict_c0.items(), key=lambda x: x[1], reverse=True)
    #
    # # if save_fig:
    fig.savefig('regions.png', dpi=500, bbox_inches='tight')
    plt.show()
    print("Hola")

def region_study2(_df, n_clusters):
    X = MinMaxScaler().fit_transform(_df[['scale', 'aspect_ratio']])
    n_cluster = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=8).fit(X)
    labels = kmeans.labels_

    df_cluster = pd.concat((_df, pd.DataFrame(labels, index=_df.index, columns=["cluster"])), axis=1)

    num_elements = [df_cluster[df_cluster['cluster'] == k].shape[0] for k in [0, 1]]
    regions_count = {}



    g = sns.jointplot("xcenter", "ycenter", data=_df[(labels == 0)],
                      kind="reg", scatter_kws={'s': 0.001})
    g.ax_joint.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    g.ax_joint.invert_yaxis()
    for line in [0.2, 0.7]:
        g.ax_joint.axhline(y=line, color='r', linestyle='--')

    # plt.show()
    g.savefig('jointplot_c0.png', dpi=500, bbox_inches='tight')

    g = sns.jointplot("xcenter", "ycenter", data=_df[(labels == 1)],
                      kind="reg", scatter_kws={'s':0.03})
    g.ax_joint.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    g.ax_joint.invert_yaxis()
    for line in [0.4]:
        g.ax_joint.axhline(y=line, color='r', linestyle='--')

    # plt.show()

    g.savefig('jointplot_c1.png', dpi=500, bbox_inches='tight')
    print()




def region_study3(_df, n_clusters):
    X = MinMaxScaler().fit_transform(_df[['scale', 'aspect_ratio']])
    n_cluster = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=8).fit(X)
    labels = kmeans.labels_

    df_cluster = pd.concat((_df, pd.DataFrame(labels, index=_df.index, columns=["cluster"])), axis=1)

    num_elements = [df_cluster[df_cluster['cluster'] == k].shape[0] for k in [0, 1]]
    regions_count = {}

    for k in range(n_clusters):
        count_dict = {}
        clust = df_cluster[df_cluster['cluster'] == k]
        keys_list = []

        for i, limit in enumerate(np.arange(0.05, 1, 0.05)):
            keys_list.append(limit)
            if i == 0:
                count_dict[limit] = clust[(clust['ycenter'] < limit)].shape[0]
            else:
                count_dict[limit] = clust[(clust['ycenter'] < limit) & (clust['ycenter'] > keys_list[i - 1])].shape[0]
        regions_count[k] = count_dict
    # fig, ax = plt.subplots(2, 2, figsize=(15, 2 * (n_clusters + 1)), sharex=True, sharey=True)

    fig, ax = plt.subplots(2, 1, figsize=(5, 9))
    for k in range(n_clusters):
        if k == 1:
            ax[k].scatter(x=_df[labels == k]['xcenter'], y=_df[labels == k]['ycenter'], s=0.002)
        else:
            ax[k].scatter(x=_df[labels == k]['xcenter'], y=_df[labels == k]['ycenter'], s=0.001)
        ax[k].set_title('Cluster {}'.format(k), fontsize=16)
        if k == 1:
            ax[k].set_xlabel("Horizontal Position", fontsize=14)
        ax[k].set_ylabel("Normalized Vertical Position", fontsize=14)
        ax[k].invert_yaxis()
        ax[k].set_xlim((0, 1920))
        ax[k].set_xticks([250, 500, 750, 1000, 1250, 1500, 1750])

        if k == 0:
            for line in [0.2, 0.7]:
                ax[k].axhline(y=line, color='r', linestyle='--')
            # ax[k][0].hlines([0.2, 0.4, 0.7], xmin=0, xmax=ax[k][0].get_xlim()[1], colors='r', linestyles='dashed')
        else:
            for line in [0.4]:
                ax[k].axhline(y=line, color='r', linestyle='--')
            # ax[k][0].hlines([0.4], xmin=0, xmax=ax[k][0].get_xlim()[1], colors='r',linestyles='dashed')
        # ax[k][0].hlines(np.arange(0.05, 1, 0.05),xmin=0, xmax=ax[k][0].get_xlim()[1], colors='r',linestyles='dashed')
        ax[k].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    plt.show()
    fig.savefig('regions_cluster4.png', dpi=500, bbox_inches='tight')

    print("Hola")




n_clusters = 2
n_clusters_elbow = []  # list(range(2, 10))  # list(range(2,21))

print(df.shape[0])
_df = df
# _df = df[df['image_height'] == 1280]
print(_df.shape[0])
_df = _df.sample(frac=0.3)
print(_df.shape[0])
_df = _df[_df['xcenter'] > anchor_base / 2]
_df = _df[_df['xcenter'] < (_df['image_width'] - anchor_base / 2)]
_df = _df[_df['ycenter'] < (_df['image_height'] - anchor_base / 2)]
_df['ycenter'] = _df['ycenter'] / _df['image_height']
_df['height'] = _df['height'] / _df['image_height']
# study_cluster(_df, n_clusters, n_clusters_elbow, save_fig=True, label='AllCameras-{}Clusters'.format(n_clusters))
region_study(_df,n_clusters)

_df['height'] = _df['height'] / _df['image_height']
_df['ycenter'] = _df['ycenter'] / _df['image_height']
study_cluster(_df, n_clusters, n_clusters_elbow, save_fig=True, label='AllCamerasNorm-{}Clusters'.format(n_clusters))

_df = df.sample(frac=0.1)
_df = _df[_df['image_height'] == 1280]
# study_cluster(_df, n_clusters, n_clusters_elbow, save_fig=True, label='FrontCameras-{}Clusters'.format(n_clusters))

_df = df.sample(frac=0.1)
_df = _df[_df['image_height'] != 1280]
# study_cluster(_df, n_clusters, n_clusters_elbow, save_fig=True, label='SideCameras-{}Clusters'.format(n_clusters))

_df = df.sample(frac=sample_frac)
# _df = _df[_df['xcenter']>anchor_base/2]
# _df = _df[_df['xcenter']<(_df['image_width']-anchor_base/2)]
# _df = _df[_df['ycenter']<(_df['image_height']-anchor_base/2)]
_df['ycenter'] = _df['ycenter'] / _df['image_height']

lines = [0.17, 0.4, 0.7]
area_names = ['SKY', 'TOP', 'MIDDLE', 'BOTTOM']
region_colors = ['rebeccapurple', 'steelblue', 'mediumaquamarine', 'gold']


def assign_area(y, lines=lines, text=True, area_names=area_names):
    lines = sorted(lines)

    def _assign_area(y):
        for i, (name, line) in enumerate(zip(area_names, lines)):
            if y < line:
                return name if text else i
        return area_names[-1] if text else len(lines)

    return np.vectorize(_assign_area)(y)


_df['region'] = assign_area(_df['ycenter'], text=True)
_df['region_n'] = assign_area(_df['ycenter'], text=False)
_df['label_name'] = np.vectorize(lambda x: boxlabels[int(x)])(_df['label'])


def clustering():
    X = MinMaxScaler().fit_transform(_df[['scale', 'aspect_ratio']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=8).fit(X)
    labels = kmeans.labels_

    _, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig = ax[0].scatter(x=_df['xcenter'], y=_df['ycenter'], c=labels, marker='+', s=1)
    ax[1].scatter(x=_df[labels == 0]['xcenter'], y=_df[labels == 0]['ycenter'], marker='+', s=1)
    ax[2].scatter(x=_df[labels == 1]['xcenter'], y=_df[labels == 1]['ycenter'], marker='+', s=1)
    fig1 = ax[3].scatter(_df['xcenter'], y=_df['ycenter'], c=_df['region_n'], cmap=ListedColormap(region_colors),
                         marker='+', s=1)
    for line in lines:
        ax[0].axhline(y=line, color='r', linestyle='--')
        ax[1].axhline(y=line, color='r', linestyle='--')
        ax[2].axhline(y=line, color='r', linestyle='--')
    # ax[0].invert_yaxis()
    legend1 = ax[0].legend(*fig.legend_elements(),
                           loc="best", title="Clusters")
    ax[0].add_artist(legend1)
    ax[3].invert_yaxis()
    legend1 = ax[3].legend(handles=fig1.legend_elements()[0], labels=area_names,
                           loc="best", title="Regions")
    ax[3].add_artist(legend1)
    plt.savefig('regions_cluster.png')
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(5, 6))
    for r, region in enumerate(area_names):
        # ax[0].hist(_df[_df['region']==region].aspect_ratio_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in aspect_ratios]).flatten(),
        #            color=region_colors[r], density=True, rwidth=0.5,) #label=str(aspect_ratios))
        ax[0].hist(_df[_df['region'] == region].aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=region_colors[r],
                   histtype='step',
                   label=region + ' [{:.1f}, {:.3f}]'.format(max(_df[_df['region'] == region].aspect_ratio),
                                                             min(_df[_df['region'] == region].aspect_ratio)))
        # ax[1].hist(_df[_df['region']==region].scale_closer, bins=np.asarray([[b - 0.1, b + 0.1] for b in scales]).flatten(),
        #            color=region_colors[r], density=True, rwidth=0.5, )#label=str(scales))
        ax[1].hist(_df[_df['region'] == region].scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r],
                   histtype='step',
                   label=region + ' [{:.3f}, {:.3f}]'.format(max(_df[_df['region'] == region].scale),
                                                             min(_df[_df['region'] == region].scale)))
    ax[0].set_title("Aspect ratio")
    ax[0].legend()
    ax[1].set_title("Scale")
    ax[1].legend()
    plt.show()

    fig, ax = plt.subplots(2, len(area_names), figsize=(5 * len(area_names), 6))
    for r, region in enumerate(area_names):
        ax[0, r].hist(_df[_df['region'] == region].aspect_ratio_closer,
                      bins=np.asarray([[b - 0.1, b + 0.1] for b in aspect_ratios]).flatten(),
                      color='r', density=True, rwidth=0.5, label=str(aspect_ratios))
        ax[0, r].hist(_df[_df['region'] == region].aspect_ratio, bins=100, range=(0, 6), alpha=0.7,
                      color=region_colors[r], density=True,
                      label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(
                          max(_df[_df['region'] == region].aspect_ratio),
                          min(_df[_df['region'] == region].aspect_ratio),
                          np.mean(_df[_df['region'] == region].aspect_ratio)))
        ax[1, r].hist(_df[_df['region'] == region].scale_closer,
                      bins=np.asarray([[b - 0.1, b + 0.1] for b in scales]).flatten(),
                      color='r', density=True, rwidth=0.5, label=str(scales))
        ax[1, r].hist(_df[_df['region'] == region].scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r],
                      density=True,
                      label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(
                          max(_df[_df['region'] == region].scale), min(_df[_df['region'] == region].scale),
                          np.mean(_df[_df['region'] == region].scale)))
        ax[0, r].set_title("Aspect ratio ({})".format(region))
        ax[0, r].legend()
        ax[1, r].set_title("Scale ({})".format(region))
        ax[1, r].legend()
    fig.savefig('regions_description.png')
    plt.show()

    sns.countplot(y="region", order=area_names, data=_df, palette={r: c for r, c in zip(area_names, region_colors)})
    plt.savefig("regions_countplot.png")
    plt.show()

    sns.countplot(x="label_name", hue='region', hue_order=area_names, data=_df,
                  palette={r: c for r, c in zip(area_names, region_colors)})
    plt.show()

    sns.countplot(y="region", order=area_names, hue='label_name', data=_df,
                  palette={**{r: c for r, c in zip(area_names, region_colors)},
                           **{boxlabels[float(i + 1)]: colors[float(i + 1)] for i in range(3)}})
    plt.show()

    fig, ax = plt.subplots(2, 5, figsize=(17, 4))
    gs = ax[0, 0].get_gridspec()
    ax[0, 0].remove()
    ax[1, 0].remove()
    ax[1, 4].remove()
    axbig = fig.add_subplot(gs[0:, 0])
    sns.countplot(y="region", order=area_names, data=_df, palette={r: c for r, c in zip(area_names, region_colors)},
                  ax=axbig)
    sns.countplot(x="label_name", data=_df[_df['region'] == 'SKY'], ax=ax[0, 1])
    ax[0, 1].set_title("SKY")
    sns.countplot(x="label_name", data=_df[_df['region'] == 'TOP'], ax=ax[0, 2])
    ax[0, 2].set_title("TOP")
    sns.countplot(x="label_name", data=_df[_df['region'] == 'MIDDLE'], ax=ax[0, 3])
    ax[0, 3].set_title("MIDDLE")
    sns.countplot(x="label_name", data=_df[_df['region'] == 'BOTTOM'], ax=ax[0, 4])
    ax[0, 4].set_title("BOTTOM")
    sns.countplot(x="region", data=_df[_df['label_name'] == 'VEHICLE'], ax=ax[1, 1], order=area_names,
                  palette={r: c for r, c in zip(area_names, region_colors)})
    ax[1, 1].set_title("VEHICLE")
    sns.countplot(x="region", data=_df[_df['label_name'] == 'PEDESTRIAN'], ax=ax[1, 2], order=area_names,
                  palette={r: c for r, c in zip(area_names, region_colors)})
    ax[1, 2].set_title("PEDESTRIAN")
    sns.countplot(x="region", data=_df[_df['label_name'] == 'CYCLIST'], ax=ax[1, 3], order=area_names,
                  palette={r: c for r, c in zip(area_names, region_colors)})
    ax[1, 3].set_title("CYCLIST")
    fig.tight_layout()
    plt.savefig("regions_countplot.png")
    plt.show()


n_aspect_ration = 3
bound_aspect_ratio = n_aspect_ration * [[0, 5.]]
n_scale = 4
bound_scale = n_scale * [[0.06, 5.]]

best_values = defaultdict(lambda: {'aspect_ratio': None, 'scale': None})


def optimize_parameters():
    from scipy.optimize import minimize, shgo

    def optimize_param(x, data, param):
        count = np.unique(closer_to(data[param], x), return_counts=True)[1]
        if len(x) != len(count):
            return 1e10
        return count.std() - np.std(x) * 1000

    print(8 * "#")
    print("Aspect ratio and Scale OPTIMIZATION")
    print("\tInit values")
    print("\t\tAspect ratio", aspect_ratios[:n_aspect_ration])
    print("\t\tScale", scales[:n_scale])
    print("\tBounds", 'ToDo')
    best_values = defaultdict(lambda: {'aspect_ratio': None, 'scale': None})
    for region in area_names:
        __df = _df[_df['region'] == region]
        optimal_aspect_ratio = shgo(optimize_param, bound_aspect_ratio, args=(__df, 'aspect_ratio'), n=1000, iters=5,
                                    sampling_method='sobol')
        optimal_scale = shgo(optimize_param, bound_scale, args=(__df, 'scale'), n=1000, iters=5,
                             sampling_method='sobol')
        best_values[region]['aspect_ratio'] = sorted(optimal_aspect_ratio.x)
        best_values[region]['scale'] = sorted(optimal_scale.x)
        print("\tBest values", region)
        print("\t\tAspect ratio", best_values[region]['aspect_ratio'])
        print("\t\tScale", best_values[region]['scale'])
    print(8 * "#")

    fig, ax = plt.subplots(2, len(area_names), figsize=(5 * len(area_names), 6))
    for r, region in enumerate(area_names):
        __df = _df[_df['region'] == region]
        __df['aspect_ratio_best'] = closer_to(__df['aspect_ratio'], best_values[region]['aspect_ratio'])
        __df['scale_best'] = closer_to(__df['scale'], best_values[region]['scale'])
        ax[0, r].hist(__df.aspect_ratio_best,
                      bins=np.asarray([[b - 0.03, b + 0.03] for b in best_values[region]['aspect_ratio']]).flatten(),
                      color='r', density=True, rwidth=0.5,
                      label=str(['{:.2f}'.format(x) for x in best_values[region]['aspect_ratio']]))
        ax[0, r].hist(__df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=region_colors[r], density=True,
                      label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.aspect_ratio),
                                                                                        min(__df.aspect_ratio),
                                                                                        np.mean(__df.aspect_ratio)))
        ax[1, r].hist(__df.scale_best,
                      bins=np.asarray([[b - 0.03, b + 0.03] for b in best_values[region]['scale']]).flatten(),
                      color='r', density=True, rwidth=0.5,
                      label=str(['{:.2f}'.format(x) for x in best_values[region]['scale']]))
        ax[1, r].hist(__df.scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r], density=True,
                      label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.scale),
                                                                                        min(__df.scale),
                                                                                        np.mean(__df.scale)))
        ax[0, r].set_title("Optimized sspect ratio ({})".format(region))
        ax[0, r].legend()
        ax[1, r].set_title("Optimized scale ({})".format(region))
        ax[1, r].legend()
    fig.savefig('regions_description_best.png')
    plt.show()


def optimize_params_genetic_algorithm(features=['aspect_ratio', 'scale'], fitness_name='ALL'):
    """

    Args:
        features: a list which can contains 'aspect_ratio', 'scale' or both
        fitness: ALL, STD, DIST, MAX_DIST, QTL

    Returns:

    """
    from pyeasyga import pyeasyga
    best_values = defaultdict(lambda: {'aspect_ratio': None, 'scale': None})

    def create_individual(data):
        _, _, bounds = data[0]
        return [np.random.randint(bound[0] * 100, bound[1] * 100) / 100 for bound in bounds]

    def mutate(individual):
        """Reverse the bit of a random index in an individual."""
        mutate_index = random.randrange(len(individual))
        individual[mutate_index] = np.random.randint(6, 500) / 100

    if fitness_name in ['STD', 'ALL']:

        def fitness(individual, data):
            param, df, bounds = data[0]

            count = np.unique(closer_to(df[param], individual), return_counts=True)[1]
            if len(count) != len(individual):
                return 1e10
            norm_std_count = count.std() / (len(df) / 2)
            norm_std_individual = np.std(individual) / (max([x[1] for x in bounds]) / 2)
            return 2 * norm_std_count  # - norm_std_individual

        for param, region in tqdm(list(itertools.product(features, area_names))):
            bounds = bound_aspect_ratio if param == 'aspect_ratio' else bound_scale
            __df = _df[_df['region'] == region]

            data = [(param, __df, bounds)]

            ga = pyeasyga.GeneticAlgorithm(data,
                                           population_size=1000,
                                           generations=10,
                                           crossover_probability=0.8,
                                           mutation_probability=0.2,
                                           elitism=True,
                                           maximise_fitness=False)
            ga.create_individual = create_individual
            ga.fitness_function = fitness
            ga.mutate_function = mutate
            ga.run()
            ga.best_individual()
            best_values[region][param] = sorted(ga.best_individual()[1])
            print(region, param, best_values[region][param])

        fig, ax = plt.subplots(2, len(area_names), figsize=(5 * len(area_names), 6))
        for r, region in enumerate(area_names):
            __df = _df[_df['region'] == region]
            __df['aspect_ratio_best'] = closer_to(__df['aspect_ratio'], best_values[region]['aspect_ratio'])
            __df['scale_best'] = closer_to(__df['scale'], best_values[region]['scale'])
            ax[0, r].hist(__df.aspect_ratio_best, bins=np.asarray(
                [[b - 0.02, b + 0.02] for b in best_values[region]['aspect_ratio']]).flatten(),
                          color='r', density=True, rwidth=0.5,
                          label=str(['{:.2f}'.format(x) for x in best_values[region]['aspect_ratio']]))
            ax[0, r].hist(__df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=region_colors[r], density=True,
                          label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.aspect_ratio),
                                                                                            min(__df.aspect_ratio),
                                                                                            np.mean(__df.aspect_ratio)))
            ax[1, r].hist(__df.scale_best,
                          bins=np.asarray([[b - 0.01, b + 0.01] for b in best_values[region]['scale']]).flatten(),
                          color='r', density=True, rwidth=0.5,
                          label=str(['{:.2f}'.format(x) for x in best_values[region]['scale']]))
            ax[1, r].hist(__df.scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r], density=True,
                          label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.scale),
                                                                                            min(__df.scale),
                                                                                            np.mean(__df.scale)))
            ax[0, r].set_title("Optimized aspect ratio ({})".format(region))
            ax[0, r].legend()
            ax[1, r].set_title("Optimized scale ({})".format(region))
            ax[1, r].legend()
        fig.savefig('regions_description_best_std.png')
        plt.show()

    if fitness_name in ['DIST', 'ALL']:

        def fitness_distance(individual, data):
            param, df, bounds = data[0]
            distance = np.sum(np.abs(closer_to(df[param], individual) - df[param]))
            if len(np.unique(individual)) != len(individual):
                return 1e10
            return distance

        for param, region in tqdm(list(itertools.product(features, area_names))):
            bounds = bound_aspect_ratio if param == 'aspect_ratio' else bound_scale
            __df = _df[_df['region'] == region]

            data = [(param, __df, bounds)]

            ga = pyeasyga.GeneticAlgorithm(data,
                                           population_size=1000,
                                           generations=5,
                                           crossover_probability=0.8,
                                           mutation_probability=0.2,
                                           elitism=True,
                                           maximise_fitness=False)
            ga.create_individual = create_individual
            ga.fitness_function = fitness_distance
            ga.mutate_function = mutate
            ga.run()
            ga.best_individual()
            best_values[region][param] = sorted(ga.best_individual()[1])
            print(region, param, best_values[region][param])

        fig, ax = plt.subplots(2, len(area_names), figsize=(5 * len(area_names), 6))
        for r, region in enumerate(area_names):
            __df = _df[_df['region'] == region]
            __df['aspect_ratio_best'] = closer_to(__df['aspect_ratio'], best_values[region]['aspect_ratio'])
            __df['scale_best'] = closer_to(__df['scale'], best_values[region]['scale'])
            ax[0, r].hist(__df.aspect_ratio_best, bins=np.asarray(
                [[b - 0.02, b + 0.02] for b in best_values[region]['aspect_ratio']]).flatten(),
                          color='r', density=True, rwidth=0.5,
                          label=str(['{:.2f}'.format(x) for x in best_values[region]['aspect_ratio']]))
            ax[0, r].hist(__df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=region_colors[r], density=True,
                          label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.aspect_ratio),
                                                                                            min(__df.aspect_ratio),
                                                                                            np.mean(__df.aspect_ratio)))
            ax[1, r].hist(__df.scale_best,
                          bins=np.asarray([[b - 0.01, b + 0.01] for b in best_values[region]['scale']]).flatten(),
                          color='r', density=True, rwidth=0.5,
                          label=str(['{:.2f}'.format(x) for x in best_values[region]['scale']]))
            ax[1, r].hist(__df.scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r], density=True,
                          label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.scale),
                                                                                            min(__df.scale),
                                                                                            np.mean(__df.scale)))
            ax[0, r].set_title("Optimized aspect ratio ({})".format(region))
            ax[0, r].legend()
            ax[1, r].set_title("Optimized scale ({})".format(region))
            ax[1, r].legend()
        fig.savefig('regions_description_best_distance.png')
        plt.show()

    if fitness_name in ['MAX_DIST', 'ALL']:

        def fitness_max_distance(individual, data):
            param, df, bounds = data[0]
            distance = np.max(np.abs(closer_to(df[param], individual) - df[param]))
            if len(np.unique(individual)) != len(individual):
                return 1e10
            return distance

        for param, region in tqdm(list(itertools.product(features, area_names))):
            bounds = bound_aspect_ratio if param == 'aspect_ratio' else bound_scale
            __df = _df[_df['region'] == region]

            data = [(param, __df, bounds)]

            ga = pyeasyga.GeneticAlgorithm(data,
                                           population_size=1000,
                                           generations=5,
                                           crossover_probability=0.8,
                                           mutation_probability=0.2,
                                           elitism=True,
                                           maximise_fitness=False)
            ga.create_individual = create_individual
            ga.fitness_function = fitness_max_distance
            ga.mutate_function = mutate
            ga.run()
            ga.best_individual()
            best_values[region][param] = sorted(ga.best_individual()[1])
            print(region, param, best_values[region][param])

        fig, ax = plt.subplots(2, len(area_names), figsize=(5 * len(area_names), 6))
        for r, region in enumerate(area_names):
            __df = _df[_df['region'] == region]
            __df['aspect_ratio_best'] = closer_to(__df['aspect_ratio'], best_values[region]['aspect_ratio'])
            __df['scale_best'] = closer_to(__df['scale'], best_values[region]['scale'])
            ax[0, r].hist(__df.aspect_ratio_best, bins=np.asarray(
                [[b - 0.02, b + 0.02] for b in best_values[region]['aspect_ratio']]).flatten(),
                          color='r', density=True, rwidth=0.5,
                          label=str(['{:.2f}'.format(x) for x in best_values[region]['aspect_ratio']]))
            ax[0, r].hist(__df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=region_colors[r], density=True,
                          label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.aspect_ratio),
                                                                                            min(__df.aspect_ratio),
                                                                                            np.mean(__df.aspect_ratio)))
            ax[1, r].hist(__df.scale_best,
                          bins=np.asarray([[b - 0.01, b + 0.01] for b in best_values[region]['scale']]).flatten(),
                          color='r', density=True, rwidth=0.5,
                          label=str(['{:.2f}'.format(x) for x in best_values[region]['scale']]))
            ax[1, r].hist(__df.scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r], density=True,
                          label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.scale),
                                                                                            min(__df.scale),
                                                                                            np.mean(__df.scale)))
            ax[0, r].set_title("Optimized aspect ratio ({})".format(region))
            ax[0, r].legend()
            ax[1, r].set_title("Optimized scale ({})".format(region))
            ax[1, r].legend()
        fig.savefig('regions_description_best_max_distance.png')
        plt.show()

    if fitness_name in ['QTL', 'ALL']:
        def quantile(df, region, param, q):
            a = df[df['region'] == region][param].values
            return np.quantile(a, q)

        for param, region in tqdm(list(itertools.product(['aspect_ratio', 'scale'], area_names))):
            best_values[region][param] = [quantile(_df, region, param, q) for q in (1 / 6, 1 / 2, 5 / 6)]

        fig, ax = plt.subplots(2, len(area_names), figsize=(5 * len(area_names), 6))
        for r, region in enumerate(area_names):
            __df = _df[_df['region'] == region]
            __df['aspect_ratio_best'] = closer_to(__df['aspect_ratio'], best_values[region]['aspect_ratio'])
            __df['scale_best'] = closer_to(__df['scale'], best_values[region]['scale'])
            ax[0, r].hist(__df.aspect_ratio_best, bins=np.asarray(
                [[b - 0.02, b + 0.02] for b in best_values[region]['aspect_ratio']]).flatten(),
                          color='r', density=True, rwidth=0.5,
                          label=str(['{:.2f}'.format(x) for x in best_values[region]['aspect_ratio']]))
            ax[0, r].hist(__df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=region_colors[r], density=True,
                          label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.aspect_ratio),
                                                                                            min(__df.aspect_ratio),
                                                                                            np.mean(__df.aspect_ratio)))
            ax[1, r].hist(__df.scale_best,
                          bins=np.asarray([[b - 0.01, b + 0.01] for b in best_values[region]['scale']]).flatten(),
                          color='r', density=True, rwidth=0.5,
                          label=str(['{:.2f}'.format(x) for x in best_values[region]['scale']]))
            ax[1, r].hist(__df.scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r], density=True,
                          label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.scale),
                                                                                            min(__df.scale),
                                                                                            np.mean(__df.scale)))
            ax[0, r].set_title("Optimized aspect ratio ({})".format(region))
            ax[0, r].legend()
            ax[1, r].set_title("Optimized scale ({})".format(region))
            ax[1, r].legend()
        fig.savefig('regions_description_best_percentile.png')
        plt.show()


def optimize_params_by_iou_ga():
    """
    GENETIC ALGORITHM:
        individual: [scale1, scale2, scale3, scale4, aspect_ratio1, aspect_ratio2, aspect_ratio3]
    """
    from pyeasyga import pyeasyga
    best_values = defaultdict(lambda: {'aspect_ratio': None, 'scale': None})

    data = pd.read_csv(OUTPUT_FILE)[['xmax', 'xmin', 'ymax', 'ymin', 'image_height']]
    data['ycenter'] = (data['ymin'] + data['ymax']) / 2
    data['region'] = assign_area(data['ycenter'] / data['image_height'], text=True)
    x_factor = data['xmin'] + ((data['xmax'] - data['xmin']) / 2)
    y_factor = data['ymin'] + ((data['ymax'] - data['ymin']) / 2)
    data['xmax'] = data['xmax'] - x_factor
    data['xmin'] = data['xmin'] - x_factor
    data['ymax'] = data['ymax'] - y_factor
    data['ymin'] = data['ymin'] - y_factor

    def create_individual(data):
        _, (bounds_sc, bounds_ar) = data[0]
        return [np.random.randint(bound[0] * 100, bound[1] * 100) / 100 for bound in bounds_sc] + [
            np.random.randint(bound[0] * 100, bound[1] * 100) / 100 for bound in bounds_ar]

    def mutate(individual):
        """Reverse the bit of a random index in an individual."""
        mutate_index = random.randrange(len(individual))
        individual[mutate_index] = np.random.randint(6, 500) / 100 if mutate_index < n_scale else np.random.randint(0,
                                                                                                                    500) / 100

    def crossover(parent_1, parent_2):
        crossover_index_aspect_ratio = random.randrange(n_scale, len(parent_1))
        crossover_index_scalar = random.randrange(1, n_scale)
        scales_parent1, aspect_ratio_parent1 = parent_1[:n_scale], parent_1[n_scale:]
        scales_parent2, aspect_ratio_parent2 = parent_2[:n_scale], parent_2[n_scale:]
        child_1 = scales_parent1[:crossover_index_scalar] + scales_parent2[
                                                            crossover_index_scalar:] + aspect_ratio_parent1[
                                                                                       :crossover_index_aspect_ratio] + aspect_ratio_parent2[
                                                                                                                        crossover_index_aspect_ratio:]
        child_2 = scales_parent2[:crossover_index_scalar] + scales_parent1[
                                                            crossover_index_scalar:] + aspect_ratio_parent2[
                                                                                       :crossover_index_aspect_ratio] + aspect_ratio_parent1[
                                                                                                                        crossover_index_aspect_ratio:]
        return child_1, child_2

    def generate_anchors(ratios=None, scales=None, base_size=anchor_base):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        """
        num_anchors = len(ratios) * len(scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def fitness_iou(individual, data):
        df, bounds = data[0]
        num_scales, num_aspect_ratio = len(bounds[0]), len(bounds[1])
        scales, aspect_ratios = individual[:num_scales], individual[num_scales:]

        # Check there are no repeated scales/aspect_ratios
        if len(np.unique(scales)) != len(scales) or len(np.unique(aspect_ratios)) != len(aspect_ratios):
            return 1

        anchors = generate_anchors(ratios=aspect_ratios, scales=scales)
        gt = df[['xmin', 'ymin', 'xmax', 'ymax']].values

        iou = compute_overlap(gt, anchors).max(axis=1).mean()

        return 1 - iou

    for region in tqdm(area_names):
        bounds = (bound_scale, bound_aspect_ratio)
        _data = data[data['region'] == region]
        print(region, len(_data))

        ga_data = [(_data, bounds)]

        ga = pyeasyga.GeneticAlgorithm(ga_data,
                                       population_size=1000,
                                       generations=5,
                                       crossover_probability=0.8,
                                       mutation_probability=0.2,
                                       elitism=True,
                                       maximise_fitness=False)
        ga.create_individual = create_individual
        ga.fitness_function = fitness_iou
        ga.mutate_function = mutate
        ga.crossover_function = crossover
        ga.run()
        ga.best_individual()
        best_values[region]['scale'] = sorted(ga.best_individual()[1][:n_scale])
        best_values[region]['aspect_ratio'] = sorted(ga.best_individual()[1][n_scale:])
        print(region, best_values[region], ga.best_individual()[0])

    fig, ax = plt.subplots(2, len(area_names), figsize=(5 * len(area_names), 6))
    for r, region in enumerate(area_names):
        __df = _df[_df['region'] == region]
        __df['aspect_ratio_best'] = closer_to(__df['aspect_ratio'], best_values[region]['aspect_ratio'])
        __df['scale_best'] = closer_to(__df['scale'], best_values[region]['scale'])
        ax[0, r].hist(__df.aspect_ratio_best, bins=np.asarray(
            [[b - 0.02, b + 0.02] for b in best_values[region]['aspect_ratio']]).flatten(),
                      color='r', density=True, rwidth=0.5,
                      label=str(['{:.2f}'.format(x) for x in best_values[region]['aspect_ratio']]))
        ax[0, r].hist(__df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=region_colors[r], density=True,
                      label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.aspect_ratio),
                                                                                        min(__df.aspect_ratio),
                                                                                        np.mean(__df.aspect_ratio)))
        ax[1, r].hist(__df.scale_best,
                      bins=np.asarray([[b - 0.01, b + 0.01] for b in best_values[region]['scale']]).flatten(),
                      color='r', density=True, rwidth=0.5,
                      label=str(['{:.2f}'.format(x) for x in best_values[region]['scale']]))
        ax[1, r].hist(__df.scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r], density=True,
                      label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.scale),
                                                                                        min(__df.scale),
                                                                                        np.mean(__df.scale)))
        ax[0, r].set_title("Optimized aspect ratio ({})".format(region))
        ax[0, r].legend()
        ax[1, r].set_title("Optimized scale ({})".format(region))
        ax[1, r].legend()
    fig.savefig('regions_description_best_distance.png')
    plt.show()


# optimize_params_by_iou_ga()

def optimize_params_by_differential_evolution():
    from scipy.optimize import differential_evolution
    import sys

    mode = 'focal'
    state = {'best_result': sys.maxsize}
    _area_names = ['MIDDLE', 'SKY', 'BOTTOM', 'TOP']
    _sample_frac = 1

    best_values = defaultdict(lambda: {'aspect_ratio': None, 'scale': None})

    data = pd.read_csv(OUTPUT_FILE)
    data = data[data['image_height'] != 1280]
    data = data[['xmax', 'xmin', 'ymax', 'ymin', 'image_height']]
    data['ycenter'] = (data['ymin'] + data['ymax']) / 2
    data['region'] = assign_area(data['ycenter'] / data['image_height'], text=True)
    x_factor = data['xmin'] + ((data['xmax'] - data['xmin']) / 2)
    y_factor = data['ymin'] + ((data['ymax'] - data['ymin']) / 2)
    data['xmax'] = data['xmax'] - x_factor
    data['xmin'] = data['xmin'] - x_factor
    data['ymax'] = data['ymax'] - y_factor
    data['ymin'] = data['ymin'] - y_factor

    data = data.sample(frac=_sample_frac)

    def generate_anchors(ratios=None, scales=None, base_size=anchor_base):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        """
        num_anchors = len(ratios) * len(scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def fitness_iou(individual, num_scales, df, state, mode='focal'):
        scales, aspect_ratios = individual[:num_scales], individual[num_scales:]

        # Check there are no repeated scales/aspect_ratios
        if len(np.unique(scales)) != len(scales) or len(np.unique(aspect_ratios)) != len(aspect_ratios):
            return 1

        anchors = generate_anchors(ratios=aspect_ratios, scales=scales)
        gt = df[['xmin', 'ymin', 'xmax', 'ymax']].values

        iou = compute_overlap(gt, anchors).max(axis=1).mean()

        if mode == 'avg':
            result = 1 - np.average(iou)
        elif mode == 'ce':
            result = np.average(-np.log(iou))
        elif mode == 'focal':
            result = np.average(-(1 - iou) ** 2 * np.log(iou))
        else:
            raise Exception('Invalid mode.')

        if result < state['best_result']:
            state['best_result'] = result

            print('Current best anchor configuration')
            print(f'Ratios: {sorted(aspect_ratios)}')
            print(f'Scales: {sorted(scales)}')

        return result

    for region in tqdm(_area_names):
        bounds = bound_scale + bound_aspect_ratio
        _data = data[data['region'] == region]
        print(region, len(_data))

        result = differential_evolution(
            lambda x: fitness_iou(x, n_scale, _data, state, mode),
            bounds=bounds, popsize=10, seed=1)

        if hasattr(result, 'success') and result.success:
            print('Optimization ended successfully!')
        elif not hasattr(result, 'success'):
            print('Optimization ended!')
        else:
            print('Optimization ended unsuccessfully!')
            print(f'Reason: {result.message}')

        values = result.x
        opt_scales = sorted(values[:n_scale])
        opt_aspect_ratio = sorted(values[n_scale:])

        best_values[region]['scale'] = opt_scales
        best_values[region]['aspect_ratio'] = opt_aspect_ratio
        print(region, best_values[region])

    fig, ax = plt.subplots(2, len(area_names), figsize=(5 * len(area_names), 6))
    for r, region in enumerate(area_names):
        __df = _df[_df['region'] == region]
        __df['aspect_ratio_best'] = closer_to(__df['aspect_ratio'], best_values[region]['aspect_ratio'])
        __df['scale_best'] = closer_to(__df['scale'], best_values[region]['scale'])
        ax[0, r].hist(__df.aspect_ratio_best, bins=np.asarray(
            [[b - 0.02, b + 0.02] for b in best_values[region]['aspect_ratio']]).flatten(),
                      color='r', density=True, rwidth=0.5,
                      label=str(['{:.2f}'.format(x) for x in best_values[region]['aspect_ratio']]))
        ax[0, r].hist(__df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=region_colors[r], density=True,
                      label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.aspect_ratio),
                                                                                        min(__df.aspect_ratio),
                                                                                        np.mean(__df.aspect_ratio)))
        ax[1, r].hist(__df.scale_best,
                      bins=np.asarray([[b - 0.01, b + 0.01] for b in best_values[region]['scale']]).flatten(),
                      color='r', density=True, rwidth=0.5,
                      label=str(['{:.2f}'.format(x) for x in best_values[region]['scale']]))
        ax[1, r].hist(__df.scale, bins=100, range=(0, 4), alpha=0.7, color=region_colors[r], density=True,
                      label=region + '\nMAX: {:.3f}\nMIN: {:.3f}\nMEAN: {:.3f}]'.format(max(__df.scale),
                                                                                        min(__df.scale),
                                                                                        np.mean(__df.scale)))
        ax[0, r].set_title("Optimized aspect ratio ({})".format(region))
        ax[0, r].legend()
        ax[1, r].set_title("Optimized scale ({})".format(region))
        ax[1, r].legend()
    fig.savefig('regions_description_best_distance.png')
    plt.show()

    _best_values_new = {
        'SKY': [0.06, 0.11036, 0.196657, 0.360178, 0.37788, 0.76, 1.82974],
        'TOP': [0.06974, 0.13669, 0.248809, 0.474911, 0.37657, 0.7821, 2.080698],
        'MIDDLE': [0.07002, 0.1447, 0.2881, 0.64271, 0.4081, 0.83919, 2.2826],
        'BOTTOM': [0.6936, 1.2549, 1.9353, 2.8861, 0.32057, 0.76915, 2.1001]
    }
    _best_values_old = {
        'SKY': [0.08, 0.21, 0.47, 1., 1.89, 3.35],
        'TOP': [0.09, 0.26, 0.59, 0.63, 1.49, 2.81],
        'MIDDLE': [0.1, 0.37, 0.97, 0.48, 1.28, 2.5],
        'BOTTOM': [1.2, 1.8, 2.46, 0.9, 2.07, 3.95]
    }
    _best_values_default = {
        'SKY': [0.25, 0.5, 1., 2., 0.5, 1., 2.],
        'TOP': [0.25, 0.5, 1., 2., 0.5, 1., 2.],
        'MIDDLE': [0.25, 0.5, 1., 2., 0.5, 1., 2.],
        'BOTTOM': [0.25, 0.5, 1., 2., 0.5, 1., 2.]
    }
    for area in area_names:
        print(area)

        avg_iou = 1 - fitness_iou(_best_values_default[area], 4, data[data['region'] == area], state={'best_result': 0},
                                  mode='avg')
        print('Default method average iou:', avg_iou)
        avg_iou = 1 - fitness_iou(_best_values_old[area], 3, data[data['region'] == area], state={'best_result': 0},
                                  mode='avg')
        print('Old method average iou:', avg_iou)
        avg_iou = 1 - fitness_iou(_best_values_new[area], 4, data[data['region'] == area], state={'best_result': 0},
                                  mode='avg')
        print('New method average iou:', avg_iou)


def ga_cyclist():
    from pyeasyga import pyeasyga
    _data = df[df['image_height'] != 1280.]

    def create_individual(data):
        _, bounds = data[0]
        return [np.random.randint(bound[0] * 100000, bound[1] * 100000) / 100000 for bound in bounds]

    def mutate(individual):
        """Reverse the bit of a random index in an individual."""
        mutate_index = random.randrange(len(individual))
        individual[mutate_index] = np.random.randint(1 * 100000, 10 * 100000) / 100000

    def crossover(parent_1, parent_2):
        parents = [parent_1, parent_2]
        child_1 = [random.choice(parents)[i] for i in range(len(parent_1))]
        child_2 = [random.choice(parents)[i] for i in range(len(parent_1))]
        return child_1, child_2

    def fitness_iou(individual, data):
        # y=1/(ax^b) | y: aspect ratio, x: scale
        df, bounds = data[0]
        _df = df[['scale', 'aspect_ratio', 'label']]
        num_anchors = len(_df)
        a, b, c, d, e = individual

        def y(x):
            if x < c:
                return 1 / (a * (x ** b))
            elif x < e:
                return d
            else:
                return 0

        vec_y = np.vectorize(y)
        _df['car_region'] = _df['aspect_ratio'] - vec_y(_df['scale'].values)
        _df = _df[_df['car_region'] > 0]

        num_vehicles = len(_df[_df['label'] == 1])
        num_pedestrians = len(_df[_df['label'] == 2])
        num_cyclists = len(_df[_df['label'] == 3])

        fitness = 2 * (154 * num_cyclists + 8 * num_pedestrians) / (
            num_vehicles)  # (154*num_cyclists + 8*num_pedestrians) / (num_vehicles)
        print(num_vehicles, num_pedestrians, num_cyclists, fitness, a, b, c, d, e)
        return fitness

    bounds_cyclist = [(1, 10), (1, 10), (1, 10), (1, 10), (1, 10)]

    ga_data = [(_data, bounds_cyclist)]

    ga = pyeasyga.GeneticAlgorithm(ga_data,
                                   population_size=200,
                                   generations=50,
                                   crossover_probability=0.8,
                                   mutation_probability=0.2,
                                   elitism=True,
                                   maximise_fitness=False)
    ga.create_individual = create_individual
    ga.fitness_function = fitness_iou
    ga.mutate_function = mutate
    ga.crossover_function = crossover
    ga.run()
    ga.best_individual()

    a, b, c, d, e = ga.best_individual()[1]

    def y_func(x):
        if x < c:
            return 1 / (a * (x ** b))
        elif x < e:
            return d
        else:
            return 0

    x = [s / 1000 for s in range(1, 5000)]
    y = [y_func(_x) for _x in x]
    fig, ax = plt.subplots(1, figsize=(12, 9))
    for i in range(3):
        l = boxlabels[i + 1]
        __df = df[df['label'] == (i + 1.)]
        __df = __df[__df['image_height'] == 1280.]
        ax.scatter(__df.scale, y=__df.aspect_ratio, marker='+', s=1, color=colors[i + 1], label=l)
    ax.plot(x, y, 'y--', linewidth=2)
    ax.set_title('Scale x Aspect ratio (1920x1289) [{}, {}, {}, {}, {}]'.format(a, b, c, d, e))
    ax.legend()
    ax.set_ylim((0, 50))
    ax.set_xlim((0, 5))
    fig.tight_layout()
    fig.show()

    return ga


def cyclist():
    fig, ax = plt.subplots(2, 3, figsize=(12, 9))
    for i in range(3):
        l = boxlabels[i + 1]
        __df = df[df['label'] == (i + 1.)]
        ax[0, i].hist(__df.aspect_ratio, bins=100, range=(0, 6), alpha=0.7, color=colors[i + 1], density=True)
        ax[1, i].hist(__df.scale, bins=100, range=(0, 3), alpha=0.7, color=colors[i + 1], density=True)
        ax[0, i].set_title('Aspect ratio ({})'.format(l))
        ax[1, i].set_title('Scale ({})'.format(l))
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(1, figsize=(12, 9))
    for i in range(3):
        l = boxlabels[i + 1]
        __df = df[df['label'] == (i + 1.)]
        ax.scatter(__df.scale, y=__df.aspect_ratio, marker='+', s=1, color=colors[i + 1], label=l)
    ax.set_title('Scale x Aspect ratio')
    ax.legend()
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(1, figsize=(12, 9))
    for i in range(3):
        l = boxlabels[i + 1]
        __df = df[df['label'] == (i + 1.)]
        ax.scatter(__df.scale, y=__df.aspect_ratio, marker='+', s=1, color=colors[i + 1], label=l)
    ax.set_ylim((0, 75))
    ax.set_xlim((0, 5))
    ax.set_title('Scale x Aspect ratio')
    ax.legend()
    fig.tight_layout()
    fig.show()

    a, b = 1.1273, 1.2723  # 1.9483, 1.0948
    x = [s / 100 for s in range(1, 500)]
    y = [1 / (a * (_x ** b)) for _x in x]
    fig, ax = plt.subplots(1, figsize=(12, 9))
    for i in range(3):
        l = boxlabels[i + 1]
        __df = df[df['label'] == (i + 1.)]
        __df = __df[__df['image_height'] == 1280.]
        ax.scatter(__df.scale, y=__df.aspect_ratio, marker='+', s=1, color=colors[i + 1], label=l)
    ax.plot(x, y, 'y--', linewidth=2)
    ax.set_title('Scale x Aspect ratio (1920x1289)')
    ax.legend()
    ax.set_ylim((0, 50))
    ax.set_xlim((0, 5))
    fig.tight_layout()
    fig.show()

    fig, ax = plt.subplots(1, figsize=(12, 9))
    for i in range(3):
        l = boxlabels[i + 1]
        __df = df[df['label'] == (i + 1.)]
        __df = __df[__df['image_height'] == 886.]
        ax.scatter(__df.scale, y=__df.aspect_ratio, marker='+', s=1, color=colors[i + 1], label=l)
    ax.set_title('Scale x Aspect ratio (1920x886)')
    ax.legend()
    ax.set_ylim((0, 75))
    ax.set_xlim((0, 5))
    fig.tight_layout()
    fig.show()

    # Cyclist
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(_df[_df['label'] == 3.]['xcenter'], y=_df[_df['label'] == 3.]['ycenter'], marker='+', s=1)
    ax[0].set_ylim((0, 1))
    ax[0].invert_yaxis()
    ax[0].set_title("Cyclist (All images)")
    __df = _df[_df['image_height'] == 1280.]
    ax[1].scatter(__df[__df['label'] == 3.]['xcenter'], y=__df[__df['label'] == 3.]['ycenter'] * 1280, marker='+', s=1)
    ax[1].set_ylim((0, 1280))
    ax[1].invert_yaxis()
    ax[1].set_title("Cyclist (1920x1280)")
    __df = _df[_df['image_height'] == 886.]
    ax[2].scatter(__df[__df['label'] == 3.]['xcenter'], y=__df[__df['label'] == 3.]['ycenter'] * 886, marker='+', s=1)
    ax[2].set_ylim((0, 886))
    ax[2].invert_yaxis()
    ax[2].set_title("Cyclist (1920x886)")
    fig.tight_layout()
    fig.savefig('cyclist.png')
    fig.show()

    # Pedestrian
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(_df[_df['label'] == 2.]['xcenter'], y=_df[_df['label'] == 2.]['ycenter'], marker='+', s=1)
    ax[0].set_ylim((0, 1))
    ax[0].invert_yaxis()
    ax[0].set_title("Pedestrian (All images)")
    __df = _df[_df['image_height'] == 1280.]
    ax[1].scatter(__df[__df['label'] == 2.]['xcenter'], y=__df[__df['label'] == 2.]['ycenter'] * 1280, marker='+', s=1)
    ax[1].set_ylim((0, 1280))
    ax[1].invert_yaxis()
    ax[1].set_title("Pedestrian (1920x1280)")
    __df = _df[_df['image_height'] == 886.]
    ax[2].scatter(__df[__df['label'] == 2.]['xcenter'], y=__df[__df['label'] == 2.]['ycenter'] * 886, marker='+', s=1)
    ax[2].set_ylim((0, 886))
    ax[2].invert_yaxis()
    ax[2].set_title("Pedestrian (1920x886)")
    fig.tight_layout()
    fig.savefig('pedestrian.png')
    fig.show()

    # Vehicle
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(_df[_df['label'] == 1.]['xcenter'], y=_df[_df['label'] == 1.]['ycenter'], marker='+', s=1)
    ax[0].set_ylim((0, 1))
    ax[0].invert_yaxis()
    ax[0].set_title("Vehicle (All images)")
    __df = _df[_df['image_height'] == 1280.]
    ax[1].scatter(__df[__df['label'] == 1.]['xcenter'], y=__df[__df['label'] == 1.]['ycenter'] * 1280, marker='+', s=1)
    ax[1].set_ylim((0, 1280))
    ax[1].invert_yaxis()
    ax[1].set_title("Vehicle (1920x1280)")
    __df = _df[_df['image_height'] == 886.]
    ax[2].scatter(__df[__df['label'] == 1.]['xcenter'], y=__df[__df['label'] == 1.]['ycenter'] * 886, marker='+', s=1)
    ax[2].set_ylim((0, 886))
    ax[2].invert_yaxis()
    ax[2].set_title("Vehicle (1920x886)")
    fig.tight_layout()
    fig.savefig('vehicle.png')
    fig.show()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(_df[_df['label'] == 3.]['xcenter'], y=_df[_df['label'] == 3.]['ycenter'], marker='+', s=1)
    ax[0].set_ylim((0, 1))
    ax[0].invert_yaxis()
    ax[0].set_title("Cyclist (All images)")
    __df = _df[_df['image_height'] == 1280.]
    ax[1].scatter(__df[__df['label'] == 3.]['xcenter'], y=__df[__df['label'] == 3.]['ycenter'] * 1280, marker='+', s=1)
    ax[1].set_ylim((0, 1280))
    ax[1].invert_yaxis()
    ax[1].set_title("Cyclist (1920x1280)")
    __df = _df[_df['image_height'] == 886.]
    ax[2].scatter(__df[__df['label'] == 3.]['xcenter'], y=__df[__df['label'] == 3.]['ycenter'] * 886, marker='+', s=1)
    ax[2].set_ylim((0, 886))
    ax[2].invert_yaxis()
    ax[2].set_title("Cyclist (1920x886)")
    fig.tight_layout()
    fig.savefig('cyclist.png')
    fig.show()

# cyclist()
