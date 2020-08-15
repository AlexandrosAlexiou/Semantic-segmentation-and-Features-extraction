import os
import tarfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from os import path
from sklearn.decomposition import PCA  # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.cluster import KMeans  # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

import tensorflow as tf

# disable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    DEEP_FEATURES_TENSOR_NAME = 'aspp0/Conv2D:0'  # deep features output layer name
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        # prints operations in order to pick one as output to get the deep features
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
            # show graph operations
            '''for op in tf.get_default_graph().get_operations():
                print(str(op.name))
            print('\n\n')'''

        self.sess = tf.Session(graph=self.graph)

        if not path.exists('/tmp/tftut/1'):
            # Write graph on disk. Visualize: $ tensorboard --logdir /tmp/tftut/1 --host localhost
            writer = tf.summary.FileWriter('/tmp/tftut/1')
            writer.add_graph(self.sess.graph)

    def run(self, image):
        """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
        )
        seg_map = batch_seg_map[0]
        print(np.asarray(resized_image).shape)
        return resized_image, seg_map

    def run_deep_features(self, image):
        """Runs inference on a single image with a specific layer output.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      That specific layer's tensor output
    """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        result = self.sess.run(
            self.DEEP_FEATURES_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
        )
        return resized_image, result[0]


class ColorMap:

    def __init__(self):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.

         Returns:
           A Colormap for visualizing segmentation results.
         """
        self.colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                self.colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

    def label_to_color_image(self, label):
        """Adds color defined by the dataset colormap to the label.

      Args:
        label: A 2D array with integer type, storing the segmentation label.

      Returns:
        result: A 2D array with floating type. The element of the array
          is the color indexed by the corresponding element in the input label
          to the PASCAL color map.

      Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
          map maximum entry.
      """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        # print(colormap)
        if np.max(label) >= len(self.colormap):
            raise ValueError('label value too large.')

        return self.colormap[label]


class SemanticSegmentation:

    def __init__(self):
        self.LABEL_NAMES = np.asarray([
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
        ])
        self.colormap = ColorMap()
        self.FULL_LABEL_MAP = np.arange(len(self.LABEL_NAMES)).reshape(len(self.LABEL_NAMES), 1)
        # print(FULL_LABEL_MAP)
        self.FULL_COLOR_MAP = self.colormap.label_to_color_image(self.FULL_LABEL_MAP)
        # print(FULL_COLOR_MAP)

        self.MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug',
        # 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

        self._DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        self._MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }

        self._TARBALL_NAME = self._MODEL_URLS[self.MODEL_NAME]
        model_dir = './deeplap_parameters'

        download_path = os.path.join(model_dir, self._TARBALL_NAME)

        # download deeplab parameters if they do not exist
        if not path.exists(download_path):
            print('downloading model at {}, this might take a while...'.format(download_path))
            urllib.request.urlretrieve(self._DOWNLOAD_URL_PREFIX + self._MODEL_URLS[self.MODEL_NAME], download_path)
            print('download completed! loading DeepLab model...')

        print('loading model...')
        self.MODEL = DeepLabModel(download_path)
        print('model loaded successfully!')

    def run_segmentation_visualization(self, url):
        """Inferences DeepLab model and visualizes result."""
        try:
            original_im = Image.open(url)
        except FileNotFoundError:
            print('Cannot open image. Please check path: ' + url)
            return

        print('running deeplab on image %s...' % url)

        resized_im, seg_map = self.MODEL.run(original_im)

        self.vis_segmentation(resized_im, seg_map, os.path.basename(url))

        print('running deeplab with %s as output on %s...\n' % (self.MODEL.DEEP_FEATURES_TENSOR_NAME, url))
        resized_im, deep_features = self.MODEL.run_deep_features(original_im)

        self.vis_deep_features(resized_im, deep_features, os.path.basename(url))

    def vis_segmentation(self, image, seg_map, url_basename):
        """Visualizes input image, segmentation map and overlay view."""
        plt.figure(figsize=(20, 10))
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

        # first subplot is the original image
        plt.subplot(grid_spec[0])
        plt.imshow(image)
        plt.axis('off')
        plt.title('input image')

        # second subplot is the segmentation image
        plt.subplot(grid_spec[1])
        seg_image = self.colormap.label_to_color_image(seg_map).astype(np.uint8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('segmentation map')

        # third subplot is the segmentation image on top of the original image with lower opacity to form the overlay
        plt.subplot(grid_spec[2])
        plt.imshow(image)
        plt.imshow(seg_image, alpha=0.7)
        plt.axis('off')
        plt.title('segmentation overlay')

        # fourth subplot is the legend for the segmentation colors
        unique_labels = np.unique(seg_map)
        # print(unique_labels)
        ax = plt.subplot(grid_spec[3])
        plt.imshow(self.FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
        plt.title('Legend')
        ax.yaxis.tick_right()
        plt.yticks(range(len(unique_labels)), self.LABEL_NAMES[unique_labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')
        # save plot on disk
        plt.savefig('./results/seg_' + url_basename)
        plt.show()

    def vis_deep_features(self, image, deep_features, url_basename):
        """Visualizes extracted deep features tensor"""
        plt.figure(figsize=(20, 10))
        grid_spec = gridspec.GridSpec(1, 3)

        # first subplot is the original image
        ax0 = plt.subplot(grid_spec[0])
        plt.imshow(image)
        plt.axis('off')
        ax0.set_title('{}'.format(url_basename), fontsize=15)

        # second subplot is the deep features tensor visualization
        ax1 = plt.subplot(grid_spec[1])
        ax1.set_title('{} deep features'.format(url_basename), fontsize=15)
        plt.axis('off')
        deepfeats_array = np.array(deep_features)
        N = deepfeats_array.shape[0] * deepfeats_array.shape[1]
        C = deepfeats_array.shape[-1]
        X = np.reshape(deepfeats_array, [N, C])

        Xreduced = PCA(n_components=3).fit_transform(X)
        deep_features_reduced = np.reshape(Xreduced, [deep_features.shape[0], deep_features.shape[1], 3])
        plt.imshow(deep_features_reduced)

        """Applies and visualizes binarization to the deep features tensor"""
        K = deep_features.shape[0] * deep_features.shape[1]
        L = deep_features.shape[-1]
        M = np.reshape(deep_features, [K, L])
        Mreduced = PCA(n_components=8).fit_transform(M)

        kmeans = KMeans(n_clusters=2, max_iter=15).fit(Mreduced)
        # print(kmeans)

        prediction = kmeans.predict(Mreduced)

        # print(prediction.shape)

        res = np.reshape(prediction, [65, 65])

        # third subplot is the binarized image
        ax2 = plt.subplot(grid_spec[2])
        plt.imshow(res)
        plt.axis('off')
        ax2.set_title('{} binarized'.format(url_basename), fontsize=15)
        plt.savefig('./results/deepfeats_' + url_basename)
        plt.show()

