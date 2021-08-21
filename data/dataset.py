import paddle
import numpy as np


# images dataset with ROIs, masks and key points (optional) annotations
class ImagesDataset(paddle.io.Dataset):
    def __init__(self, name,
                 augumentator=None# used in trainning process
                 ):
        super(ImagesDataset, self).__init__()
        self._name = name
        self._class_names = []
        self._data_path = None
        self._augumented = True
        self.augmentator = augumentator

    # ===== paddle interface : @todo TODO
    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))


    # ===== PaddleDetection data loader interface
    def check_or_download_dataset(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    def parse_dataset(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    # ===== general dataset function to implement for instances recognition training task
    def load_image_annotations(self, image_id, mask_on, augumented):
        """
            reference: https://github.com/aleju/imgaug

            One of the most difficult things to augument an image is to obtain augumented images and the correspoinding annotations which
            include masks, bounding boxes and labels.

            The later requires us to apply the same transfromation upon images to annotations data like masks and bounding boxes if that changes
            the geometry of images. As for some special tasks like human face recognition, we need special data as part of annotations like key points, rotated bounding boxes.

            Transformation affects geometry

            @return (image, bbounding boxes, masks, class_ids)
        """
        raise Exception("Not Implemented Yet!")

    def get_minibatch(self, images_info, batch_size):
        """
        Since I was using keras to train the model, the `get_mini_batch` function should work exactly as what
        [ImageDataGenerator](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L233) does.
        The data generator will generate batches of non-symbolic tensor data with real-time data augumentation.

        It is an common mistake to use `ImageDataGenerator` with images with annotations of multiple rois because `ImageDataGenerator`
        does not care about shape, coordiantes of annotations.

        @return python generator. Each time calling next() on it, the generator returns two lists, inputs and outputs.

        By default in Keras, `fit` function is used to prepare statistics computed from input data and use them to normalize
        the generted batch of data:
        - featurewise_center: means on batch, rowaxis and column axis
        - featurewise_std_normalization: std on batch, rowaxis and column axis
        - zca_whiting: compute pca components

        references: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py#L1630
                    https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py

        """
        raise Exception("Not Implemented Yet!")

    def get_sample(self, index):
        pass

    def __str__(self):
        return "ImagesDataset"