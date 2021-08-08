from lib.dataset import ImageDataset
from lib.dataset.Dukereid import DukeMTMCreID
from lib.dataset.market1501 import Market1501
import numpy as np
import paddle.fluid as fluid

if __name__ == "__main__":
    Epochs = 20
    batch_size = 10
    device = "cpu"
    place = fluid.CPUPlace()  # for cuda use: fluid.CUDAPlace(0)
    fluid.enable_imperative(place)


