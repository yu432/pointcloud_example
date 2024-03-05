# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import os
from urllib.request import urlretrieve

import open3d as o3d
from core.deep_global_registration import DeepGlobalRegistration
from config import get_config

BASE_URL = "http://node2.chrischoy.org/data/"
DOWNLOAD_LIST = [
    #(BASE_URL + "datasets/registration/", "redkitchen_000.ply"),
    #(BASE_URL + "datasets/registration/", "redkitchen_010.ply"),
    (BASE_URL + "projects/DGR/", "ResUNetBN2C-feat32-3dmatch-v0.3.pth")
]

# Check if the weights and file exist and download

if __name__ == '__main__':
  config = get_config()
  if config.weights is None:
    config.weights = DOWNLOAD_LIST[-1][-1]

  pcd0_floor = o3d.io.read_point_cloud("pointclouds/5/floor_cloud.pcd")
  pcd0_core = o3d.io.read_point_cloud("pointclouds/5/core_cloud.pcd")
  pcd1_floor  = o3d.io.read_point_cloud("pointclouds/6/floor_cloud.pcd")
  pcd1_core = o3d.io.read_point_cloud("pointclouds/6/core_cloud.pcd")
  # preprocessing
  pcd0 = pcd0_floor + pcd0_core
  pcd0.estimate_normals()
  pcd1 = pcd1_floor + pcd1_core
  pcd1.estimate_normals()

  # registration
  dgr = DeepGlobalRegistration(config)
  T01 = dgr.register(pcd0, pcd1)

  o3d.visualization.draw_geometries([pcd0, pcd1])

  pcd0.transform(T01)
  print(T01)

  o3d.visualization.draw_geometries([pcd0, pcd1])
