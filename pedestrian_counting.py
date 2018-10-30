#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf
import time

# Object detection imports
from utils import backbone
from api import object_counting_api

#if tf.__version__ < '1.4.0':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

input_video = "pedestrian_survaillance.mp4"
#input_video="vehicle_survaillance.mp4"
#input_video = "supermarket.mp4"
#input_video="bps.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
model_name = 'ssd_mobilenet_v1_coco_2018_01_28'
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28')
#detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')
#detection_graph, category_index = backbone.set_model('faster_rcnn_resnet101_coco_2018_01_28')

fps = 12 # change it with your input video fps
width = 626 # 626 change it with your input video width
height = 360 # 360 change it with your input vide height
is_color_recognition_enabled = 0
targeted_object="person"

start = time.time()
#object_counting_api.cumulative_object_counting_x_axis_bps(model_name,input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, 400) # counting all the objects
object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, 400) # counting all the objects
#object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, 200) # counting all the objects
#object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled,targeted_object, fps, width, height) # counting target the objects
elap = time.time() - start
print("Fim da execução. Tempo total: %3.4i" % elap)
output = input_video.strip('.mp4') + "_output.avi"
print ("Gerado arquivo de saida de nome: ", output)

