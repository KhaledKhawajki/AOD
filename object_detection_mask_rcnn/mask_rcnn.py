import numpy as np
import random
import cv2
from PIL import Image


class DetectionModel:
	def __init__(self, value = 0.3, thresh = 0.3):
		self.value = value
		self.thresh = thresh
		labelsPath = "object_detection_mask_rcnn/object_detection_classes_coco.txt"
		self.LABELS = open(labelsPath).read().strip().split("\n")
		colorsPath = "object_detection_mask_rcnn/colors.txt"
		self.COLORS = open(colorsPath).read().strip().split("\n")
		self.COLORS = [np.array(c.split(",")).astype("int") for c in self.COLORS]
		self.COLORS = np.array(self.COLORS, dtype="uint8")
		weightsPath = "object_detection_mask_rcnn/frozen_inference_graph.pb"
		configPath = "object_detection_mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
		self.net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

	def detect(self,image):
		W = image.shape[0]
		H = image.shape[1]
		blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
		self.net.setInput(blob)
		(boxes, masks) = self.net.forward(["detection_out_final", "detection_masks"])
		objects = []
		bbxs = []
		for i in range(0, boxes.shape[2]):
			classID = int(boxes[0, 0, i, 1])
			confidence = boxes[0, 0, i, 2]
			if confidence > self.value:
				box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				objects.append(self.LABELS[classID])
				bbxs.append((startX, startY, endX, endY))
		return objects,bbxs
