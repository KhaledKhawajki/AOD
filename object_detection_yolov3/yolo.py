import numpy as np
import cv2

class YOLOV3:
    def __init__(self):
        self.confidence = 0.35
        self.threshold = 0.3
        self.labels = open('object_detection_yolov3/coco.names').read().strip().split('\n')
        # Create a list of colors for the labels
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')
        # Load weights using OpenCV
        self.net = cv2.dnn.readNetFromDarknet('object_detection_yolov3/yolov3.cfg', 'object_detection_yolov3/yolov3.weights')

        # Get the ouput layer names
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


    def extract_boxes_confidences_classids(self,outputs, confidence, width, height):
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:
                # Extract the scores, classid, and the confidence of the prediction
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]

                # Consider only the predictions that are above the confidence threshold
                if conf > confidence:
                    # Scale the bounding box back to the size of the image
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, w, h = box.astype('int')

                    # Use the center coordinates, width and height to get the coordinates of the top left corner
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))
                    classIDs.append(classID)

        return boxes, confidences, classIDs


    def draw_bounding_boxes(self,image, boxes, confidences, classIDs, idxs):
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw the bounding box and label on the image
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image


    def make_prediction(self,image):
        height, width = image.shape[:2]

        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        # Extract bounding boxes, confidences and classIDs
        boxes, confidences, classIDs = self.extract_boxes_confidences_classids(outputs, self.confidence, width, height)

        # Apply Non-Max Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        return boxes, confidences, classIDs, idxs

    def detect(self,image):
        boxes, confidences, classIDs, idxs = self.make_prediction(image)
        objects = []
        if len(idxs) != 0:
            for i in idxs:
                objects.append(self.labels[classIDs[i[0]]])
        self.draw_bounding_boxes(image, boxes,confidences,classIDs,idxs)
        return objects
