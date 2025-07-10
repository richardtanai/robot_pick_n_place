import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import cv2
import numpy as np
import random


class MaskRCNN_MODEL():
    def __init__(self) -> None:
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def get_prediction(self, img, threshold=0.5):
        transform = T.Compose([T.ToTensor()]) # Turn the image into a torch.tensor
        img = transform(img)
        # img = img.cuda() # Only if GPU, otherwise comment this line
        pred = self.model([img]) # Send the image to the model. This runs on CPU, so its going to take time
        #Let's change it to GPU
        # pred = pred.cpu() # We will just send predictions back to CPU
        # Now we need to extract the bounding boxes and masks
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
        pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return masks, pred_boxes, pred_class

    def random_color_masks(self, image):
        # I will copy a list of colors here
        colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        r[image==1], g[image==1], b[image==1] = colors[random.randrange(0, 10)]
        colored_mask = np.stack([r,g,b], axis=2)
        return colored_mask
    
    def instance_segmentation(self, img, threshold=0.5, rect_th=1,
                          text_size=1, text_th=1, url=False):
        masks, boxes, pred_cls = self.get_prediction(img, threshold=threshold)
    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For working with RGB images instead of BGR
        for i in range(len(masks)):
            rgb_mask = self.random_color_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            pt1 = tuple(int(x) for x in boxes[i][0])
            pt2 = tuple(int(x) for x in boxes[i][1])
            cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        return img, pred_cls, masks
    
    def draw_mask_color(self, img, pred_cls, boxes, masks):
        for i in range(len(masks)):
            rgb_mask = self.random_color_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        return img

    def draw_labels(self, img, pred_cls, boxes, masks, text_size=1, text_th=1):
        # For working with RGB images instead of BGR
        for i in range(len(masks)):
            pt1 = tuple(int(x) for x in boxes[i][0])
            pt2 = tuple(int(x) for x in boxes[i][1])
            cv2.putText(img, pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
        return img
    
    def draw_bbox(self, img, pred_cls, boxes, masks, rect_th=1):
        for i in range(len(masks)):
            pt1 = tuple(int(x) for x in boxes[i][0])
            pt2 = tuple(int(x) for x in boxes[i][1])
            cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=rect_th)
        return img
        

    
    def process_image(self, img, is_draw_labels=True, is_draw_boxes=True, is_draw_masks=True):
        
        pred_masks, pred_boxes, pred_class= self.get_prediction(img)
        # convert to rgb to enable manipulation
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if is_draw_labels:
            self.draw_labels(img, pred_class, pred_boxes, pred_masks)
        
        if is_draw_boxes:
            self.draw_bbox(img, pred_class, pred_boxes, pred_masks)

        if is_draw_masks:
            self.draw_mask_color(img, pred_class, pred_boxes, pred_masks)
        
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, pred_class, pred_boxes, pred_masks