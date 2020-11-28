'''
Created on Aug 24, 2020

@author: Michal.Busta at gmail.com
'''

import random 
from tensorpack.dataflow.imgaug import ImageAugmentor
import cv2 

class CoarseDropout(ImageAugmentor):
  """ Random rotate and crop the largest possible rect without the border
      This will produce images of different shapes.
  """
  def __init__(self, min_holes = 0, max_holes = 8, min_size = 1, max_size = 8):
    super(CoarseDropout, self).__init__()
    self._init(locals())
    self.min_holes = min_holes
    self.max_holes = max_holes
    self.min_size = min_size
    self.max_size = max_size
    self.fillvalue = 0

  def _get_augment_params(self, img):
  
    height, width = img.shape[:2]
    holes = []
    for _n in range(random.randint(self.min_holes, self.max_holes)):
        hole_height = random.randint(self.min_size, self.max_size)
        hole_width = random.randint(self.min_size, self.max_size)

        y1 = random.randint(0, height - hole_height)
        x1 = random.randint(0, width - hole_width)
        y2 = y1 + hole_height
        x2 = x1 + hole_width
        holes.append((x1, y1, x2, y2))
        
    return holes

  def _augment(self, img, holes):
    
    for x1, y1, x2, y2 in holes:
        img[y1:y2, x1:x2] = self.fillvalue
    return img
  
class CannyAug(ImageAugmentor):
  """ Random rotate and crop the largest possible rect without the border
      This will produce images of different shapes.
  """
  def __init__(self):
    super(CannyAug, self).__init__()
    self._init(locals())
    

  def _get_augment_params(self, img):
    
    p1 = random.randint(90, 130)
    p2 = random.randint(190, 220)
    return p1, p2

  def _augment(self, img, param):
    
    p1, p2 = param
    edges = cv2.Canny(img,p1,p2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    

