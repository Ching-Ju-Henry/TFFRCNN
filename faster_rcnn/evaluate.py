
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[3]:

DTYPE = np.float
def bbox_overlaps(boxes,query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=DTYPE)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


# In[43]:

def evaluate(predict_boxes, predict_cls):
    df_test_answer = pd.read_pickle('/home/henry/pva/TFFRCNN/faster_rcnn/test_data_answer2.pkl')
    gt_boxes = df_test_answer['boxes'].as_matrix()
    gt_cls = df_test_answer['gt_classes'].as_matrix()
    
    threshold = 0.01
    f1 = []
    for img in range(len(df_test_answer)):
        hit = 0
        print('image #', img)
        print(predict_boxes[img], gt_boxes[img])
        overlaps = bbox_overlaps(predict_boxes[img], gt_boxes[img])
        print(overlaps)
        for box in range(len(predict_boxes[img])):
            for gt in range(len(gt_boxes[img])):
                print ('=========>>>>>>>>>',predict_cls[img][box],gt_cls[img][gt])
                if overlaps[box, gt] > threshold and predict_cls[img][box] == gt_cls[img][gt]:
                    gt_cls[img][gt] = 0
                    hit += 1.0
        pre = 0 if len(predict_boxes[img]) == 0 else hit/len(predict_boxes[img])
        rec = 0 if len(gt_boxes[img]) == 0 else hit/len(gt_boxes[img])
        if pre+rec == 0:
            f = 0
        else:
            f = 2*((pre*rec)/(pre+rec))
        f1.append(f)
        
    result = [[str(i), f1[i]] for i in range(len(f1))]
    df_output = pd.DataFrame(result, columns=['Id', 'F1Score'])
    df_output.to_csv('output.csv', index=False)


# In[ ]:



