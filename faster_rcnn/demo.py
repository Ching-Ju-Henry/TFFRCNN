import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob, scipy
import heapq

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


#CLASSES = ('__background__','person','bicycle','motorbike','car','bus')

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, features = im_detect(sess, net, im)##########################################!!!!!!!!!!!!!!!!!!!!!!!!!
    #print "feature ===============================>",features.shape
    #print "boxes ===============================>",boxes.shape
    #print "scores ===============================>",scores.shape

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    #================================
    n_input = 4096 # fc6 or fc7(1*4096)
    n_detection = 6 # number of object of each image (include image features)
    batch_size = 1
    n_frames = 1 # number of frame in each video 
    
    img_features = features[0]
    det_features = features[1:]
    scores_obj = scores[1:,1:]
    CONF_THRESH = 0.8 #0.8
    NMS_THRESH = 0.3
    dets_all_temp = np.zeros(shape=(1,1))
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
      cls_ind += 1 
      cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
      cls_scores = scores[:, cls_ind]
      dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
      keep_nms = nms(dets, NMS_THRESH)
      dets2 = ((dets[keep_nms, :][:n_detection-1])[:,-1]).reshape((-1, 1))
      dets_all_temp = np.vstack([dets_all_temp,dets2])
      
    dets_top_all = np.zeros(shape=(1,n_detection-1))
    features_sort = np.zeros(shape=(1,n_input))
    dets_all = np.sort(dets_all_temp[1:,-1])[-n_detection+1:]
    
    for i in range(len(dets_all)):
      get_scores_index = np.argwhere(dets_all[i] == scores)
      features_temp = features[get_scores_index[0,0]] 
      features_sort = np.vstack([features_sort,features_temp])
      #boxes_test = boxes[get_scores_index[0,0], 4*get_scores_index[0,1]:4*(get_scores_index[0,1]+1)] 
      #dets_temp = np.hstack((boxes_test, dets_all[i]))
      #dets_top_all = scipy.sparse.vstack([dets_top_all,dets_temp]) 
    new_bboxes = dets_top_all[1:]
    features_sort = features_sort[1:]

    features = np.vstack([img_features,features_sort])
    features = features.reshape([n_detection,n_input])#[batch_size,n_frames,n_detection,n_input]
    
    #=========================================
    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8#0.8
    NMS_THRESH = 0.3
    pb = [] #prob list for all class and obj
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        for num in range(dets.shape[0]):
            pb.append(dets[:, -1][num])
        print len(pb)
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
    
    
    # taking 5 dets (higher prob)
    need_det = []    
    maxvalues = heapq.nlargest(5, pb)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        for number in range(dets.shape[0]):
            for check in range(len(maxvalues)):
                if dets[:, -1][number] == maxvalues[check]:
                    need_det.append(dets[number])
    need_det = np.reshape(need_det, [5,5])
    print need_det.shape
    return features, need_det


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='PVAnet_test')#VGGnet_test
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    '''
    if args.model == ' ' or not os.path.exists(args.model):
        print ('current path is ' + os.path.abspath(__file__))#########################################
        raise IOError(('Error: Model not found.\n'))
    '''
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, args.model)


    
    print (' done.')

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _, _ = im_detect(sess, net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    det = []
    feature = []
    x = 0
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        a, b = demo(sess, net, im_name)
        feature.append(a)
        det.append(b)
        
    det = np.reshape(det, [1,10,5,5])
    feature = np.reshape(feature, [1,10,6,4096])
    np.savez("/home/henry/pva/TFFRCNN/output/parameter_top5.npz", ID=['00002'], labels=[[0,1]], det=det, data=feature)

    plt.show()

