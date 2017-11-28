import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob, scipy
import heapq
import pandas as pd
#please ad ./evaluate file into your system path
sys.path.insert(0, './evaluate')#DL
import evaluate#DL

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

def vis_detections(im, class_name, dets, ax, cls_ind, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return None, None #DL

    loc = []#DL
    cla = []#DL
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        #loc.append(bbox)#DL
        #cla.append(cls_ind)#DL
        #ax.add_patch(
        #    plt.Rectangle((bbox[0], bbox[1]),
        #                  bbox[2] - bbox[0],
        #                  bbox[3] - bbox[1], fill=False,
        #                  edgecolor='red', linewidth=3.5)
        #)
        #ax.text(bbox[0], bbox[1] - 2,
        #        '{:s} {:.3f}'.format(class_name, score),
        #        bbox=dict(facecolor='blue', alpha=0.5),
        #        fontsize=14, color='white')
        loc.append(np.array([bbox[0],bbox[1],bbox[2],bbox[3]]))#DL
        cla.append(cls_ind)#DL
        
    #ax.set_title(('{} detections with '
    #              'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                              thresh),
    #             fontsize=14)
        
    #print loc
    #print cla
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()
    return loc, cla #DL


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, features = im_detect(sess, net, im)##########################################!!!!!!!!!!!!!!!!!!!!!!!!!

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    
    
    #=========================================
    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    ax = 0
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.5
    pb = [] #prob list for all class and obj
    #location = []
    whichclass = []
    check = 0
    double_check = 0
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
        #print len(pb)
        #global location
        c, d = vis_detections(im, cls, dets, ax, cls_ind, thresh=CONF_THRESH) #DL
        if c != None and d != None: #DL
            if check == 0 and len(c) != 1:
                location = c[0]
                #whichclass = d[0]
                for x in range(len(c)-1):
                    location = np.vstack((location,c[x+1]))
                    #whichclass = np.vstack((whichclass,d[x+1]))
                check = check + 1
            elif check == 0 :
                location = np.array(c)
                #whichclass = np.array(d)
                check = check + 1
            else:
                location = np.vstack((location,c))
                #whichclass = np.vstack((whichclass,d))
            whichclass.extend(d)#DL
            #whichclass = np.reshape(whichclass,[len(whichclass),1])
        elif c == None and d == None:
            double_check = double_check + 1
            if double_check == 20:
                location = np.array([])
                #whichclass = np.array([])
    new = []
    #for number in range(len(whichclass)):
    #size = len(whichclass)
    new = np.reshape((whichclass), [len(whichclass),1]) #len(whichclass)
    #print (location) #DL
    #print (new) #DL
    
    return location, new #whichclass #DL


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='PVAnet_test')#net_test PVA
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
    #im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    #for i in xrange(2):
    #    _, _, _ = im_detect(sess, net, im)
    '''
    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))
    '''
    
    #DL
    df_test = pd.read_pickle('/home/henry/DL/try/a_py2.pkl')
    im_names = []
    for num in range(len(df_test['image_name'])):
        im_names.append('/home/henry/DL/try/JPEGImages/'+ str(df_test['image_name'][num]))
    
    final_loc = []#DL
    final_cls = []#DL
    x = 0
    for im_name in im_names:
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print ('Demo for {:s}'.format(im_name))
        print (im_name)
        c, d = demo(sess, net, im_name)
        final_loc.append(c)
        final_cls.append(d)
        
    print final_loc
    print final_cls
    evaluate.evaluate(final_loc,final_cls)

    #plt.show()

