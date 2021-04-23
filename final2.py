import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import pynput
import math
from pynput.mouse import Button, Controller
mouse = Controller()

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
colo = (0, 255, 0)
cole = (255,0,0)
thickness = 2
error = 0

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            #cv2.line(im0, (100,200),(100,0),cole, thickness)
            cv2.circle(im0, (100,100), radius=5, color=(0, 0, 255), thickness=-1)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)

                        #print(label)
                        #print(names[int(cls)])
                    if names[int(cls)] =="person":

                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))


                        #cv2.line(im0, c1, c2,colo, thickness)
                        #cv2.line(im0, (round(c1[0]/2),c1[1]),(round(c2[0]/2),c2[1]),colo, thickness)
                        #cv2.line(im0, (round((c2[0]+c1[0])/2),c1[1]),(round((c2[0]+c1[0])/2),c2[1]),colo, thickness)
                        #cv2.line(im0, (c1[0],round((c2[1]+c1[1])/2)),(c2[0],round((c2[1]+c1[1])/2)),colo, thickness)
                        #this line above is preserverd for the center of the box
                        #cv2.line(im0, (c1[0],round((c2[1]+c1[1])/3)),(c2[0],round((c2[1]+c1[1])/3)),cole, thickness)
                        #time.sleep(0.5)
                        #print(c1,c2)
                        #print((c2[0]-c1[0])/2)

#important code supressed
                        cv2.circle(im0, (round((c2[0]+c1[0])/2),round((c2[1]+c1[1])/3)), radius=5, color=(0, 0, 255), thickness=-1)
                        #cv2.line(im0, (round((c2[0]+c1[0])/2), round((c2[1]+c1[1])/3)),colo, thickness)
                        cv2.line(im0, (round((c2[0]+c1[0])/2),round((c2[1]+c1[1])/3)), (100,100),colo,thickness)



                        print(round((c2[0]+c1[0])/2)-100)
                        e1 = error
                        error = round((c2[0]+c1[0])/2)-100
                        e2 = error

                        print(e1,e2, "error terms")
                        errory = round((c2[1]+c1[1])/2)-100


                        if round((c2[0]+c1[0])/2) > 110:
                            #move right
                            mouse.move(-0.01*error, 0)

                        if round((c2[0]+c1[0])/2) < 90:
                            mouse.move(-0.01*error, 0)
                            #left
                        #if round((c2[1]+c1[1])/2) > 110:
                            #down
                        #    mouse.move(0, -0.1*error)

                        #if round((c2[1]+c1[1])/2) < 90:
                            #up
                        #    mouse.move(0, -0.1*error)
                            #print("go to the left")

                        #cv2.rectangle(xyxy, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            #else:
                #mouse.move((25*math.sin(t1/3)),0)
                #print(math.sin(2*t1))
                #t1 = t1 +0.1
                    #if cls = person
            # Print time (inference + NMS)
        #    print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
            #    cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration



    #print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    #print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
