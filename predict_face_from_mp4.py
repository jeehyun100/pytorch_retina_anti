from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import config
import glob

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--crop', default=True, help='Use cpu inference')
parser.add_argument('--one_face', default=True, help='Use cpu inference')
parser.add_argument('--debug', default=False, help='Use cpu inference')






args = parser.parse_args()

def get_datasets_rose(root):
    return glob.glob(root)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def rotate_face(img_raw, b, between_eyes_gap):
    # red y b[6]

    # cv2.imshow('RetinaFace-Pytorch', img_raw)
    # cv2.waitKey()

    if (b[6] > b[8]) and (abs(b[6] - b[8]) > between_eyes_gap):
        img_raw = cv2.rotate(img_raw, cv2.ROTATE_90_CLOCKWISE)
    elif (abs(b[6] - b[8]) < between_eyes_gap and b[10] < b[6]):
        img_raw = cv2.rotate(img_raw, cv2.ROTATE_180)
    elif (b[6] < b[8]) and (abs(b[6] - b[8]) > between_eyes_gap):
        img_raw = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_raw

def show_bbox_lm(img_raw, b):
    if args.debug:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)  # red
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)  # yellow
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)  #
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    return img_raw

def crop_bbox(img_raw, b, ratio=1.0):
    if args.crop:
        #height_resize = ((int(b[3]) - int(b[1])) * scale) / (int(b[3]) - int(b[1]))
        h = int(b[1]) / ratio
        h_delta = int(b[3]) * ratio

        w = int(b[0]) / ratio
        w_delta = int(b[2]) * ratio

        print("h, h_delta, w, w_delta : {0}/({1}), {2}/({3}), {4}/({5}), {6}/({7})".format(h, b[1], h_delta, b[3], w, b[0], w_delta, b[2]))


        img_raw = img_raw[int(h):int(h_delta), int(w):int(w_delta)]
        # cv2.imshow("cropped", crop_img)
        # text = "{:.4f}".format(b[4])
        # b = list(map(int, b))
        # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        #
        #
        # cx = b[0]
        # cy = b[1] + 12
        # cv2.putText(img_raw, text, (cx, cy),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        #
        # # landms
        # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)  # red
        # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)  # yellow
        # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)  #
        # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
    return img_raw

def get_face(img_raw, type_name, output_file_name,count, sec ):
    #img_raw = cv2.imread("/Users/yewoo/dev/Pytorch_Retinaface/datasets/output/negative/Vm_NT_HW_wg_E_20_172_6.jpg", cv2.IMREAD_COLOR)

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    # scale = torch.Tensor([img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    between_eyes_gap = img_raw.shape[0] * 0.1

    if args.one_face:
        dets = dets[:1]

    # show image
    if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            img_raw = show_bbox_lm(img_raw, b)
            img_raw = crop_bbox(img_raw, b, 1.2)
            # text = "{:.4f}".format(b[4])
            # b = list(map(int, b))
            # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            # cx = b[0]
            # cy = b[1] + 12
            # cv2.putText(img_raw, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            #
            # # landms
            # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4) #red
            # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4) # yellow
            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4) #
            # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            # red y b[6]
            # between_eyes_gap = img_raw.shape[0]*0.1
            img_raw = rotate_face(img_raw, b, between_eyes_gap)

            try:
                cv2.imwrite(config.output_dir + type_name + "/" + output_file_name + "_" + str(count) + ".jpg",
                            img_raw)  # save frame as JPG file
                print("Saving Neg file name : {0}[{1}]".format(output_file_name, sec))
            except Exception as e:
                print("Saving error")

            # if  (b[6] > b[8]) and (abs(b[6] - b[8]) > between_eyes_gap):
            #     img_raw = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # elif (abs(b[6] - b[8]) < between_eyes_gap):
            #     img_raw = cv2.rotate(img_raw, cv2.ROTATE_180)
            # elif (b[6] < b[8]) and (abs(b[6] - b[8]) > between_eyes_gap):
            #     img_raw = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # cv2.imshow('RetinaFace-Pytorch', img_raw)
            # cv2.waitKey()


def get_frame():
    file_list = get_datasets_rose(config.root_dir)

    def getFrame(vidcap, sec, count):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite("image" + str(count) + ".jpg", image)  # save frame as JPG file
        return hasFrames


    for video_file in file_list:
        count = 0;
        sec = 0
        cap = cv2.VideoCapture(video_file)

        while True:
            ret, frame = cap.read()
            #cv2.imshow("Video", frame)
            if ret:
                #output_file_name = ""
                output_file = os.path.basename(video_file)
                first_segment = output_file.split("_")[0]
                output_file_name = os.path.splitext(output_file)[0]
                if first_segment == 'G':
                    type_name = "positive"
                    get_face(frame, type_name, output_file_name,count, sec )
                    #cv2.imwrite(config.output_dir+"positive/"+ output_file_name +"_"+ str(count) + ".jpg", frame)  # save frame as JPG file
                    #print("Saving Pos file name : {0}[{1}]".format(output_file_name, sec))

                    #crop_face_with_retinaface(frame, retinaface_model)
                else :
                    type_name = "negative"
                    get_face(frame,  type_name, output_file_name,count, sec )
                    #cv2.imwrite(config.output_dir+"negative/"+ output_file_name +"_"+ str(count) + ".jpg", frame)  # save frame as JPG file
                    #print("Saving Neg file name : {0}[{1}]".format(output_file_name, sec))

                    #crop_face_with_retinaface(frame, retinaface_model)

                sec = sec + config.frame_rate
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                count +=1
            else:
            #if cv2.waitKey(20) & 0xFF == ord('q'):
                break

if __name__ == '__main__':

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.output_dir+"positive", exist_ok=True)
    os.makedirs(config.output_dir+"negative", exist_ok=True)

    #retinaface_model = load_retina_model()


    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin
    #for i in range(100):
    #image_path = "./curve/test.jpg"
    #image_path = "/Users/yewoo/dev/RetinaFace_Pytorch/datasets/output/negative/Mc_NT_5s_g_E_20_51_29.jpg"
    #image_path = "/Users/yewoo/dev/RetinaFace_Pytorch/datasets/output/positive/G_NT_5s_wg_E_18_8_274.jpg"
    image_path = "/Users/yewoo/dev/RetinaFace_Pytorch/datasets/output/negative/Vl_NT_5s_g_E_20_140_98.jpg"

    get_frame()
    #get_face(image_path)
