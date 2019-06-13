import megbrain as mgb
import os,sys
from multiprocessing import Queue, Process
import argparse
import math
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy
import cv2
from config import config
from network import *
from megskull.network import NetworkVisitor
from pycocotools.coco import *
from pycocotools.mask import encode
import random
from meghair.train.interaction import parse_devices
from tqdm import tqdm
import megbrain
import megbrain as mgb
from meghair.utils.misc import ensure_dir
from multiprocessing import Pool
import time



sys.path.insert(0,'/unsullied/sharefs/zangyuhang/plate/text_detect/lib')
from meg_kernels.bilinear import bilinear


clr = [[32, 178, 170], [152, 251, 152], [0, 255, 127], [255, 236, 139],
       [255, 193, 37], [139, 101, 8], [238, 121, 66], [238, 216, 174],
       [139, 131, 120], [147, 112, 219], [240, 255, 240], [28, 134, 238]]
clr_nr = [0]

def get_color():
    idx = clr_nr[0] % 12
    clrs = numpy.zeros((1, 3)).astype(numpy.float)
    clrs[0] = clr[idx]
    clr_nr[0] += 1
    return clrs

def load_model(model_file, dev, use_input_rois=False):
    global __desc__
    network = make_network(1, 'test', model_file, use_input_rois)
    funcs = []

    vis = NetworkVisitor(network.outputs)
    vis = vis.reset_comp_node(dev)
    env = FpropEnv()
    env.flags.enforce_var_shape = False
    megbrain.config.set_comp_graph_option(env.comp_graph,
                                          'log_static_mem_alloc', False)
    func = env.comp_graph.compile_outonly(
        [env.get_mgbvar(var) for var in vis.dest_vars])
    return func



def solverange(img_id, func_first, func_second,img_name):
    max_size = 1200  # config.eval_image_max_size
    short_size = 720 # config.eval_image_short_size
    nms_thres = 0.3
    max_per_img_first=config.test_max_boxes_per_image

    img_name = img_name

    inp = cv2.imread(img_name, cv2.IMREAD_COLOR)

    max_per_img_first=config.max_boxes_of_image

    sz = inp.shape
    max_wh, min_wh = max(sz[0], sz[1]), min(sz[0], sz[1])
    sca = min(max_size / max_wh, short_size / min_wh)

    img = cv2.resize(inp, (0, 0), fx=sca, fy=sca)
    img = img.transpose([2, 0, 1])[numpy.newaxis, :]
    h, w = img.shape[2], img.shape[3]
    img = np.ascontiguousarray(img, dtype=np.float32)
    im_info = np.array([(h, w, sca, h / sca, w / sca, 100, h / sca, w / sca)],
                       dtype=np.float32)
    kwargs = {'data': img, 'im_info': im_info}
    res = func_first(**kwargs)
    from IPython import embed
    tot = res[1].shape[0]
    dtboxes = list()
    for c in range(2):
        if c <= 0:
            continue
        tmp = []
        for i in range(tot):
            score = float(res[1][i, c])
            if score<0.7:
                continue

            x0, y0, x1, y1 = map(float, res[0][i, c])
            tmp.append((x0, y0, x1, y1, score))

        if tmp == []:
            print("imgid{} is empty".format(img_id))
            return None,inp,img_id
        tmp.sort(key=lambda x: x[4], reverse=True)
        tmp = numpy.array(tmp, dtype=numpy.float32)
        from meg_kernels.lib_nms.gpu_nms import gpu_nms
        keep = gpu_nms(tmp, nms_thres)
        tmp = tmp[keep]
        cls = c
        tmp_c = map(lambda x: (x[0], x[1], x[2], x[3], x[4], cls), tmp)
        dtboxes.extend(tmp_c)
    dtboxes.sort(key=lambda x: x[4], reverse=True)
    if len(dtboxes) > max_per_img_first:
        dtboxes = dtboxes[:max_per_img_first]
    dtboxes_np = numpy.array(dtboxes, dtype=numpy.float)
    rois = numpy.zeros((dtboxes_np.shape[0], 5), dtype=numpy.float32)
    rois[:, 1:5] = dtboxes_np[:, 0:4] * sca
    rois = np.ascontiguousarray(rois, dtype=np.float32)
    res2 = func_second(data=img, rois=rois, im_info=im_info)
    fm_order = res2[2]
    available_inds = res2[3]
    idx = fm_order[available_inds]
    pred_mask = res2[0]
    # new_rois = res2[1]
    masks = numpy.zeros((pred_mask.shape[0], sz[0], sz[1]), dtype=numpy.uint8)
    all_info = []
    for i in range(idx.shape[0]):
        if idx[i] >= len(dtboxes):
            continue
        dt = dtboxes[idx[i]]
        sc = dt[4]
        c = int(dt[5] + .5)
        x0, y0, x1, y1 = dtboxes[idx[i]][:4]
        h, w = y1 - y0, x1 - x0
        x0 -= w / 56
        y0 -= h / 56
        x1 += w / 56
        y1 += h / 56
        msk = pred_mask[i, c]
        msk = 1. / (1 + np.exp(-msk))
        newmsk = np.zeros((msk.shape[0] + 2, msk.shape[1] + 2),
                          dtype=np.float32)
        newmsk[1:-1, 1:-1] = msk
        # newmsk = newmsk - 0.12246 #掉0.3
        msk = newmsk
        mask = bilinear.bilinear(msk, x0, y0, x1, y1, sz[0], sz[1])
        bbox = (x0, y0, x1, y1)
        all_info.append((sc, c, bbox, mask))

    return all_info,inp,img_id

def draw_rec(all_info,img,img_id,masks,draw_mask=True,output_dir='./output/images'):
    def MydrawMask(img, masks, lr=(None, None), alpha=0.4, clrs=get_color()):
        n, h, w = masks.shape[0], masks.shape[1], masks.shape[2]
        for i in range(max(0, lr[0]), min(n, lr[1])):
            M = masks[i].reshape(-1)
            B = numpy.zeros(h * w, dtype=numpy.int8)
            for y in range(h - 1):
                for x in range(w - 1):
                    k = y * w + x
                    if M[k] != M[k + 1]:
                        B[k], B[k + 1] = 1, 1
                    if M[k] != M[k + w]:
                        B[k], B[k + w] = 1, 1
                    if M[k] != M[k + 1 + w]:
                        B[k], B[k + 1 + w] = 1, 1
            M.shape = (h, w)
            B.shape = (h, w)
            for j in range(3):
                O, c, a = img[:, :, j], clrs[0][j], alpha
                am = a * M
                O = O - O * am + c * am
                img[:, :, j] = O * (1 - B) + c * B
        return img
    if all_info == None:
        cv2.imwrite(os.path.join(output_dir,"img{}.jpg".format(img_id)),img)
        return
    else:
        for k,i in enumerate(all_info):

            #画水平矩形
            x1,y1,x2,y2 = i[2]
            #过滤太小的框
            #if abs(x2-x1)<15 or abs(y2-y1)<15:
            #    continue

            #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #font_size = 1.2/2000*img.shape[1]
            #cv2.putText(img,str(i[0])[:4],(int(x1),int(y1)),font,font_size,(0,255,0),1)

            #画mask外接最小斜矩形
            mask = i[3]
            image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            try:
                contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
                pts = contours[0]
            except:
                continue
            #rect = cv2.minAreaRect(pts)
            #box = cv2.boxPoints(rect)
            #box=np.array(box,dtype=np.int32)
            #img = cv2.polylines(img,[box],True,(0,255,255),3)

            if len(masks)>0 and draw_mask==True:
                #img = MydrawMask(img, masks,(k,k+1))
                img = cv2.drawContours(img, [pts], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir,"img{}.jpg".format(img_id)),img)

def write_res(all_info,img,img_id):
    with open('/unsullied/sharefs/zangyuhang/plate/text_detect/model/admin/total_text/input_dir/img{}.txt'.format(img_id),'w+')  as f:
        if all_info == None:
            print(img_id,"empty")
            return
        else:
            for k,i in enumerate(all_info):
                x1,y1,x2,y2 = i[2]
                #if abs(x2-x1)<5 or abs(y2-y1)<5:
                #    continue

                mask = i[3]
                image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                try:
                    contours = sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
                    pts = contours[0]
                except:
                    continue
                #embed()
                #rect = cv2.minAreaRect(pts)
                #box = cv2.boxPoints(rect)
                #box=np.array(box,dtype=np.int32)
                #pts = cv2.convexHull(pts, returnPoints=True)
                #epsilon = 0.01*cv2.arcLength(pts, True)
                #pts = cv2.approxPolyDP(pts, epsilon, True)
                pts = pts.reshape(-1)
                #互换xy位置
                res = []
                for i in range(0,len(pts),2):
                    res.append(pts[i+1])
                    res.append(pts[i])
                #res = box.reshape(-1)
                res = [str(i) for i in res]
                if len(res)<6:
                    print('pic {} has bad point')
                    continue
                res = ','.join(res)

                f.write(res+'\n')

num_gpu = 8
def test(p_id,epoch,offset):
    path = os.path.abspath('.').split('/')[-1]

    model_path = '/unsullied/sharefs/zangyuhang/plate/text_detect/output/xieenze/mask_rcnn/' + \
        path + \
        '/model_dump/epoch-' + epoch + '.brainmodel'
    print(model_path)
    if not os.path.exists(model_path):
        print('model_path is empty!!!!!')
    dev = 'gpu{}'.format(p_id)
    mgb.config.set_default_device(dev)
    m1 = load_model(model_path, dev)
    m2 = load_model(model_path, dev,use_input_rois=True)
    for img_id in range(p_id+offset, 1880, num_gpu):
        img_name = '/unsullied/sharefs/zangyuhang/isilon-home/DataSet/TOTAL_TEXT/Test/img{}.jpg'.format(img_id)
        if not os.path.exists(img_name):
            #print(img_name,' does not exist!')
            continue

        print('[{}] img_id={}'.format(p_id, img_id))
        all_info, img, img_id = solverange(img_id, m1, m2, img_name)
        write_res(all_info, img, img_id)
        masks = []
        if not all_info == None:
            for i in all_info:
                mask = i[3]
                masks.append(mask)
            masks = np.array(masks)
        draw_rec(all_info, img, img_id, masks, draw_mask=True)
    return



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch", default=0)
    args = parser.parse_args()
    epoch = args.epoch

    ensure_dir('output/out_res')
    ensure_dir('output/out_res9000')
    ensure_dir('output/images')

    pool = Pool(processes=num_gpu)
    start = time.time()
    offset = 0
    #test(0,epoch,offset)
    for p_id in range(num_gpu):
        pool.apply_async(test, (p_id,epoch,offset, ))
    pool.close()
    pool.join()
    end = time.time()
    print('time used:',end-start)


if __name__ == "__main__":
    main()

