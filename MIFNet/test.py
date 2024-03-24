import argparse
import os
import time
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from PIL import Image
from eval import *
from model.MIFNetBlocks import *
from model.MIFNet import *
from utils import *
from torchvision import transforms


def predict_img(net,
                out_threshold=0.5,
                use_dense_crf=False,
                use_gpu=True,
                save_img=True):

    net.eval()

    dir_img = 'datasets/38-Cloud_test/'
    dir_mask = 'datasets/38-Cloud_test/test_nir/'

    """Returns a list of the ids in the directory"""
    id_image = get_ids(dir_img + 'test_nir/')

    test_img = get_test_img(id_image, dir_img, '.TIF')#返回图像三维数组,图像名称

    # val_dice = 0
    # oa, recall, precision, f1 = 0, 0, 0, 0
    for i, b in enumerate(test_img):
        img = np.array(b[0]).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)

        if use_gpu:
            img = img.cuda()

        with torch.no_grad():
            output_img = net(img)
            probs = output_img.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor()
                ]
            )

            probs = tf(probs.cpu())

            probs_np = probs.squeeze().cpu().numpy()
     

        result_img = mask_to_image(probs_np > out_threshold)
        if save_img:
            result_img.save(save_path + 'gt' + b[1][3:] + '.TIF')
        
       

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='./checkpoints/MIFNet.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', default='image1.TIF',
                        help='filenames of input images')
    parser.add_argument('--output', '-o', default='output.jpg', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--gpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--save_img', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=True)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    # 从数组array转成Image
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args() #得到输入的选项设置的值
    in_files = args.input
    out_files = get_output_filenames(args)
    save_path = 'result/'

    # net = torch.load(args.model)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    net = MIFNet(weights = None, cfg='./model/cfg/fasternet_s.yaml')
    net = nn.DataParallel(net.cuda(), device_ids=[0])

    print("Loading model {}".format(args.model))

    if args.gpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net = torch.load(args.model)
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    time1 = time.time()
    predict_img(net=net,
                out_threshold=args.mask_threshold,
                use_dense_crf=args.no_crf,
                save_img=args.save_img)
    print(time.time() - time1)

