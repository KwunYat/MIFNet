import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix)[1] / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Pixel_Precision_Class(self):
        precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix,axis=0)
        precision = np.nanmean(precision)
        return precision

    def Pixel_Recall_Class(self):
        recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix,axis=1)
        recall = np.nanmean(recall)
        return recall
    
    def Pixel_JaccardIndex_Class(self):
        jaccard_index = np.diag(self.confusion_matrix)[1] / \
                        (np.sum(self.confusion_matrix,axis=0)[1] + self.confusion_matrix[1,0])
        return jaccard_index
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """
        没看懂哦！！！
        :param gt_image:
        :param pre_image:
        :return:
        """
        mask = (gt_image >= 0) & (gt_image < self.num_class) #消除边界
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

if __name__ == '__main__':
    Pre_Dir = './results/'
    # Gt_Dir = '/extend/lixian/lixian/dataset/Test/Entire_scene_gts'
    Gt_Dir = './datasets/38-Cloud_test/Entire_scene_gts'
    evaluator = Evaluator(2)
    evaluator.reset()
    #/home/ices/lixian/dataset/Test/Entire_scene_gts/edited_corrected_gts_LC08_L1TP_003052_20160120_20170405_01_T1.TIF
    for filename in tqdm(os.listdir(Pre_Dir)):
        for file in os.listdir(Gt_Dir):
            judge = '_'+filename.split('.')[0]+'_'
            if judge in file:
                gt_filename = file
                break
        pre_file_path = os.path.join(Pre_Dir,filename)
        gt_file_path = os.path.join(Gt_Dir,file)
        assert os.path.isfile(pre_file_path)
        assert os.path.isfile(gt_file_path)
        pred = Image.open(pre_file_path).convert('1')
        target = Image.open(gt_file_path)
        #pred.save('real.png','PNG')
        w = np.ceil((pred.size[0] - target.size[0])/2)
        h = np.ceil((pred.size[1] - target.size[1])/2)
        pred = pred.crop((w, h, w+target.size[0], h+target.size[1]))
        #pred = pred.resize(target.size, Image.ANTIALIAS)
        #pred.save('pred.png','PNG')
        #target.save('target.png',"PNG")
        pred = np.asarray(pred).astype(int)
        # pred[pred>0] = 1
        target = np.asarray(target).astype(int)

        #assert 1==2
        evaluator.add_batch(target, pred)
        
        
    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    precision = evaluator.Pixel_Precision_Class()
    recall = evaluator.Pixel_Recall_Class()
    jaccard_index = evaluator.Pixel_JaccardIndex_Class()
    confusion_matrix = evaluator._generate_matrix(target, pred)
    print(confusion_matrix)
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {},precision:{},recall:{},jaccard_index:{}"
          .format(Acc, Acc_class, mIoU, FWIoU,precision,recall,jaccard_index))