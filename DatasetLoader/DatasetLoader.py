import tensorflow as tf
from scipy.ndimage import imread
import os
import matplotlib.pyplot as plt
import numpy as np
DATASET_DIR = './Dataset'

class datasetLoader:

    def __init__(self, dataset_dir, batch_size, image_size,check_readability = True, max_images_in_ram = None, shuffled = True):
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.image_size = image_size

        self.load_itr = 0
        self.batch_itr = 0
        self.internal_batch_itr = 0
        self.inputs_file, self.targets, self.target_2_idx,self.idx_2_target = self.get_filename(check_readability)
        self.data_size = len(self.targets)
        print('Data Size:', self.data_size)
        self.total_batch = self.data_size//self.batch_size
        if max_images_in_ram is None:
            self.miir = self.data_size
        else:
            self.miir = max_images_in_ram
            if self.miir < self.batch_size or self.miir % self.batch_size:
                raise ValueError("max_images_in_ram parameter should be multiple of batch_size and equal or less than data_size ")
        self.images_in_ram = np.zeros([self.miir, self.image_size[0], self.image_size[1], self.image_size[2]], dtype= np.float32)
        self.targets_in_ram = np.zeros([self.miir])
        self.data_idx = np.arange(0,int(self.data_size))
        if shuffled is True:
            np.random.shuffle(self.data_idx)
        self.load_data()

    def get_filename(self,check):
        if not os.path.exists(self.dataset_dir):
            raise NameError(self.dataset_dir, "doesn't exist.")
        inputs_file = []
        targets = []
        target_2_idx = {}
        idx_2_target = {}
        ErrorTag = 0
        print('Targets are:')
        for itr, class_name in enumerate(sorted(os.listdir(self.dataset_dir))):
            idx_2_target[itr] = class_name
            target_2_idx[class_name] = itr
            images_name = sorted(os.listdir(os.path.join(self.dataset_dir, class_name)))

            if check is True:
                temp = np.zeros([self.image_size[0], self.image_size[1], self.image_size[2]])
                for idx, image in enumerate(images_name):
                    data_dir = os.path.join(os.path.join(self.dataset_dir, class_name), image)
                    try:
                        temp = imread(data_dir).astype(float)
                    except (ValueError, IOError, OSError) as e:
                        ErrorTag = 1
                        print(e,'(It is recommended to delete this file and run again)')
                        del images_name[idx]


            inputs_file.extend(images_name)
            print(class_name)
            targets.extend(itr*np.ones(len(images_name)))
        if check == 1 and ErrorTag == 0:
            print('All Files are verified, No need to check readability in next run.')
        return inputs_file, targets, target_2_idx, idx_2_target

    def load_data(self):
        itr = 0
        load_itr = 0
        while itr < self.miir:
            self.load_itr = self.load_itr % self.data_size
            load_itr = self.data_idx[self.load_itr]
            data_dir = os.path.join(
               os.path.join(self.dataset_dir, self.idx_2_target[self.targets[load_itr]]),
               self.inputs_file[load_itr])

            self.images_in_ram[itr] = imread(data_dir,mode = 'RGB').astype(float)
            self.targets_in_ram[itr] = self.targets[load_itr]
            self.load_itr += 1
            itr += 1

    def next_batch(self ):
        if self.batch_itr > self.total_batch:
            self.batch_itr = 0
            self.load_itr = 0
            self.internal_batch_itr = 0
            if self.miir<self.data_size:
                self.load_data()
        if self.miir<self.data_size and self.internal_batch_itr >= (self.miir//self.batch_size):
            self.load_data()
            self.internal_batch_itr = self.internal_batch_itr % (self.miir // self.batch_size)
        input_batch = self.images_in_ram[self.internal_batch_itr*self.batch_size:(self.internal_batch_itr+1)*
                                                                                 self.batch_size]
        target_batch = self.targets_in_ram[self.internal_batch_itr*self.batch_size:(self.internal_batch_itr+1)*
                                                                                   self.batch_size]
        self.internal_batch_itr += 1
        self.batch_itr += 1
        return input_batch, target_batch

def main():
    test = datasetLoader(
        dataset_dir = DATASET_DIR,
        batch_size = 3,
        image_size = [28, 28, 3])
    print('Hint: Some random images are plotted from each batch')
    for i in range(test.total_batch):
        img, target = test.next_batch()
        random_num = np.random.randint(0,test.batch_size)
        fig = plt.figure()
        plt.imshow(img[random_num])
        fig.suptitle('Target:' + test.idx_2_target[target[random_num]])
        plt.show()


if __name__ == '__main__':
    main()
