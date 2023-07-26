import os
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        # init variables
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.epoch = -1
        self.current_iter = 0
        self.end_epoch = False

        # read image names and  labels
        self.image_names = np.array(os.listdir(self.file_path))
        self.labels = self.read_json(self.label_path)
        assert len(self.image_names) == len(self.labels.keys())

        self.total_instances = len(self.labels)
        # in case last iteration does not contain full batch size
        self.total_iterations = int(np.ceil(self.total_instances / self.batch_size))
        self.shuffle_images()

    def shuffle_images(self):
        if (self.shuffle):
            np.random.shuffle(self.image_names)

    def read_json(self, path):
        with open(path, "r") as ff:
            labels = json.load(ff)
        return labels
    
    def get_batch_indices(self):
        # returns current batch indices
        start = self.current_iter * self.batch_size
        stop = start + self.batch_size
        indices = np.arange(start, stop)
        return indices
    
    def read_batch_data(self, indices):
        # reads images and labels of given indices, also performs augmentation
        images = []
        labels = []
        for idx in indices:
            image_path = os.path.join(self.file_path, self.image_names[idx])
            img = np.load(image_path)
            img = resize(img, self.image_size)
            label = self.labels[self.image_names[idx].split(".")[0]]
            img = self.augment(img)
            images.append(img)
            labels.append(label)

        return np.array(images), np.array(labels)
    
    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        if (self.current_iter == 0):
            self.shuffle_images()
            self.epoch += 1

        indices = self.get_batch_indices()
        self.current_iter += 1
        if (self.current_iter == self.total_iterations):
            # indices greater than the last will start from 0
            indices = indices % self.total_instances
            self.current_iter = 0

        images, labels = self.read_batch_data(indices)
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if (self.rotation):
            rotation_coeff = np.random.randint(0,4)
            img = np.rot90(img, k=rotation_coeff)

        if (self.mirroring):
            flip = np.random.randint(0,2)
            if flip:
                ax = np.random.randint(0,2) # randomly decide from these axes
                img = np.flip(img, axis = ax)
        
        img = np.array(img)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #n_rows, n_cols = 4, self.batch_size

        #fig, rows = plt.subplots(n_rows, n_cols, figsize = self.image_size[:2])
        #for i in range(n_rows):
        #    images, labels = self.next()
        #    k = 0
        #    for img, label in zip(images, labels):
        #        rows[i][k].imshow(img)
        #        rows[i][k].axis("off")
        #        rows[i][k].set_title(self.class_name(label))
        #        k+=1
            
        #plt.show()

        (batch_images, batch_labels) = self.next()
        num_rows = int(np.ceil(self.batch_size/4))

        print(batch_images.shape)

        for i in range(self.batch_size):
            ax = plt.subplot(num_rows,4, i+1)
            im = batch_images[i]
            ax.imshow(im)
            ax.set_title(self.class_name(batch_labels[i]))
        
        plt.show()

    


if __name__=="__main__":

    file_path = "./exercise_data/"
    label_path = "./Labels.json"
    batch_size = 4
    image_size = (32,32,3)
    img_generator = ImageGenerator(file_path, label_path, batch_size, image_size, 
                                rotation=True, mirroring=True, shuffle=True)
    
    res = img_generator.next()
    img_generator.show()

    

