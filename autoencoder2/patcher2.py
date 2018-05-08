import numpy as np
from PIL import Image, ImageFilter
import h5py
import random as rng

from PIL import ExifTags
import scipy.misc
#import sobol_seq
import math

class Patcher():
    def __init__(self, _img_arr, _lbl_arr, _dim, _patches=None, _labels=None):

        # Here we put the image and labels in a padded array so predicting works
        # without cutting off data.

        # shape = _img_arr.shape

        if _lbl_arr is not None:
            assert _img_arr.shape[0] == _lbl_arr.shape[0]
            assert _img_arr.shape[1] == _lbl_arr.shape[1]

        # s0 = shape[0] + 2 * _dim[0]
        # s1 = shape[1] + 2 * _dim[1]

        self.img_arr = _img_arr

        # self.img_arr = np.zeros((s0, s1, _img_arr.shape[2]))
        # self.img_arr[_dim[0]:_dim[0]+shape[0], _dim[1]:_dim[1]+shape[1], :] = _img_arr
          
        if _lbl_arr is None:
            self.lbl_arr = None
        else:
            self.lbl_arr = _lbl_arr
            # self.lbl_arr = np.zeros((s0, s1))
            # self.lbl_arr[_dim[0]:_dim[0]+shape[0], _dim[1]:_dim[1]+shape[1]] = _lbl_arr

        self.img_shape = _img_arr.shape

        self.dim = _dim
        self.patches = _patches
        self.labels = _labels

    @classmethod
    def for_test(cls, _img_file, _dim=(256, 256)):
        img = Image.open(_img_file)

        for orientation in ExifTags.TAGS.keys(): 
            if ExifTags.TAGS[orientation]=='Orientation' : break 

        if img._getexif() != None:
            exif=dict(img._getexif().items())

            if exif[orientation] == 3: 
                img=img.rotate(180, expand=True)
            elif exif[orientation] == 6: 
                img=img.rotate(270, expand=True)
            elif exif[orientation] == 8: 
                img=img.rotate(90, expand=True)

        return cls(np.array(img), None, _dim)

    @classmethod
    def from_image(cls, settings, _img_file, _lbl_file, _dim=(256, 256)):
        rotate = settings['preprocess']['rotate']
        scale = settings['preprocess']['scale']
        blur = settings['preprocess']['blur']
        small_patch = settings['preprocess']['small_patch']

        img = Image.open(_img_file)

        # Here we look through the image meta-data to see if it 
        # needs to be rotated. If so, do that here. 
        # 
        # Note: we assume the label image does not need rotating.

        for orientation in ExifTags.TAGS.keys(): 
            if ExifTags.TAGS[orientation]=='Orientation' : break 

        if img._getexif() != None:
            exif=dict(img._getexif().items())

            if exif[orientation] == 3: 
                img=img.rotate(180, expand=True)
            elif exif[orientation] == 6: 
                img=img.rotate(270, expand=True)
            elif exif[orientation] == 8: 
                img=img.rotate(90, expand=True)

        img_size = (0,0)
        if scale > 0:
            direction = rng.random()
            if direction > 0.5:
                direction = -1
            else:
                direction = 1
            width, height = img.size
            amount = rng.random() * scale
            factor = max(0,1 + amount * direction)
            width = int(width * factor)
            height = int(height * factor)
            img_size = (width, height)
            img = img.resize(img_size)
        if small_patch:
            img = img.resize((256,256))

        if blur > 0:
            filter = ImageFilter.GaussianBlur(blur)
            img = img.filter(filter)

        img_arr = np.array(img)

        # Load the label if one was supplied. If the user intends
        # to predict an image, a label may not be passed in. 

        if _lbl_file is None:
            lbl_arr = None
        else:
            lbl = Image.open(_lbl_file)
            if scale > 0:
                lbl = lbl.resize(img_size)
            if small_patch:
                lbl = lbl.resize((256,256))
            lbl_arr = np.array(lbl)/255.0

            # For now, we always assume a binary label image

            if len(lbl_arr.shape) == 3:
                lbl_arr = lbl_arr[:,:,0]

            assert img_arr.shape[0] == lbl_arr.shape[0]
            assert img_arr.shape[1] == lbl_arr.shape[1]

        if (rotate == True):
            num_rot = rng.randint(0,3)
            img_arr = np.rot90(img_arr, num_rot)
            lbl_arr = np.rot90(lbl_arr, num_rot)

        return cls(img_arr, lbl_arr, _dim)

    def create_patch(self, pos, label=False):
        # rotate = settings['preprocess']['rotate']

        d0 = self.dim[0]
        d1 = self.dim[1]

        left = pos[0]
        right = pos[0] + d0

        top = pos[1]
        bottom = pos[1] + d1

        if label:
            patch = self.lbl_arr[left:right, top:bottom]
            patch = patch.reshape((d0, d1, 1))
        else:
            patch = self.img_arr[left:right, top:bottom]

        # if (rotate == True):
        #     num_rot = rng.randint(0,3)
        #     img_arr = np.rot90(img_arr, num_rot)
        #     lbl_arr = np.rot90(lbl_arr, num_rot)

        assert patch.shape[0] == d0
        assert patch.shape[1] == d1

        return patch

    def patchify(self, settings):
        patches = []
        labels = []

        max_patches = settings['train']['batch_size']
        method = settings['preprocess']['method']
        patch_size = settings['patch_size']

        shape = self.img_shape

        if method == "rng":
            while len(patches) < max_patches:

                i0 = rng.randint(0, shape[0]-256)
                i1 = rng.randint(0, shape[1]-256)

                label_patch = self.create_patch([i0, i1], label=True)

                # TODO: This helps reject patches that are all background
                #       to help with data bias. Perhaps a more sophisticated
                #       approach and some settings could be added here.

                if np.sum(label_patch.flatten()) > 0 or rng.randint(0,100) < 25:
                    patches.append(self.create_patch([i0, i1], label=False))
                    labels.append(label_patch)

        elif method == "sobol":
            sobol = sobol_seq.i4_sobol_generate(2, max_patches)
            nums = np.arange(0, max_patches)
            rng.shuffle(nums)
            while len(patches) < max_patches:
                i0 = sobol[nums[len(patches)],0]*(shape[0] - 256)
                i0 = math.floor(i0 / 2 + i0 / 4)
                i1 = sobol[nums[len(patches)],1]*(shape[1] - 256)
                i1 = math.floor(5 * i1 / 8 + i1 * 1 / 8)
                label_patch = self.create_patch([i0,i1], label=True)
                if np.sum(label_patch.flatten()) > 0 or rng.randint(0,100) < 25:
                    patches.append(self.create_patch([i0, i1], label=False))
                    labels.append(label_patch)

        #Creates too many patches, have to find solution
        elif method == "overlap":
        #     for i0 in range(0, shape[0] - 256, 256): 
        #         for i1 in range(0, shape[1] - 256, 256):
        #             label_patch = self.create_patch([i0, i1], label=True)
        #             if np.sum(label_patch.flatten()) > 0  or rng.randint(0,100) < 25:
        #                 patches.append(self.create_patch([i0, i1], label=False))
        #                 labels.append(label_patch)
            stride = int(patch_size / 2)
            for i0 in range(0, int(shape[0] - patch_size), stride):
                for i1 in range(0, int(shape[1] - patch_size), stride):
                    label_patch = self.create_patch([i0,i1], label=True)
                    if np.sum(label_patch.flatten()) > 0 or rng.randint(0,100) < 25:
                        patches.append(self.create_patch([i0,i1], label=False))
                        labels.append(label_patch)

        return patches, labels

    def predict(self, predictor, div=1):
        label_shape = (self.img_arr.shape[0], self.img_arr.shape[1])
        pred_label = np.zeros(label_shape)
        shape = pred_label.shape

        # Patch dimensions
        d0 = self.dim[0]
        d1 = self.dim[1]

        # Create a padded image to 
        # shape = self.lbl_arr.shape
        # shape[0] = int((shape[0] + d0 - 1) / d0) * d0
        # shape[1] = int((shape[1] + d1 - 1) / d1) * d1

        d0_stride = int(d0 / div)
        d1_stride = int(d1 / div)

        patches = []

        # TODO: This cuts off any part of the image not aligned with d0, d1, boundarys.
        #       For small enough patch dimensions this isn't a huge deal, but still would
        #       be a good idea to create a smarter algorithm here.

        for i0 in range(0, shape[0] - d0 - 1, d0_stride): 
            for i1 in range(0, shape[1] - d1 - 1, d1_stride):
                patches.append(self.create_patch([i0, i1], label=False))

        patches = np.array(patches)
        preds = predictor(patches)

        i = 0
        for i0 in range(0, shape[0] - d0 - 1, d0_stride): 
            for i1 in range(0, shape[1] - d1 - 1, d1_stride):
                pred_label[i0:i0+d0, i1:i1+d1] += preds[i].reshape((d0, d1))
                i = i + 1

        return pred_label
