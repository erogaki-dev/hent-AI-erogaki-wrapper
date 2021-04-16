"""
wrapper_detector.py - wrapper based on hent-AI/detector.py
Copyright (C) 2020 Nathan Cueto, 2021 Yurifag and undefinedprojects

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import json
import numpy as np
import skimage.draw
from skimage.filters import unsharp_mask
# import imgaug # should augment this improt as well haha
import time
# Root directory of project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN and ESRGAN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(os.path.join(os.path.abspath('.'), 'hent-AI/ColabESRGAN/'))
from mrcnn.config import Config
from mrcnn import model as modellib, utils
# It's too late to undo this now
from cv2 import imshow, waitKey, multiply, add, erode, VideoCapture, Canny, cvtColor,COLOR_GRAY2RGB, imdecode, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FPS, VideoWriter, VideoWriter_fourcc, resize, INTER_LANCZOS4, INTER_AREA, GaussianBlur, filter2D, bilateralFilter, blur
import ColabESRGAN.test
# Adatptive mosaic granularity
from green_mask_project_mosaic_resolution import get_mosaic_res

from PIL import Image

from NoCensoredRegionsFoundError import NoCensoredRegionsFoundError

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights
WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights.h5")

# taking this from hentai to avoid import
class HentaiConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "hentai"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1

    # Number of training steps per epoch, equal to dataset train size
    STEPS_PER_EPOCH = 1490

    # Skip detections with < 75% confidence
    DETECTION_MIN_CONFIDENCE = 0.75

# Detector class. Handles detection and potentially esr decensoring. For now, will house an ESR instance at startup
class Detector():
    # at startup, dont create model yet
    def __init__(self, weights_path):
        class InferenceConfig(HentaiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        self.config = InferenceConfig()

        self.weights_path = weights_path

        try:
            self.out_path = os.path.join(os.path.abspath('.'), "ESR_temp/ESR_out/")
            self.out_path2 = os.path.join(os.path.abspath('.'), "ESR_temp/ESR_out2/")
            self.temp_path = os.path.join(os.path.abspath('.'), "ESR_temp/temp/")
            self.temp_path2 = os.path.join(os.path.abspath('.'), "ESR_temp/temp2/")
            self.fin_path = os.path.join(os.path.abspath('.'), "ESR_output/")
        except:
            print("ERROR in Detector init: Cannot find ESR_out or some dir within.")
            return
        # Create esrgan instance for detector instance
        try:
            self.esr_model_path = os.path.join(os.path.abspath('.'), "4x_FatalPixels_340000_G.pth")
        except:
            print("ERROR in Detector init: ESRGAN model not found, make sure you have 4x_FatalPixels_340000_G.pth in this directory")
            return
        # Scan for cuda compatible GPU for ESRGAN. Mask-RCNN *should* automatically use a GPU if available.
        self.hardware = 'cpu'

    # Clean out temp working images from all directories in ESR_temp. Code from https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
    def clean_work_dirs(self):
        print("Cleaning work dirs...")
        folders = [self.out_path, self.out_path2, self.temp_path, self.temp_path2]
        for folder in folders:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('ERROR in clean_work_dirs: Failed to delete %s. Reason: %s' % (file_path, e))

    # Make sure this is called before using model weights
    def load_weights(self):
        print('Creating model, Loading weights...', end='  ')
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                        model_dir=DEFAULT_LOGS_DIR)
        try:
            self.model.load_weights(self.weights_path, by_name=True)
            print("Weights loaded")
        except Exception as e:
            print("ERROR in load_weights: Model Load. Ensure you have your weights.h5 file!", end=' ')
            print(e)

    """Apply cover over image. Based off of Mask-RCNN Balloon color splash function
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result covered image.
    """
    def apply_cover(self, image, mask, dilation):
        # Copy color pixels from the original color image where mask is set
        green = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
        green[:,:] = [0, 255, 0]

        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) < 1)
            # dilate mask to ensure proper coverage
            mimg = mask.astype('uint8')*255
            kernel = np.ones((dilation,dilation), np.uint8)
            mimg = erode(src=mask.astype('uint8'), kernel=kernel, iterations=1) #
            # dilation returns image with channels stripped (?!?). Reconstruct image channels
            mask_img = np.zeros([mask.shape[0], mask.shape[1],3]).astype('bool')
            mask_img[:,:,0] = mimg.astype('bool')
            mask_img[:,:,1] = mimg.astype('bool')
            mask_img[:,:,2] = mimg.astype('bool')

            cover = np.where(mask_img.astype('bool'), image, green).astype(np.uint8)
        else:
            # error case, return image
            cover = image
        return cover, mask

    # Similar to above function, except it places the decensored image over the original image.
    def splice(self, image, mask, gan_out):
        if mask.shape[-1] > 0:
            mask = (np.sum(mask, -1, keepdims=True) < 1)
            mask = 1 - mask # invert mask for blending
            mask = mask.astype('uint8')*255
            mask = GaussianBlur(mask, (29,29), 0)
            # mask_img = np.zeros([mask.shape[0], mask.shape[1],3]).astype('uint8')
            # for i in range(3):
            #     mask_img[:,:,i] = mask
            mask_img = mask.astype(float) / 255
            # proper blending courtesy of https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
            fg_o = gan_out.astype(float)
            bg_o = image.astype(float)
            fg = np.zeros([mask.shape[0], mask.shape[1],3]).astype(float)
            bg = np.zeros([mask.shape[0], mask.shape[1],3]).astype(float) # create foreground and background images with proper rgb channels
            cover = image
            for i in range(3):
                # Multiply the fg with the mask matte
                fg[:,:,i] = multiply(mask_img, fg_o[:,:,i])
                # Multiply the bg with ( 1 - mask_img )
                bg[:,:,i] = multiply(1.0 - mask_img, bg_o[:,:,i])
                # Add the masked fg and bg.
                cover[:,:,i] = add(fg[:,:,i], bg[:,:,i])
        else:
            #error case, return image
            cover=image
        return cover

    # function to handle all of the esrgan stuff
    def resize_GAN(self, img_path, img_name, is_video=False):
        # Attempt to obtain image
        try:
            image = skimage.io.imread(img_path) # problems with strange shapes
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image) # convert to rgb if greyscale
            if image.shape[-1] == 4:
                image = image[..., :3] # strip alpha channel
        except Exception as e:
            print("ERROR in resize_GAN: Image read. Skipping. image_path=", img_path)
            print(e)
            return
        # Calculate mosaic granularity.
        granularity = get_mosaic_res(np.array(image))
        if granularity < 10: #TODO: implement adaptive granularity by weighted changes
            print("Granularity of image was less than threshold at ", granularity)
            granularity = 10
        # Resize image down
        try:
            mini_img = resize(image, (int(image.shape[1]/granularity), int(image.shape[0]/granularity)), interpolation=INTER_AREA) # TODO: experiment with interpolations
            # After resize, run bilateral filter to keep colors coherent
            file_name = self.temp_path + img_name[:-4] + '.png'
            skimage.io.imsave(file_name, mini_img)
        except Exception as e:
            print("ERROR in resize_GAN: resize. Skipping. image_path=",img_path, e)
            return
        # Now run ESRGAN inference
        gan_img_path = self.out_path + img_name[:-4] + '.png'
        self.esrgan_instance.run_esrgan(test_img_folder=file_name, out_filename=gan_img_path, mosaic_res=granularity)

    # Runs hent-AI detection and splice. Mosaic only.
    def ESRGAN(self, img_path, img_name, is_video=False):
        try:
            image = skimage.io.imread(img_path) # problems with strange shapes
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image) # convert to rgb if greyscale
            if image.shape[-1] == 4:
                image = image[..., :3] # strip alpha channel
        except Exception as e:
            print("ERROR in detector.ESRGAN: Image read. Skipping. image_path=", img_path)
            print(e)
            return
        # Run detection first
        r = self.model.detect([image], verbose=0)[0]
            # Remove bars from detection; class 1

        if len(r["scores"]) == 0:
            print("Skipping image with no detection")
            return
        remove_indices = np.where(r['class_ids'] != 2)
        new_masks = np.delete(r['masks'], remove_indices, axis=2)

        # load image from esrgan
        gan_img_path = self.out_path + img_name[:-4] + '.png'
        gan_image = skimage.io.imread(gan_img_path)
        gan_image = resize(gan_image, (image.shape[1], image.shape[0]))
        # Splice newly enhanced mosaic area over original image
        fin_img = self.splice(image, new_masks, gan_image)
        try:
            # Save output, now force save as png
            file_name = self.fin_path + img_name[:-4] + '.png'
            skimage.io.imsave(file_name, fin_img)
        except Exception as e:
            print("ERROR in ESRGAN: Image write. Skipping. image_path=", img_path, e)

    # ESRGAN folder running function
    def run_ESRGAN(self, in_path = None, is_video = False, force_jpg = True):
        assert in_path

        # Parse directory for files.
        img_list = []
        for file in os.listdir(in_path):
            try:
                if file.endswith('.png') or file.endswith('.PNG') or file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".mp4") or file.endswith(".avi"):
                    img_list.append((in_path + '/' + file, file))
            except Exception as e:
                print("ERROR in run_ESRGAN: File parsing. file=", file, e)
        # begin ESRGAN on every image. Create esrgan instance too.
        star = time.perf_counter()
        self.esrgan_instance = ColabESRGAN.test.esrgan(model_path=self.esr_model_path, hw=self.hardware)
        for img_path, img_name in img_list:
            self.resize_GAN(img_path=img_path, img_name=img_name, is_video=is_video)
        # destroy esrgan model. Create hent-AI model.
        # self.esrgan_instance = []
        del self.esrgan_instance
        for img_path, img_name in img_list:
            self.ESRGAN(img_path=img_path, img_name=img_name, is_video=is_video)
        fin = time.perf_counter()
        total_time = fin-star
        print("Completed ESRGAN detection and decensor in {:.4f} seconds".format(total_time))
        self.clean_work_dirs() #NOTE: DISABLE ME if you want to keep the images in the working dirs
        #TODO: maybe unload hent-AI tf model here

    def detect_and_cover(self, censored_img, dilation=0):
        # Run on Image
        image = np.array(censored_img) # skimage.io.imread(image_path) # problems with strange shapes
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image) # convert to rgb if greyscale
        if image.shape[-1] == 4:
            image = image[..., :3] # strip alpha channel

        # Detect objects
        try:
            r = self.model.detect([image], verbose=0)[0]
        except Exception as e:
            print("ERROR in detect_and_cover: Model detection.",e)
            return
        # Remove unwanted class, code from https://github.com/matterport/Mask_RCNN/issues/1666
        remove_indices = np.where(r['class_ids'] != 1) # remove mosaic: class 1
        new_masks = np.delete(r['masks'], remove_indices, axis=2)

        cov, mask = self.apply_cover(image, new_masks, dilation)

        if mask.size == 0:
            raise NoCensoredRegionsFoundError("No censored regions detected.")

        return Image.fromarray(cov.astype("uint8"))

    # Function for file parsing, calls the aboven detect_and_cover
    def run_on_folder(self, input_folder, output_folder, is_video=False, orig_video_folder=None, is_mosaic=False, dilation=0):
        assert input_folder
        assert output_folder # replace with catches and popups

        self.esrgan_instance = [] # rare case where esrgan instance not destroyed but new action started, catch it here
        if dilation < 0:
            print("ERROR: dilation value < 0")
            return
        print("Will expand each mask by {} pixels".format(dilation/2))

        file_counter = 0
        # obtain inputs from the input folder
        img_list = []
        for file in os.listdir(str(input_folder)):
            file_s = str(file)
            try:
                if file_s.endswith('.png') or file_s.endswith('.PNG') or file_s.endswith(".jpg") or file_s.endswith(".JPG"):
                    img_list.append((input_folder + '/' + file_s, file_s))
            except:
                print("ERROR in run_on_folder: File parsing. file=", file_s)

        # save run detection with outputs to output folder
        for img_path, img_name in img_list:
            star = time.perf_counter()
            self.detect_and_cover(img_path, img_name, output_folder, is_mosaic=is_mosaic, dilation=dilation)  #sending force_jpg for debugging
            fin = time.perf_counter()
            total_time = fin-star
            print('Detection on image', file_counter,'of', len(img_list),'finished in {:.4f} seconds'.format(total_time))
            file_counter += 1
