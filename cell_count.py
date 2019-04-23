from skimage import filters
from skimage.morphology import white_tophat, disk
from skimage.measure import regionprops,label
import os
import tifffile
import numpy
import glob
from scipy import signal
from scipy.ndimage.filters import median_filter
import os
import json
from pathos.multiprocessing import ProcessingPool
import datetime
import pytz
import time
import logging


channels = ['ch01','ch02']



def medFilt_runnable(img,median_filter=median_filter,sigma = 3):
    return  median_filter(img,size = (sigma,sigma))


def threshold_runnable(img,filters = filters):
    print('thresh')
    return filters.threshold_otsu(img)

def getBinary_runnable(img,threshold = threshold_runnable):
    img_threshold = threshold(img)
    return (img > img_threshold)

def medFiltStack_runnable(imgStack,medFilt_runnable = medFilt_runnable):
    newstack = numpy.zeros((imgStack.shape))
    for i in range(len(imgStack)):
        newstack[i] = medFilt_runnable(imgStack[i,:,:])
    return newstack

def readimg(image_path,tfimread = tifffile.imread):
        image = tfimread(image_path)
        return image

def getImageCoordinates_runnable(img,intensity_image = None,regionprops=regionprops):
    labeledimg = label(img,neighbors=8,connectivity = 5)
    if intensity_image is not None:
        props = regionprops(labeledimg,intensity_image=intensity_image)
    else:
        props = regionprops(labeledimg)
    return props

def labelBinaryImage_runnable(img,label = label):
    return label(img)


# binary_dict = {'name':'getBinary_runnable','runnable': getBinary_runnable}
# thresh_dict = {'name':'thresholds','runnable': threshold_runnable}
# medsingle_dict = {'name': 'median filter', 'runnable':medFilt_runnable}
# medstack_dict = {'name' : 'median filter stack','runnable':medFiltStack_runnable}
# backsub_dict = {'name': 'Background Stack','runnable':backgroundSubtraction_runnable}
# maxPro_dict = {'name': 'MIP','runnable':getMaxPro_runnable}


class CellData(object):
    def __init__(self, directory,setupPool = True):
        self.directory = directory

        self.stackfilenames = [os.path.join(directory,x) for x in os.listdir(directory) if 'tif' in x]
        self.basedir = os.path.dirname(directory)
        self.setupPool = setupPool
        if self.setupPool:
            self.initializeProcessingPool()

        else:
            self.unloadPool()


        self.labeled_properties = None

        self.channels = ['ch01','ch02']
        
        self.stack_channel_files = None
        
        self.stack_channel_images = None
        
        self.processed_stack_images = None

        self.initializeProperties()
        
        # for ch in self.channels:
        #     self.stack_channel_files[ch] = sorted([imgfile for imgfile in self.stackfilenames if ch in imgfile])
            # print('channel filenames: ',self.stack_channel_files[ch])


    def initializeProperties(self):
        if not self.labeled_properties:
            self.labeled_properties = {}

        if not self.stack_channel_files:
            self.stack_channel_files = {}
            for ch in self.channels:
                self.stack_channel_files[ch] = sorted([imgfile for imgfile in self.stackfilenames if ch in imgfile])

        if not self.processed_stack_images:
            self.processed_stack_images = {'ch02': {},'ch01':{}}



    def initializeProcessingPool(self,ncpus = 4):
        self.pool = ProcessingPool(ncpus=ncpus)
        

    def unloadImages(self):
        self.stack_channel_images = None
        self.processed_stack_images = None
        self.stack_channel_files = None
        self.basedir = None
        self.setupPool = False
        self.directory = None
        self.stackfilenames = None

    def unloadPool(self):
        self.pool = None


            
    def loadImages(self):
        #loads images in parallel
        to_unload = False
        if self.pool is None:
            self.initializeProcessingPool()
            to_unload = True
        
        self.stack_channel_images = {}
        
        
        for ch in self.channels:
            stack_imglist = self.stack_channel_files[ch]
            # print(stack_imglist)
            self.stack_channel_images[ch] = numpy.asarray(self.pool.map(readimg,stack_imglist),dtype = numpy.uint8)
            # print(self.stack_channel_images[ch].shape)
            if len(self.stack_channel_images[ch].shape) == 3:
                z,y,x = self.stack_channel_images[ch].shape
                if numpy.abs(x-y) != 0:
                    self.stack_channel_images[ch] = self.padImagestack_runnable(self.stack_channel_images[ch])
            else:
                y,x = self.stack_channel_images[ch].shape
                if numpy.abs(x-y) != 0:
                    self.stack_channel_images[ch] = self.padSingleImage_runnable(self.stack_channel_images[ch])
                
        if to_unload:
            self.unloadPool

        return

    def saveImages(self,image,savedir,foldername,filename):

        final_savedir = os.path.join(savedir,foldername)
        if not os.path.exists(final_savedir):
            os.mkdir(final_savedir)
        if '.tif' not in filename:
            filename = filename + '.tif'
        tifffile.imsave(os.path.join(final_savedir,filename),image)
        return final_savedir


    def padImagestack_runnable(self,imgstack,numpy=numpy):
        imgstack = numpy.asarray(imgstack)
        shape = imgstack.shape
        
        z = shape[0]
        x = shape[1]
        y = shape[2]
        diff = numpy.abs(x-y)
        if y == x:
            return imgstack
        
        elif x > y:
            y += diff
            newstack = numpy.pad(imgstack,((0,0),(0,0),(0,diff)),'constant',
                               constant_values = 0)
        elif y>x:
            x += diff
            
            newstack = numpy.pad(imgstack,((0,0),(0,diff),(0,0)),'constant',constant_values = 0)
        imgstack = None
        return newstack

    def padSingleImage_runnable(self,img,numpy=numpy):
        x,y = img.shape
        diff = numpy.abs(x-y)
        if x == y:
            return img
        elif x > y:
            y+= diff
            newimg = numpy.pad(img,((0,0),(0,diff)),'constant',constant_values = 0)
            img = None
            return newimg
        elif y > x:
            x += diff
            newimg = numpy.pad(img,((0,diff),(0,0)),'constant',constant_values = 0)
            img = None
            return newimg

    def getMaxPro_runnable(self,imgStack):
        return numpy.max(imgStack,axis = 0)
        
    def processImages(self,runnabledict = None,process_stack = True,process_others = False,
                                                process_chan = 'ch02', stack_to_process = None):
        """
        take a runnable_dict and apply to all the specified images, returning the values to a dictionary of results
        runnable_dict should look like {"name":str , "runnable": runnable},
        and the runnable should return a single value per image
        """

        if self.pool is None:
            return

        else:
        
            t0 = time.time()
            print('starting')
            
            if (runnabledict is None) or (self.stack_channel_images is None):
                print('returning None')
                return
            
            for ch in self.channels:
                if process_stack:
                    imgs = self.stack_channel_images[ch]
                    if ch not in self.processed_stack_images.keys():
                        self.processed_stack_images[ch] = {}
                    print('starting process %s for channel %s '%(runnabledict['name'],ch),'at ',
                        datetime.datetime.now(pytz.timezone('US/Pacific')).isoformat())
                    self.processed_stack_images[ch][runnabledict['name']] = self.pool.map(runnabledict['runnable'],imgs)
                    imgs = None
                    print('finishing process %s for channel %s '%(runnabledict['name'],ch),'at ',
                        datetime.datetime.now(pytz.timezone('US/Pacific')).isoformat() )

            if process_others:
                images_to_process = stack_to_process
                if process_chan not in self.processed_stack_images.keys():
                    self.processed_stack_images[process_chan] = {}
                self.processed_stack_images[process_chan][runnabledict['name']] = self.pool.map(runnabledict['runnable'],
                                                                                images_to_process)
                images_to_process = None

            return



class initialProcessor(object):

    def __init__(self, directory,cellData = None):
        """
        directory should be for image stacks
        """
        # CellData.__init__(self, directory, setupPool = True)
        self.basedir = os.path.dirname(directory)
        if cellData is None:
            self.cellData = CellData(directory,setupPool=True)
            self.cellData.loadImages()
        else:
            self.cellData = cellData
            print('cell data basedir:',self.cellData.basedir)

        print('basedir is: ',self.basedir)
        self.destruct = False
        

    @classmethod
    def BackgroundSubtraction_runnable(cls,image, strucsize=50, 
        disk = disk,white_tophat=white_tophat):
    
        struc = disk(strucsize)

        image = white_tophat(image,struc)

        return image

    def unloadData(self):
        self.cellData = None

    def run(self,save_results = True,saveMIPs = True):
        backsub_dict = {'name': 'Background Stack','runnable':initialProcessor.BackgroundSubtraction_runnable}
        self.cellData.processImages(runnabledict = backsub_dict)
        if save_results:
            directorydict = self.saveBackgroundImages(saveMIPs)
            self.destruct = True

        return directorydict


    def saveBackgroundImages(self,saveMIPs):
        subtracted_stackdirname = 'Background Subtracted Stack'
        newMIP_dirname = 'Background Subtracted MIP'
        directorydict = {}

        for ch in self.cellData.channels:

            stackfilenames = sorted([os.path.basename(x) for x in self.cellData.stack_channel_files[ch]])
            imageStack = self.cellData.processed_stack_images[ch]['Background Stack']

            for i,z in enumerate(imageStack):
               self.processed_dir = self.cellData.saveImages(z,self.cellData.basedir,subtracted_stackdirname,stackfilenames[i])

            if saveMIPs:
                newMIP = self.cellData.getMaxPro_runnable(imageStack)
                self.processed_mipdir = self.cellData.saveImages(newMIP,self.cellData.basedir,newMIP_dirname,'%s_BGMIP'%(ch))
                directorydict['Background MIP'] = self.processed_mipdir

        directorydict['Background Stack'] = self.processed_dir
        directorydict['goahead'] = True

        self.clearAttributes()
        return directorydict


    def clearAttributes(self):
        if self.destruct:
            self.cellData.unloadImages()
            self.cellData.unloadPool()




        




# practice_data_folder = os.getcwd()


# for item in os.listdir(os.getcwd()):
#     if 'data' in item.lower():
#         # print(item)
#         practice_data_folder = os.path.join(os.getcwd(),item)

# print(practice_data_folder)







# print('starting processing at ',datetime.datetime.now(pytz.timezone('US/Pacific')).isoformat())
# if __name__ == '__main__':
#     # testData = CellData(practice_data_folder,setupPool=True)
#     # testData.loadImages()
#     # for item in os.listdir(practice_data_folder):

#         # if os.path.isdir(os.path.join(practice_data_folder,item,'exmaple')):
#     testData = initialProcessor(directory = practice_data_folder)
#     # print(testData.stackdir)
#     # print('hello')



#     testData.loadImages()
#     # print(testData.stack_channel_images['ch01'][0].dtype)
#     testData.executeBackgroundSubtraction(saveMIPs = False)
#     # print(testData.processed_stack_images['ch02'].keys())
#     testData = None
#             testData.processImages(runnabledict = backsub_dict, process_stack=True)
#             filenames = {}
#             mipFileNames = {}
#             for ch in channels:
#                 filenames[ch] = sorted([x.split(os.sep)[-1] for x in testData.stack_channel_files[ch]])
#                 mipFileNames[ch] = [item + ch + 'MIP.tif']
#             for ch in channels:
#                 filenameSet = filenames[ch]
#                 ImgSet = testData.processed_stack_images[ch]['Background Stack']
#                 saveImages(ImgSet,savedir = Folder,foldername = 'Background_SubTracted_Images',
#                     filenames = filenameSet)
#                 MIP = getMaxPro_runnable(ImgSet)
#                 mipfolder = os.path.join(Folder,'BGsubMIPs')
#                 if not os.path.exists(mipfolder):
#                     os.mkdir(mipfolder)
#                 tifffile.imsave(os.path.join(mipfolder,mipFileNames[ch][0]),MIP)

                
#             testData = None
#         else:
#             pass
