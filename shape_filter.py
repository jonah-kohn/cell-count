from cell_count import CellData
import os,json
import numpy
from skimage.morphology import watershed, disk, square, remove_small_objects,opening
from scipy.ndimage.filters import gaussian_laplace
from skimage import filters
from skimage.measure import regionprops,label

def threshold_runnable(image,filters = filters):
	threshold = filters.threshold_otsu(image)
	return threshold

class shapeFilter(object):
	def __init__(self, directory,cellData = None):
		"""
		directory should contain background subtracted MIPs from ch01 and ch02
		"""
		self.directory = directory
		if cellData is None:
			self.cellData = CellData(self.directory,setupPool = False)
			self.cellData.loadImages()
		else:
			self.cellData = cellData
		
		


	def initialShapeFilter(self):
		openRedMIP = self.openImage_runnable(self.cellData.stack_channel_images[self.cellData.channels[1]][0])
		 #runs an opening to amplify separation between cells

		gaussianredmip = shapeFilter.gausLap_runnable(openRedMIP) 
		#gaussian smoothing for binary mask


		binary_gaussian_red = shapeFilter.getBinary_runnable(gaussianredmip,use_percentile=True,percentile = 0.5)
		self.cellData.saveImages(binary_gaussian_red.astype(numpy.uint16),self.cellData.basedir,'Labeled_Binary_Red','binary_mip')
		#creates binary mask using otsu

		binary_gaussian_red = shapeFilter.labelBinaryImage_runnable(binary_gaussian_red)
		self.cellData.saveImages(binary_gaussian_red.astype(numpy.uint16),self.cellData.basedir,'Labeled_Binary_Red','initial_labeling')

		binary_gaussian_red = shapeFilter.areaFilter_runnable(binary_gaussian_red)
		self.cellData.saveImages(binary_gaussian_red.astype(numpy.uint16),self.cellData.basedir,'Labeled_Binary_Red','binary_opened_mip')
		#removes small objects from binary mask

		Image_properties, binary_gaussian_red = shapeFilter.getImageCoordinates_runnable(binary_gaussian_red)
		# gets properties of labeled binary objects

		for i in range(len(Image_properties)):
			props = Image_properties[i]
			self.cellData.labeled_properties[i] = {
			'bbox': list(props.bbox),
			'area': int(props.filled_area),
			'y' : int(props.centroid[1]),
			'x' : int(props.centroid[0]),
			'diameter': int(props.equivalent_diameter),
			'label' : int(props.label)}

		self.cellData.processed_stack_images[self.cellData.channels[1]]['Labeled Binary Red'] = binary_gaussian_red
		self.cellData.saveImages(binary_gaussian_red.astype(numpy.uint16),self.cellData.basedir,
			'Labeled_Binary_Red','binary_mip_labeled')
		self.saveMetaData(foldername='Labeled_Binary_Red')

	@classmethod
	def areaFilter_runnable(cls,image,objectFilter = remove_small_objects,default_size = 20):
		return objectFilter(image,default_size)

	@classmethod
	def getBinary_runnable(cls,image,threshold = threshold_runnable,use_percentile = True,percentile = 0.7,np=numpy):
		img_threshold = threshold_runnable(image)
		if use_percentile:
			return np.asarray((image > percentile * img_threshold),dtype = np.int)
		else:
			return np.asarray((image > img_threshold),dtype=np.int)

	@classmethod
	def gausLap_runnable(cls,image,sigma = 3,gaussianLap = gaussian_laplace):
		return gaussianLap(image,sigma)

	@classmethod
	def labelBinaryImage_runnable(cls,image,label = label,neighbors = 8):
		return label(image,neighbors = neighbors)

	@classmethod
	def getImageCoordinates_runnable(cls,image,intensity_image = None,regionprops=regionprops):
		labeledimg = shapeFilter.labelBinaryImage_runnable(image)
		if intensity_image is not None:
			props = regionprops(labeledimg,intensity_image=intensity_image)
		else:
			props = regionprops(labeledimg)
		return props, labeledimg


	def openImage_runnable(self,image,opening = opening, selem = square(1)):
		return opening(image,selem)

	def unloadImages(self):
		if self.cellData is not None:
			self.cellData.unloadImages()

	def removeObjectsByLabel_runnable(self,labeled_image_cutout,image_properties):
		objectlabel = image_properties['label']
		clean_image = labeled_image_cutout*(labeled_image_cutout == label).astype(int)
		return clean_image

	def saveMetaData(self,foldername):
		if len(self.cellData.labeled_properties) != 0:
			filepath = os.path.join(self.cellData.basedir,foldername,'measured_properties.json') 

			if not os.path.exists(os.path.join(self.cellData.basedir,foldername)):
				os.mkdir(os.path.join(self.cellData.basedir,foldername))
			with open(filepath,'w') as f:
				json.dump(self.cellData.labeled_properties,f)











# for item in os.listdir(os.getcwd()):
#     if 'data' in item.lower():
#         practice_data_folder = os.path.join(os.getcwd(),item,'06_R','BGsubMIPs')
#         print(practice_data_folder)

# atest = ShapeFilter(practice_data_folder)
# atest = None
# datalabel = '06'
# print('starting processing at ',datetime.datetime.now(pytz.timezone('US/Pacific')).isoformat())
# if __name__ == '__main__':
#     for item in os.listdir(practice_data_folder):
#     	if datalabel in item:
# 	        if os.path.isdir(os.path.join(practice_data_folder,item)):
# 	            Folder = os.path.join(practice_data_folder,item,'BGsubMIPs')
# 	            testDataMIPs = CellData(directory = Folder)
# 	            # print(Folder)



# 	            testDataMIPs.loadImages()

# 	            newredMIP = testDataMIPs.stack_channel_images['ch02'][0]

# 	            openRedMIP = openImage_runnable(newredMIP)

# 	            gaussianredmip = gausLap_runnable(openRedMIP)

# 	            binary_gaussian_red = getBinary_runnable(gaussianredmip,use_percentile=True,percentile = 0.5)
# 	            binary_gaussian_red = labelBinaryImage_runnable(binary_gaussian_red)

# 	            testDataMIPs.processed_stack_images['ch02'] = {}
# 	            testDataMIPs.processed_stack_images['ch02']['Binarized Labeled'] = binary_gaussian_red

# 	            area_filtered_binary = areaFilter_runnable(binary_gaussian_red).astype(numpy.uint8)

# 	            labeled_area_filtered = labelBinaryImage_runnable(area_filtered_binary)
# 	            mipProps = getImageCoordinates_runnable(labeled_area_filtered)

# 	            for i in range(len(mipProps)):
# 	            	props = mipProps[i]
# 	            	properties = {'bbox' : props.bbox,
# 	            						'area' : int(props.filled_area),
# 	            						'y' : int(props.centroid[1]),
# 	            						'x' : int(props.centroid[0]),
# 	            						'diameter' : int(props.equivalent_diameter),
# 	            						'label' : props.label}
# 	            	testDataMIPs.labeled_properties[i] = properties

# 	            jsonfilename = os.path.join(practice_data_folder,item,'Labeled_binary_red','measured_properties.json')
	            
# 	            saveImages(area_filtered_binary,
# 	            	os.path.join(practice_data_folder,item),'Labeled_binary_red','red_binary_mip')
	            

# 	            with open(jsonfilename,'w') as f:
# 	            	json.dump(testDataMIPs.labeled_properties,f)



# 	            testDataMIPs = None