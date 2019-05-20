from cell_count import CellData
import os,json
import numpy
from skimage.morphology import watershed, disk, square, remove_small_objects,opening
from scipy.ndimage.filters import gaussian_laplace
from skimage.filters import gaussian,laplace, threshold_otsu
from skimage.measure import regionprops,label
import scipy.ndimage as ndi
from collections import OrderedDict
from dicttoxml import dicttoxml
import xmltodict
from skimage.feature import peak_local_max

def threshold_runnable(image,threshold = threshold_otsu):
	thresh = threshold(image)
	return thresh

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
		# openRedMIP = self.openImage_runnable(self.cellData.stack_channel_images[self.cellData.channels[1]][0])
		 #runs an opening to amplify separation between cells

		gaussianredmip = gaussian(self.cellData.stack_channel_images[self.cellData.channels[1]][0],sigma = 7)
		gaussianredmip = laplace(gaussianredmip)
		#gaussian smoothing for binary mask


		binary_gaussian_red = shapeFilter.getBinary_runnable(gaussianredmip,use_percentile=True,percentile = 0.5)

		self.cellData.saveImages(binary_gaussian_red.astype(numpy.uint16),self.cellData.basedir,
															'Labeled_Binary_Red','binary_mip')
		#creates binary mask using otsu

		binary_gaussian_red = shapeFilter.labelBinaryImage_runnable(binary_gaussian_red)

		self.cellData.saveImages(binary_gaussian_red.astype(numpy.uint16),self.cellData.basedir,
														'Labeled_Binary_Red','initial_labeling')

		binary_gaussian_red = shapeFilter.areaFilter_runnable(binary_gaussian_red)

		self.cellData.saveImages(binary_gaussian_red.astype(numpy.uint16),self.cellData.basedir,
														'Labeled_Binary_Red','binary_opened_mip')
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


	def countCells(self,stackCellData):
		image_properties = self.cellData.labeled_properties
		binary_gaussian_red = self.cellData.processed_stack_images[self.cellData.channels[1]]['Labeled Binary Red']


		cell_count_list = []

		for key in sorted(image_properties.keys()):

			item = image_properties[key]
			print(key,item['label'])
			x,y = item['x'],item['y']
			item['z'] = 1 
			item['type'] = 1

			x1,x2 = item['bbox'][0],item['bbox'][2]

			y1,y2 = item['bbox'][1],item['bbox'][3]

			cutoutsize = int(2*item['diameter'])
			
			objectarea = int(item['area'])

			binary_cutout_image = self.cutImageByBoundary(binary_gaussian_red,xstart=x1,xstop=x2,ystart=y1,ystop=y2)

			binredcut = shapeFilter.getImageCutouts_runnable(binary_cutout_image,x,y,cutout=cutoutsize)




			binredcut = self.removeObjectsByLabel_runnable(binredcut,item)
			print(binredcut.shape)

			fieldStacks = self.getCutoutFieldStacks_runnable(stackCellData,item)


			smallredstack = fieldStacks['red_stack']
			smallgreenstack = fieldStacks['green_stack']
			redhist = fieldStacks['sigmas']
			greenbin = fieldStacks['green_bin']
			redbin = fieldStacks['red_bin']

			print(greenbin.shape,redbin.shape)

			fieldStacks = None
			finalfield = []
			
			for i in range(len(smallredstack)):


				if redhist[i] > numpy.percentile(redhist,20):


					smallbingreen = greenbin[i]

					smallbinred = redbin[i]


					labeledGreen1 = shapeFilter.labelBinaryImage_runnable(smallbingreen)
					labeledGreen1 = remove_small_objects(labeledGreen1,5)

					themax = numpy.amax(labeledGreen1)

					# if themax == 1: 
					# 	pass

					# else:
					labeledRed = shapeFilter.labelBinaryImage_runnable(smallbinred)
					fieldsize = labeledGreen1.shape[0]*labeledGreen1.shape[1]
					strucsize = int(0.25*fieldsize)

					trial = ndi.distance_transform_edt(smallbingreen)
					local_max =  peak_local_max(trial,indices = False,labels = labeledGreen1,min_distance=3)

					markers = ndi.label(local_max)[0]

					watershedGreen = watershed(-trial,markers,mask = labeledGreen1, watershed_line=True)

					labeledGreen = label(watershedGreen,8)

					greenfieldProps,labeledGreen = shapeFilter.getImageCoordinates_runnable(labeledGreen)



					gareas = []
					subimg = numpy.zeros(labeledGreen.shape)
					for gprop in greenfieldProps:
						props = {}
						props['area'] = gprop.filled_area
						props['label'] = gprop.label
						gareas.append(props)

					for area in gareas:
						if area['area'] < strucsize:

							subimg = numpy.zeros(labeledGreen.shape)
							pass
						else:

							subimg = area['label']*(labeledGreen == area['label'])
							labeledGreen = labeledGreen - subimg
					# print('max green',numpy.amax(labeledGreen))

					redfieldProps = shapeFilter.getImageCoordinates_runnable(labeledRed)
					subimg = numpy.zeros(labeledRed.shape)
					rareas = []
					for rprop in redfieldProps[0]:
						props = {}
						props['area'] = rprop.filled_area
						props['label'] = rprop.label
						rareas.append(props)

					for area in rareas:
						if area['area'] < strucsize:

							subimg = numpy.zeros(labeledGreen.shape)
							pass
						else:

							subimg = area['label']*(labeledRed == area['label'])
							labeledRed = labeledRed - subimg

					labeledRed = label(remove_small_objects(labeledRed,5),4)
					# print('max labeled red',numpy.amax(labeledRed))

					# print('binred shape', binredcut.shape,'\n','label green shape',labeledGreen.shape)

					product = label(labeledRed*labeledGreen,neighbors = 8,connectivity = 2)
					# print('max of product',numpy.amax(product))

					final_product = binredcut*product
					final_product = remove_small_objects(label(final_product),5)

					labeledproductprops,labeled_final_product = shapeFilter.getImageCoordinates_runnable(final_product)
					# print("Detection Number:",len(labeledproductprops))

					if len(labeledproductprops) != 0:
						# print('appending')
						finalfield.append(final_product)


			# print(len(finalfield),key)
			if len(finalfield) != 0:
				print('cell found at:',key,item['label'])

				cellDict = OrderedDict()
				cellDict['MarkerX'] = item['y']
				cellDict['MarkerY'] = item['x']
				cellDict['MarkerZ'] = item['z']
				cell_count_list.append(cellDict)
				print("Count is: \t",len(cell_count_list))

		return cell_count_list


	def cutImageByBoundary(self,img,xstart,xstop,ystart,ystop):
		newImage = numpy.zeros(img.shape)
		newImage[xstart:xstop,ystart:ystop] = img[xstart:xstop,ystart:ystop]
		newImage >= 0
		newImage = newImage.astype(int)
		return newImage

	def getMaxPro_runnable(self,imgStack):
		imgStack = numpy.asarray(imgStack)
		return numpy.max(imgStack,axis=0)



	@classmethod
	def areaFilter_runnable(cls,image,objectFilter = remove_small_objects,default_size = 20):
		return objectFilter(image,default_size)

	@classmethod
	def getBinary_runnable(cls,image,threshold = threshold_runnable,use_percentile = True,
							percentile = 0.4,np=numpy):
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

	@classmethod
	def getImageCutouts_runnable(cls,img,x,y,cutout=50):
		if x < cutout:
			xstart = 0
			xstop = cutout
			# print('xstart',xstart,'xstop',xstop)
		else:
			xstart,xstop = x - int(cutout/2), x + int(cutout/2)
		if y < cutout:
			ystart = 0
			ystop = cutout
			# print('ystart',ystart,'ystop',ystop)
		else:
			ystart,ystop = y - int(cutout/2), y + int(cutout/2)
		newimg = img[xstart:xstop,ystart:ystop]
		# print('cutoutshape',newimg.shape)
		return newimg


	def getCutoutFieldStacks_runnable(self,cellDataObject,image_properties):
		assert(type(cellDataObject == CellData))

		x,y = image_properties['x'],image_properties['y']

		redstack = cellDataObject.stack_channel_images['ch02']
		greenstack = cellDataObject.stack_channel_images['ch01']

		assert(len(redstack)==len(greenstack))

		smallredstack = []
		smallgreenstack = []
		redhist = []
		greenbin = []
		redbin = []

		cutoutsize = int(2*image_properties['diameter'])
		# print('Field cutoutsize',cutoutsize)

		for r,g in zip(redstack,greenstack):

			redcut = shapeFilter.getImageCutouts_runnable(r,x,y,cutout=cutoutsize)
			redhist.append(numpy.std(redcut))
			redcut = opening(redcut,selem = square(1))
			smallbinred = shapeFilter.getBinary_runnable(redcut,use_percentile = True,percentile = .5)

			greencut = shapeFilter.getImageCutouts_runnable(g,x,y,cutout=cutoutsize)
			greencut = opening(greencut,selem = square(1))

			smallbingreen = shapeFilter.getBinary_runnable(greencut,use_percentile=True,percentile = .3)

			smallredstack.append(redcut)
			redbin.append(smallbinred)

			smallgreenstack.append(greencut)
			greenbin.append(smallbingreen)



		smallredstack = numpy.asarray(smallredstack)
		smallgreenstack = numpy.asarray(smallgreenstack)
		redhist = numpy.asarray(redhist)
		greenbin = numpy.asarray(greenbin)
		redbin = numpy.asarray(redbin)

		dataDictionary = {'red_stack' : smallredstack,
						'sigmas' : redhist,
						'green_stack' : smallgreenstack,
						'green_bin' : greenbin,
						'red_bin' : redbin}
		return dataDictionary


	def openImage_runnable(self,image,opening = opening, selem = square(1)):
		return opening(image,selem)

	def unloadImages(self):
		if self.cellData is not None:
			self.cellData.unloadImages()

	def removeObjectsByLabel_runnable(self,labeled_image_cutout,image_properties):
		objectlabel = image_properties['label']
		clean_image = labeled_image_cutout*(labeled_image_cutout == objectlabel).astype(int)
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