from cell_count import CellData, initialProcessor
from shape_filter import shapeFilter
import numpy
import os,json
from skimage.morphology import watershed, disk, square, remove_small_objects,opening
from skimage import filters
from skimage.measure import regionprops,label
import sys
import tifffile
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QMessageBox
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
from dicttoxml import dicttoxml,parseString,convert_dict
import xmltodict


basicxml = [os.path.join(os.getcwd(),f) for f in os.listdir(os.getcwd()) if 'basicxml' in f]


class Executor(QWidget):
	def __init__(self,basedir=None,stack_folder='Stack'):
		self.stausDict = {'Ready':'','Processing Background':'','Initial Shape Filtering': '',}
		self.background_directory_info = {'Background MIP': None,
										'Background Stack': None,
										'goahead' : False}

		if basedir is None:
			basedir = self.getDataPath()
			self.basedir = basedir
			print('basedir is: ',self.basedir)
		else:
			self.basedir = basedir


		self.stack_folder = os.path.join(self.basedir,stack_folder)

		self.cellData = None

		self.cell_count_list = None


		self.initializeCellData(setupPool=True)
		print(os.getcwd())

		with open(basicxml[0],'r') as f:
			self.saveStructure = xmltodict.parse(f.read())


	def initializeCellData(self,setupPool,directory = None):
		if directory is None:
			self.cellData = CellData(self.stack_folder,setupPool = setupPool)
			
		else:
			self.cellData.unloadImages()
			self.cellData.unloadPool()
			self.cellData = None

			self.cellData = CellData(directory,setupPool=setupPool)


		self.cellData.loadImages()
			

	def getDataPath(self):
		datapath = QFileDialog.getExistingDirectory(caption='Choose Directory',
													options = QFileDialog.ShowDirsOnly)
		datapath = str(datapath)
		return datapath
	
	def loadInitialProcessor(self,goahead = False):
		assert(self.cellData is not None)
		self.processor = initialProcessor(directory = self.basedir,cellData=self.cellData)
		print(self.processor.basedir)
		if goahead:
			self.background_directory_info = self.processor.run()



	def loadshapeFilter(self):
		newdir = self.background_directory_info['Background MIP']
		self.initializeCellData(setupPool=False,directory = newdir)

		self.shapeFilter = shapeFilter(directory=newdir,cellData=self.cellData)

		goahead = self.background_directory_info['goahead']
		if goahead:
			self.shapeFilter.initialShapeFilter()
			return True
		else:
			return False

	def countCells(self):
		stackCellData = self.getStackCellData()
		stackCellData.loadImages()
		self.cell_count_list = self.shapeFilter.countCells(stackCellData)

	def getStackCellData(self):
		stackdir = self.background_directory_info['Background Stack']
		stackCellData = CellData(stackdir,setupPool=False)
		return stackCellData

	def saveCount(self):
		self.saveStructure['CellCounter_Marker_File']["Marker_Data"]['Marker_Type'][0]["Marker"] = self.cell_count_list
		self.saveStructure['CellCounter_Marker_File']["Image_Properties"]["Image_Filename"] = os.path.basename(self.basedir) + ".tif"

		savefile = os.path.join(self.basedir,os.path.basename(self.basedir) + "_cellCount.xml")

		thestring = xmltodict.unparse(self.saveStructure,pretty=True)
		print(savefile)
		print(type(thestring))

		with open(savefile,'w') as f:
			f.write(thestring)





	def quit(self):
		print('quitting')
		QCoreApplication.quit()




if __name__ == '__main__':
	app = QApplication(sys.argv)
	Ex = Executor(basedir = None)
	messagebox = QMessageBox()
	messagebox.setText('Do you want to run a Background Subtraction?')
	messagebox.setStandardButtons(QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel)

	ret = messagebox.exec_()
	if ret == QMessageBox.Yes:
		sysgoahead = True
		Ex.loadInitialProcessor(goahead = True)
	if ret == QMessageBox.No:
		for item in os.listdir(Ex.basedir):
			print(item)
			if 'Background' in item:
				if 'MIP' in item:
					Ex.background_directory_info['Background MIP'] = os.path.join(Ex.basedir,item)
				if 'Images' in item:
					Ex.background_directory_info['Background Stack'] = os.path.join(Ex.basedir,item)
					print(Ex.background_directory_info)




		if Ex.background_directory_info['Background MIP'] is None:
			newMessage = QMessageBox()
			newMessage.setText('System could not find background subtracted datasets. Data must be background subtracted to run count method')
			newMessage.setStandardButtons(QMessageBox.Ok)

			newret = newMessage.exec_()
			sysgoahead = False
		
		else:
			sysgoahead = True
			Ex.background_directory_info['goahead'] = sysgoahead




	if ret == QMessageBox.Cancel:
		print('Exiting')
		sys.exit()

	if not sysgoahead:
		print('Exiting')
		sys.exit()

	else:
		print('Finding Red Cells')
		initial_process = Ex.loadshapeFilter()

		if initial_process:

			messagebox = QMessageBox()
			messagebox.setText('Red Cells Detected. Would you like to execute counting method?')
			messagebox.setStandardButtons(QMessageBox.Yes|QMessageBox.No)

			ret = messagebox.exec_()

			if ret == QMessageBox.Yes:
				Ex.countCells()
				Ex.saveCount()

			if ret == QMessageBox.No:
				print('Exiting')

	print('Exiting')


	sys.exit()

















