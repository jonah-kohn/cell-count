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

		self.initializeCellData(setupPool=True)


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
				if 'Stack' in item:
					Ex.background_directory_info['Background Stack'] = os.path.join(Ex.basedir,item)




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
		Ex.loadshapeFilter()


	sys.exit()


















