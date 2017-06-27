from __future__ import print_function

import os
import csv
import glob
import operator

import numpy as np

from collections import namedtuple
from ast import literal_eval
from itertools import chain, groupby
#from natsort import natsorted, ns

from PIL import Image
from chainercv.transforms import resize

# these are only necessary for the test functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

target = namedtuple("target", "frame id bb_left bb_top bb_width bb_height")

class MOT:

	def __init__(self, base_path, sequence="MOT16-02", train=True, frame_range=None):
		self.sequence = sequence
		self.train = train

		# define paths to everything
		if self.train:
			self.full_path = os.path.join(base_path, "train", self.sequence)
			self.groundtruth_path = os.path.join(self.full_path, "gt")
		else:
			self.full_path = os.path.join(base_path, "test", self.sequence)
			self.groundtruth_path = None

		self.image_path = os.path.join(self.full_path, "img1") # "img1" is also defined in seqinfo.ini

		# load sequence information
		self.seqinfo = self.load_sequence_info()

		self.gt_track = None

		print(self.full_path)
		print(self.seqinfo)

		# load everything
		self.load_images()
		self.load_gt()
		#self.load_bboxes()

	def __len__(self):
		if self.gt_track is None:
			self.load_gt()
		return len(self.gt_track)

	def load_sequence_info(self):
		# get the summary file
		info_file = os.path.join(self.full_path, "seqinfo.ini")
		seqinfo = {}
		with open(info_file, "r") as f:
			reader = csv.reader(f, delimiter="=")
			next(reader, None)  # skip the headers

			for item in reader:
				if len(item) == 2:
					seqinfo[item[0]] = item[1]
			
		# hacky way to turn strs into ints
		seqinfo["seqLength"] = int(seqinfo["seqLength"])
		seqinfo["imHeight"] = int(seqinfo["imHeight"])
		seqinfo["imWidth"] = int(seqinfo["imWidth"])

		return seqinfo

	def load_images(self):
		"""
		Load images from image folder
		"""

		#img_wildcard = os.path.join(self.image_path, "*{}".format(self.seqinfo["imExt"]))
		#self.img_files = sorted(glob.glob(img_wildcard))
		
		self.img_files = []
		for i in range(1, self.seqinfo["seqLength"] +1):
			num = "{}".format(i).zfill(6)
			num = "{}{}".format(num, self.seqinfo["imExt"])
			#print(num)
			self.img_files.append(os.path.join(self.image_path, num))

		#for img in self.img_files: 
		#	print(img)

		"""self.images = []

		print("Loading images...")
		if load_to_memory:
			for image in self.img_files:
				img = Image.open(image)
				self.images.append(np.asarray(img, dtype=np.float32))
		print("done.")"""

	def load_gt(self):
		"""
		Load ground truth
		"""

		if self.train is False:
			print("No ground truth info with test data set!")
			return

		# ground truth not organized 
		self.gt = []

		gt_file = os.path.join(self.groundtruth_path, "gt.txt")
		
		with open(gt_file, "r") as f:
			annotations = csv.reader(f, delimiter=",")
			
			for row in annotations:
				#convert from string
				row = [literal_eval(el) for el in row]
				if row[6] > 0.0 or row[6] < 0.0:
					_target = target(frame=row[0]-1, id=row[1]-1,
						    		 bb_left=row[2], bb_top=row[3],
									 bb_width=row[4], bb_height=row[5])
					self.gt.append(_target)

		def group_by_attr(_list, attr):
			id_attr = operator.attrgetter(attr)
			return [list(g) for k, g in groupby(sorted(_list, key=id_attr), id_attr)]

		# ground truth organized by track
		self.gt_track = group_by_attr(self.gt, "id")
		
		# ground truth organized by frame 
		self.gt_frame = group_by_attr(self.gt, "frame")

	def load_bboxes(self):
		"""
		load bounding box-cropped images into memory
		"""
		print("loading bbox cropped images to memory...")
		if self.gt_track is None:
			self.load_gt()
			self.load_images()

		self.bbox_track = []
		for i, track in enumerate(self.gt_track):
			print("{} / {}".format(i, len(self.gt_track)))
			
			#print(track)
			track_bboxes = []
			for detection in track:
				im = Image.open(self.img_files[detection.frame])
				im = np.asarray(im)

				y1 = detection.bb_top
				y2 = y1 + detection.bb_height
				x1 = detection.bb_left
				x2 = detection.bb_left+detection.bb_width

				# make sure box doesnt go off screen
				if y1 < 0:
					y1 = 0
				if x1 < 0:
					x1 = 0
				if y2 > self.seqinfo["imHeight"]:
					y2 = self.seqinfo["imHeight"]
				if x2 > self.seqinfo["imWidth"]:
					y2 = self.seqinfo["imWidth"]

				im = im[y1:y2, x1:x2, :]
				
				#plt.figure()
				#plt.imshow(im)
				#plt.show()
				box = im #resize(im.transpose(2,0,1), (224, 224))
				track_bboxes.append(box)
				#bbox[i-1] = box
				#plt.imshow(box)

			self.bbox_track.append(track_bboxes)

def open_frame(data, frame):
	print("frame {}".format(frame))
	img = mpimg.imread(data.img_files[frame])
	plt.figure()
	plt.imshow(img)
	ax = plt.gca()

	for det in data.gt_frame[frame]:
		# http://matthiaseisen.com/pp/patterns/p0203/
		ax.add_patch(Rectangle((det.bb_left, det.bb_top), det.bb_width, det.bb_height, edgecolor=cm.jet(det.id/100.0), linewidth=3, fill=None))

	#plt.show()
	plt.savefig("img/{}.png".format(frame))

if __name__=="__main__":
	data = MOT("/mnt/sakuradata3/datasets/MOT/MOT16/", sequence="MOT16-04", train=True)

	for i in range(100):
		open_frame(data, i)

