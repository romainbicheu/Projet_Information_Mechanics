####
# This code is from the mt3000/Information_Mechanics repository, utils.py file.
###

import numpy as np
from scipy import ndimage


from tensorflow.nn import depthwise_conv2d
from tensorflow.math import multiply, reduce_sum, reduce_mean,reduce_euclidean_norm, sin, cos, abs
from tensorflow import stack, concat, expand_dims, reshape, transpose

import tensorflow_probability as tfp
from scipy.fftpack import dct, idct

from tensorflow.image import flip_up_down, flip_left_right, rot90

from cv2 import getDerivKernels



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from io import BytesIO


def get_filter(model, layer , sev=False):

	conv_layers = []
	for l in model.layers:
		if 'conv2d' in str(type(l)).lower():
			if l.kernel_size == (3,3) or (l.kernel_size == (7,7) and sev):
				conv_layers.append(l)

	layer = conv_layers[layer]
	# check for convolutional layer
	if 'conv' not in layer.name:
		raise ValueError('Layer must be a conv. layer')
	# get filter weights
	filters, biases = layer.get_weights()
	#print("biases shape : ", biases.shape)
	#print("filters shape : ", filters.shape)
	#print(layer.name)
	return (filters)
	#print(layer.name, filters.shape)


'''def getDominantAngle(filters):
	theta = getSobelTF(filters)
	print(filters.shape)
	s, a = getSymAntiSymTF(filters)
	a_mag = reduce_euclidean_norm(a, axis=[0,1])
	s_mag = reduce_euclidean_norm(s, axis=[0,1])

	mag = reduce_euclidean_norm(filters, axis=[0,1])


	domTheta = []
	for i in range(filters.shape[-1]):
		x =(a_mag[:,i]*np.cos((theta[:,i]))).numpy()
		y =( a_mag[:,i]*np.sin((theta[:,i]))).numpy()


		cov = np.cov([x,y])
		e_val, e_vec = np.linalg.eig(cov)
		e_vec = e_vec[:, np.argmax(e_val)]
		e_val = np.max(e_val)
		if np.sign(e_vec[0]) != np.sign(x[np.argmax(np.abs(x))]):
			e_vec *= -1
		domTheta.append(np.arctan2(e_vec[1], e_vec[0]))
	#x =a_mag[:,f_num]*np.cos((theta[:,f_num]))
	#y = a_mag[:,f_num]*np.sin((theta[:,f_num]))

	return np.array(domTheta)'''


def getDominantAngle(filters):
	domTheta = []
	vec = []
	old_vec = []

	_, a = getSymAntiSymTF(filters)
	a_mag = reduce_euclidean_norm(a, axis=[0,1])
	theta = getSobelTF(filters)
	for i in range(filters.shape[-1]):

		#print(a_mag.shape)
		x =a_mag[:, i]*np.cos((theta[:, i]))
		y = a_mag[:, i]*np.sin((theta[:, i]))

		u_x = np.mean(x)
		#print(u_x)
		u_y = np.mean(y)
		cov = np.cov([x, y])
		e_val, e_vec = np.linalg.eigh(cov)
		#print(e_val, e_vec)
		e_vec = e_vec[:, np.argmax(e_val)]
		e_val = np.max(e_val)

		new_vec =    ((e_vec[0] * x + e_vec[1]*y)/(e_vec[0]**2+e_vec[1]**2))[:, None] * e_vec
		new_vec = np.mean(new_vec, axis=0)

		'''if any((np.sign(new_vec)-np.sign(e_vec))!=0) :
			print("DOWN", e_vec, new_vec)
		else:
			print("OK", e_vec, new_vec)'''
		'''if np.sign(e_vec[0]) != np.sign(x[np.argmax(np.abs(x))]):
			e_vec *= -1'''
			#print(e_val, e_vec)
		#print(np.arctan2(e_vec[1], e_vec[0]))
		domTheta.append(np.arctan2(new_vec[1], new_vec[0]))
		vec.append(new_vec)
		#old_vec.append(e_vec)

	return np.array(vec), np.array(domTheta)#, old_vec


def getSobelTF(f):

	ksize = f.shape[0]
	sobel = getDerivKernels(1,0,ksize=ksize, normalize=True)
	sobel_v = -np.expand_dims(np.expand_dims(np.outer(sobel[0], sobel[1]), -1),-1)  # * -1
	sobel = getDerivKernels(0,1,ksize=ksize, normalize=True)
	sobel_h = np.expand_dims(np.expand_dims(np.outer(sobel[0], sobel[1]), -1),-1)

	#print(sobel_h, sobel_v)

	s_h = reduce_sum(multiply(f, sobel_h), axis=[0,1])
	s_v = reduce_sum(multiply(f, sobel_v), axis=[0,1])


	'''sobel_v = np.expand_dims(np.expand_dims(np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=np.float32)/-4., -1),-1)
	sobel_h = np.expand_dims(np.expand_dims(np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype=np.float32)/-4., -1) ,-1) 
	#print(sobel_h, sobel_v)

	s_h = reduce_sum(multiply(f, sobel_h), axis=[0,1])
	s_v = reduce_sum(multiply(f, sobel_v), axis=[0,1])'''

	return (np.arctan2(s_v,s_h))




def getSymAntiSymTF(filter):

	#patches = extract_image_patches(filters, [1, k, k, 1],  [1, k, k, 1], rates = [1,1,1,1] , padding = 'VALID')
	#print(patches)
	'''a = filter[0,0,:,:]
	b = filter[0,1,:,:]
	c = filter[0,2,:,:]
	d = filter[1,0,:,:]
	e = filter[1,1,:,:]
	f = filter[1,2,:,:]
	g = filter[2,0,:,:]
	h = filter[2,1,:,:]
	i = filter[2,2,:,:]

	fs1 = expand_dims(a+c+g+i, 0)/4
	fs2 = expand_dims(b+d+f+h,0)/4
	fs3= expand_dims(e, 0)

	sym = stack([concat([fs1, fs2, fs1],  axis=0), 
						 concat([fs2, fs3, fs2], axis=0),
						 concat([fs1, fs2, fs1], axis=0)])
		
	anti = filter - sym'''

	f_reshaped = transpose(filter, perm=[3, 0, 1, 2])
	mat_flip_x = flip_left_right(f_reshaped)

	mat_flip_y = flip_up_down(f_reshaped)
	mat_flip_xy = flip_left_right(flip_up_down(f_reshaped))
	#print(mat_flip_x.shape, mat_flip_y.shape, mat_flip_xy.shape)
	sum = f_reshaped + mat_flip_x + mat_flip_y + mat_flip_xy
	
	mat_sum_rot_90 = rot90(sum)
	#gc.collect()
	#print("mat_sum_rot_90 shape " , mat_sum_rot_90.shape, self._name)
	
	#print("OUT SHAPE," , out.shape)
	out = (sum + mat_sum_rot_90) / 8
	
	sym = transpose(out, perm=[1, 2, 3, 0])
	anti = filter - sym
	return  sym, anti




def topKfilters(model, layer_num, k=10, sev=False):
	#print(i, l.name)
	filters = get_filter(model, layer_num, sev)

	mag = reduce_euclidean_norm(filters, axis=[0,1])**2
	avg_mag = reduce_mean(mag, axis=0)
	idx = list(range(mag.shape[-1]))
	
	idx = [x for _, x in sorted(zip( avg_mag, idx), reverse=True)]
	return idx[:int(np.floor(len(idx)*k/100))]

def topKchannels(model, layer_num, f_num, k=10, sev=False):
	#print(i, l.name)
	filters = get_filter(model, layer_num, sev)[:,:,:,f_num]

	mag = reduce_euclidean_norm(filters, axis=[0,1])**2
	#avg_mag = reduce_mean(mag, axis=0)
	idx = list(range(mag.shape[-1]))
	'''if int((k/100)*len(idx)) == 0:
		return idx'''
	
	idx = [x for _, x in sorted(zip( mag, idx), reverse=True)]
	return idx[:int(np.floor(len(idx)*k/100))]


def dct2(a):
	return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
	return idct(idct(a.T, norm='ortho').T, norm='ortho')    




def plot_filter_x(beta2 , ax=None):

	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",         # Computer Modern by default
		"font.serif": ["Computer Modern Roman"],
	})


	# Example point
	X, Y, Z = 4*np.sqrt(beta2), 0, 4*np.sqrt(1-beta2)
	print(X,Z)
	if ax is None:
	

		fig = plt.figure(figsize=(8, 6))
		ax = fig.add_subplot(111, projection='3d')

	

	# Plot the XY plane
	xx, yy = np.meshgrid(range(-5, 6), range(-5, 6))
	zz = np.zeros_like(xx)
	ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue')

	# Draw the coordinate axes
	ax.quiver(0, 0, 0, -5, 0, 0, color='k', arrow_length_ratio=0.1)
	ax.quiver(0, 0, 0, 5, 0, 0, color='k', arrow_length_ratio=0.1)

	ax.text(5.9, -0.5, 0, r"$\nabla_x$", fontsize=18, ha='center')
	ax.quiver(0, 0, 0, 0, -5, 0, color='k', arrow_length_ratio=0.1)
	ax.quiver(0, 0, 0, 0, 5, 0, color='k', arrow_length_ratio=0.1)

	ax.text(0.5, 5.1, 0.0, r"$\nabla_y$", fontsize=18, ha='center')
	ax.quiver(0, 0, 0, 0, 0, 5, color='k', arrow_length_ratio=0.1)
	ax.quiver(0, 0, 0, 0, 0, -5, color='k', arrow_length_ratio=0.1)

	ax.text(0, 0, 5.5, r"$||f_s|| \cdot sign(\Sigma)$", fontsize=16, ha='center')

	# Plot the point
	ax.scatter(X, Y, Z, color='k', s=200)

	# Dotted projection lines
	ax.plot([X, X], [Y, Y], [0, Z], 'k:', linewidth=1.5)
	ax.plot([0, X], [Y, Y], [Z, Z], 'k:', linewidth=1.5)

	# Set limits and remove axes panes
	ax.set_xlim(-5, 5)
	ax.set_ylim(-5, 5)
	ax.set_zlim(0, 5)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	ax.set_axis_off()

	ax.set_box_aspect([1,1,0.5])  # makes axes proportionate
	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove margins
	plt.margins(0, 0, 0)             # no extra margins

	'''buf = BytesIO()
	plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=fig.dpi, transparent=True)

	buf.seek(0)
	plt.close(fig)
	# Load as PIL Image
	img = Image.open(buf)
	img_cropped = img.crop(img.getbbox())  # trim transparent borders

	return(img_cropped)'''
