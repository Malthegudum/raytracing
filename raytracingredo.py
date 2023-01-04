import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import image
from random import randint, uniform
import random
import time

# Creates the scene by setting values for spheres
def createScene():
	global light
	global sphere
	global ground
	global sky

	light = np.array([10, 120, 100])

	ground = [0, 0.2, np.asarray(Image.open("./assets/skak.jpg")), 1]
	sky = [np.asarray(Image.open("./assets/skypan.jpg")), np.asarray(Image.open("./assets/skytop.jpg"))]

	nspheres = 5
	sphere = [] #[[1, np.array([3, 0, 1]), np.array([100, 30, 140]), 0.5]]
	
	for i in range(nspheres):
		r = uniform(0.2, 1.5)
		center = np.array([uniform(2, 8), uniform(-4, 4), r])
		colour = np.array([randint(0, 255), randint(0, 255), randint(0, 255)])
		reflectivity = random.random()
		sphere.append([r, center, colour, reflectivity])

# Function for rotating vectors in 3D-space
def rotate3d(a, b, c):
	a *= 0.0175
	b *= 0.0175
	c *= 0.0175

	rx = np.array([[1,         0,          0],
	               [0, np.cos(a), -np.sin(a)],
	               [0, np.sin(a), np.cos(a)]])

	ry = np.array([[ np.cos(b), 0, np.sin(b)],
	               [         0, 1,         0],
	               [-np.sin(b), 0, np.cos(b)]])

	rz = np.array([[np.cos(c), -np.sin(c), 0],
	               [np.sin(c),  np.cos(c), 0],
	               [        0,          0, 1]])

	return np.dot(np.dot(rx, ry), rz)

# Finds the color of a pixel given the directionvector of the ray of the pixel
def intersection(sphere, ground, sky, light, rv, campos, bounces):
	tmin = None
	imin = None
	shade = 1
	for i in range(len(sphere)):
		t = None

		# Finding the value t where the ray intersects an object
		# a, b, c and d is refering to the values of an quadratic equation
		a = np.sum(np.square(rv))
		b = np.sum(2 * campos * rv - 2 * sphere[i][1] * rv)
		c = np.sum(np.square(campos) - 2 * campos * sphere[i][1] + np.square(sphere[i][1])) - sphere[i][0] ** 2
		d = b * b - 4 * a * c

		# Doing the quadratic equation
		if d < 0:
			continue
		elif d == 0:
			t = -b / (2 * a)
		else:
			t = (np.sqrt(d) - b) / (2 * a)
			t2 = (-np.sqrt(d) - b) / (2 * a)
			if t2 < t and t2 >= 0:
				t = t2

		# Finding the values of for the closest intersection
		if t == None or t < 0:
			continue
		elif tmin == None:
			tmin = t
			imin = i
		elif t < tmin:
			tmin = t
			imin = i

	# Raytracing for the reflection of the spheres
	if tmin != None:
		rgb1 = sphere[imin][2]
		pos = campos + tmin * rv

		# Finding the direction vector for the new ray
		nv = pos - sphere[imin][1]
		rv2 = np.dot(-2 * np.dot(rv, nv) / np.linalg.norm(nv), nv) + rv

		# Nudging the intersecting position away from the object so it won't intersect with it self
		pos += 0.01 * nv / sphere[imin][0]

		rgb2 = intersection(sphere, ground, sky, light, rv2, pos, bounces - 1)
		rgb = rgb1 * (1 - sphere[imin][3]) + rgb2 * sphere[imin][3]

		# Finding the shadow
		lightv = light - sphere[imin][1]
		shade = 2 - np.dot(lightv, nv) / (np.linalg.norm(lightv) * sphere[imin][0])
		if shade > 2:
			shade = 2


	# Intersections with the ground
	elif rv[2] < 0:
		# Finding the colour of the pixel that intersects the ray
		t = (ground[0] - campos[2]) / rv[2]
		w = len(ground[2])
		scale = int(w / ground[3])
		pos = campos + t * rv
		imgpix = (pos[:2] * scale).astype(int) % w
		rgb1 = ground[2][imgpix[0]][imgpix[1]]

		# Reflectning the ray off the ground
		nv = np.array([0, 0, 1])
		rv2 = np.dot(-2 * np.dot(rv, nv) / np.sum(np.square(nv)), nv) + rv
		pos[2] += 0.01

		rgb2 = intersection(sphere, ground, sky, light, rv2, pos, bounces - 1)
		rgb = rgb1 * (1 - ground[1]) + rgb2 * ground[1]

	# Intersections with the sky
	else:
		normhor = np.linalg.norm(rv[:2])
		hor = np.arctan2(rv[0], rv[1]) * 0.159
		if normhor == 0:
			vertical = 2
		else:
			vertical = np.arctan(rv[2] / normhor) * 1.557
		if abs(rv[0]) < abs(rv[2]) and abs(rv[1]) < abs(rv[2]): #vertical > 1:
			t = 256 / rv[2]
			x = int(t * rv[0] + 256)
			y = int(t * rv[1] + 256)
			rgb = sky[1][y][x]
			return rgb
			"""
			side = int(hor * 4)
			y = int((vertical - 1) * 256)
			x = int(hor % 1) * y * 2
			if side == 0:
				rgb = sky[1][x][y]
			elif side == 1:
				rgb = sky[1][511 - y][x]
			elif side == 2:
				rgb = sky[1][511 - x][y]
			else:
				rgb = sky[1][y][511 - x]
			return rgb #np.array([75, 117, 175])"""

		else:
			if rv[1] == 0:
				x = 0
			else:
				x = int(hor * 2048)
			y = 256 - int(vertical * 256)
			return sky[0][y][x]

	# Finding out if it is in a shadow
	if shade != 2:
		rvlight = light - pos
		for i in range(len(sphere)):
			a = np.sum(np.square(rvlight))
			b = np.sum(2 * pos * rvlight - 2 * sphere[i][1] * rvlight)
			c = np.sum(np.square(pos) - 2 * pos * sphere[i][1] + np.square(sphere[i][1])) - sphere[i][0] ** 2
			d = b * b - 4 * a * c

			if d >= 0:
				if d == 0:
					t = -b / (2 * a)
					t2 = t
				else:
					t = (np.sqrt(d) - b) / (2 * a)
					t2 = (-np.sqrt(d) - b) / (2 * a)
				if t > 0 or t2 > 0:
					shade = 2
					break
	return (rgb / shade)

# Renders the entire image
def raytrace(sphere, ground, sky, light, resolution, lens, campos, camdir, bounces):
	out = np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8)

	tan = np.tan(lens * 0.0175)
	slope = -2 * tan / resolution

	i = 0
	di = 100 / (resolution[0] * resolution[1])
	for x in prange(resolution[0]):
		for y in prange(resolution[1]):
			point = slope * np.array([x, y]) + tan
			rv = np.dot(np.array([1, point[1], point[0]]), camdir)
			out[x][y] = intersection(sphere, ground, sky, light, rv, campos, bounces)
			i += di
			print(f"{int(i)} %")
	return out

if __name__ == "__main__":
	createScene()

	resolution = [128, 128]
	lens = np.array([45, 45])
	campos = np.array([-4, 0, 2])
	camdirection = rotate3d(0, 40, 0)
	bounces = 3

	start = time.time()
	img = raytrace(sphere, ground, sky, light, resolution, lens, campos, camdirection, bounces)
	end = time.time()
	print(end - start)

	plt.imshow(img)
	plt.show()