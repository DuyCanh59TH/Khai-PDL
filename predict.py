# make a prediction for a new image.
from matplotlib import pyplot
from matplotlib.image import imread
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	# image = imread(img)
	# pyplot.imshow(image)
	return img

# load an image and predict the class
def run_example():
	# show the image
	filename = 'test_dog_example_3.png'
	image = imread(filename)
	# plot raw pixel data
	pyplot.imshow(image)

	# load the image
	img = load_image(filename)
	# load model
	model = load_model('final_model.h5')
	# predict the class
	result = model.predict(img)
	pyplot.show()
	if result[0] > 0.8 and result[0] <= 1.0 :
		print("CON CHÓ")
	else:
		print("CON MÈO")

# entry point, run the example
run_example()