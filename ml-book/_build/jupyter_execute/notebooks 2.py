#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/TrickyTroll/ML-intro/blob/Tricky/OCR.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Following a tutorial to learn about classification
# 
# https://www.pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/
# 
# https://www.pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/

# In[1]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# ## Loading the minst data

# In[167]:


((train_data, train_labels), (test_data, test_labels)) = mnist.load_data()
num_data = np.vstack([train_data, test_data])
num_labels = np.hstack([train_labels, test_labels])


# In[168]:


train_data.shape


# In[169]:


len(train_labels)


# In[170]:


train_labels


# In[171]:


test_data.shape


# ## Preprocessing

# In[172]:


plt.figure()
plt.imshow(train_data[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[173]:


train_labels[0]


# In[174]:


train_images = train_data / 255.0
test_images = test_data / 255.0


# In[175]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()


# In[176]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])


# In[177]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[178]:


model.fit(train_images, train_labels, epochs=10)


# In[179]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# In[180]:


model.save("/content/drive/My Drive/OCR_Models/test", save_format="h5")


# In[181]:


model.summary()


# In[182]:


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


# In[183]:


predictions = probability_model.predict(test_images)


# In[184]:


predictions[0]


# In[185]:


np.argmax(predictions[0])


# In[186]:


test_labels[0]


# In[187]:


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[188]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# In[189]:


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# In[190]:


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# In[191]:


# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)


# In[192]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[193]:


predictions_single = probability_model.predict(img)

print(predictions_single)


# In[194]:


plot_value_array(1, predictions_single[0], [0,1,2,3,4,5,6,7,8,9])
_ = plt.xticks(range(10), [0,1,2,3,4,5,6,7,8,9], rotation=45) #The labels are wrong


# In[195]:


np.argmax(predictions_single[0])


# ## Testing it on my own images

# In[283]:


from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename


# In[284]:


from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))


# In[285]:


import numpy as np
import imutils
import cv2
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt


# In[286]:


image = cv2.imread("photo.jpg")
resized_image = cv2.resize(image, (28, 28))
gray = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
(thresh, blackAndWhiteImage) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
imagem = cv2.bitwise_not(blackAndWhiteImage)
cv2_imshow(imagem)


# In[287]:


imagem.shape


# In[288]:


img2 = (np.expand_dims(imagem,0))

print(img2.shape)


# In[289]:


plt.figure()
plt.imshow(imagem)
plt.colorbar()
plt.grid(False)
plt.show()


# In[290]:


predictions_array = probability_model.predict(img2)

print(predictions_array)


# In[291]:


predicted_label = np.argmax(predictions_array)


# In[292]:


predicted_label


# In[293]:


plot_value_array(1, predictions_array[0], [0,1,2,3,4,5,6,7,8,9])
_ = plt.xticks(range(10), [0,1,2,3,4,5,6,7,8,9], rotation=45) #The labels are wrong


# In[293]:




