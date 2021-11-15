# from clmodel import *
from DHN import *
# from multi_layer_output_unetcl import *
# from unetcl_layer_4 import*
# from model import *
from data import *
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
import keract
from keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# model=unetcl()
model = dhn()
model.load_weights('/home/SHussain/DHN/DHN_keras_L3_L4/FL/model-epoch.1200-0.96.h5')
testGene = testGenerator('/home/SHussain/U-net-sumi/all_imgs(QHSP)/images/')
results = model.predict_generator(testGene, 186, verbose=1)
saveResult('/home/SHussain/DHN/DHN_keras_L3_L4/FL/1200e/',results)

# for i in range(2):
#     # for i,img in enumerate([img1, img2]):
#     img = "/home/SHussain/U-net-sumi/all-imgs/%d" % (i + 162) + ".png"
#     image_add = image.load_img(img, target_size=(256, 256), color_mode='grayscale', interpolation='nearest')
#     img = image.img_to_array(image_add)
#     print(img.shape)
#     img= np.reshape(img,(1,)+img.shape)
#     img=img.astype('float32')
#     img/=255
#     activations=keract.get_activations(model, img)
#     display=keract.display_activations(activations, cmap='gray')


# img = "/home/SHussain/U-net-sumi/all-imgs/1.png"
# image_add = image.load_img(img, target_size=(256, 256), color_mode='grayscale', interpolation='nearest')
# img = image.img_to_array(image_add)
# img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
# img=preprocess_input(img)
# # img_ten /=255.
# activations=keract.get_activations(model,img)
# keract.display_heatmaps(activations, img_tensor)

# ixs=[2,5,9]
# outputs=[model.layers[i].output for  i in ixs]
# model2= Model(inputs=model.inputs, outputs=outputs)
# results2=model2.predict_generator(testGene, 186, verbose=1)

# layer_name=[]
# for layer in model.layers[1:7]:
#     layer_name.append(layer.name) # store name of layers in layer_name

# images_per_row=16
#
# for layer_name, layer_activation in zip(layer_name, results): #display feature aps
#     n_features = layer_activation.shape[-1] # number of features in feature map
#     size=layer_activation.shape[1] #the feature map has shape (1, size, size, n_features)
#     n_cols=n_features // images_per_row #tiles the activation channels in this matrix
#     display_grid=np.zeros((size*n_cols, images_per_row*size))
#     for col in range(n_cols): #tiles each figure into big horizonta; grid
#         for row in range(images_per_row):
#             channel_image=layer_activation[0,:,:,col*images_per_row+row]
#             channel_image -= channel_image.mean() # post-processes the feature to make it visually palatable
#             channel_image /= channel_image.std()
#             channel_image *= 64
#             channel_image +=128
#             channel_image = np.clip(channel_image,0, 255).astype('uint8')
#             display_grid[col*size: (col+1)*size, #displays the grid
#                          row*size: (row+1)*size ]=channel_image
#             scale = 1. / size
#             plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
#             plt.title(layer_name)
#             plt.grid(False)
#             plt.imshow(display_grid, aspect='auto', cmap='viridis')
#     plt.show()






# square=8
# for fmap in results:
#     ix=1
#     for _ in range(square):
#         for _ in range(square):
#             ax=plt.subplot(square, square, ix)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             plt.title(ix)
#             plt.imshow(results[ix-1,:,:,0],cmap='gray')
#             ix+=1
#     plt.show()