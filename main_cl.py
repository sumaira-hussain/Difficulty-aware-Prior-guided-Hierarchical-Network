# from clmodel import *
from DHN import *
from data import *
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard
from time import time
import matplotlib.pyplot as plt
from tensorflow import keras

# config= tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.5  #limit gpu use to 50%
# #config.gpu_options.visible_device_list="0"
# session=tf.Session(config=config)
# set_session(tf.Session(config=config))

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# sess = tf.Session (config=tf.ConfigProto(gpu_options=gpu_options))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
testGene = trainGenerator(2,'/home/SHussain/U-net-sumi/BUS_QHSP/train/','images','labels',data_gen_args,save_to_dir =  None)
validation_generator=trainGenerator(2,'/home/SHussain/U-net-sumi/BUS_QHSP/validation/','images','labels',data_gen_args,save_to_dir = None)

# model = unetcl()
model=dhn()
# keras.utils.plot
# _model(model, "UNet.png")
#es=EarlyStopping(monitor='loss', clpatience=10, verbose=1)
rop=ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=5, min_lr=0.001, verbose=1)
# rop=ReduceLROnPlateau(monitor='loss',factor=0.2, patience=5, min_lr=0.001, verbose=1)
# mc=ModelCheckpoint('/home/SHussain/ULS/UNet/bhaye/L5/model-epoch.{epoch:02d}-{acc:.2f}.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True,  period= 150)
mc=ModelCheckpoint('/home/SHussain/DHN/DHN_keras_L3_L4/FL/model-epoch.{epoch:02d}-{val_acc:.2f}.h5', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, period= 150)
tb=TensorBoard(log_dir='dhn-log/{}'.format(time()))
#path for tensorboard can be replaced from logs/{} tobdgintu /home/SHussain/U-net-sumi/
#monitor progress in tensorboard by running following command in separate terminal
#tensorboard --logdir=logs/          logs/ is pointing to log root directory
results=model.fit_generator(testGene, validation_data=validation_generator, validation_steps= 5, verbose=1, steps_per_epoch=300, epochs=1200, callbacks=[rop,mc,tb])

# results=model.fit_generator(testGene, verbose=1, steps_per_epoch=300, epochs=1200, callbacks=[rop,mc,tb])
# to save multiple weights file
# checkpoint_path="weights.{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period= 50)
# results=model.fit_generator(testGene, validation_data=validation_generator, validation_steps= 5, verbose=1, steps_per_epoch=300, epochs=700, callbacks=[rop,checkpoint,tb])
# model.save('model.h5')


#epochs=1000
#plt.style.use("ggplot")
plt.figure()
#plt.plot(np.arange(0, epochs), results.history["loss"], label="train_loss")
#plt.plot(np.arange(0, epochs), results.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, epochs), results.history["acc"], label="train_acc")
#plt.plot(np.arange(0, epochs), results.history["val_acc"], label="val_acc")
plt.plot(results.history["loss"], label="train_loss")
# plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(results.history["acc"], label="train_acc")
# plt.plot(results.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.title("DHN")
plt.show(block=True)
plt.interactive(False)
plt.savefig("plot-1200.jpg")
