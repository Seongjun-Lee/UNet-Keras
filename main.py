from model import *
from data import *
from unet2p import *
import tf2onnx
from datetime import datetime
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(tf.config.experimental.list_physical_devices('GPU'))

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
#myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
myGene = trainGenerator(2,'data/Data2/train','image','label',data_gen_args,save_to_dir = None, num_class=7, flag_multi_class=False)

epoch = 100
step = 300

model = unet()
#model = unet2p()
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model_checkpoint = ModelCheckpoint('unet_eye.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=step,epochs=epoch,callbacks=[model_checkpoint])

#testGene = testGenerator("data/membrane/test")
testGene = testGenerator("data/Data2/test", num_image=2, flag_multi_class=False, as_gray=False)
results = model.predict_generator(testGene,4,verbose=1)
saveResult("data/Data2/test",results, num_class = 7, flag_multi_class = False)

#_INPUT = model.input.name
#_OUTPUT = model.output.name
spec = (tf.TensorSpec((None, 256, 256, 3), tf.float32, name="input"),)
model_proto, external = tf2onnx.convert.from_keras(model, input_signature=spec, output_path="UNET " + datetime.now().strftime('%Y-%m-%d %H_%M') + " step-" + str(step) + " epoch-" + str(epoch) + ".onnx")
model_proto, external = tf2onnx.convert.from_keras(model, input_signature=spec, output_path="UNET " + datetime.now().strftime('%Y-%m-%d %H_%M') + " step-" + str(step) + " epoch-" + str(epoch) + "_nchw.onnx", inputs_as_nchw=['input'])

model2 = unet2p()
model2_checkpoint = ModelCheckpoint('unet2p_eye.hdf5', monitor='loss',verbose=1, save_best_only=True)
model2.fit_generator(myGene,steps_per_epoch=step,epochs=epoch,callbacks=[model_checkpoint])

testGene2 = testGenerator("data/Data2/test2", num_image=2, flag_multi_class=False, as_gray=False)
results2 = model2.predict_generator(testGene2, 4, verbose=1)
saveResult("data/Data2/test2",results2, num_class = 7, flag_multi_class = False)

model_proto2, external2 = tf2onnx.convert.from_keras(model2, input_signature=spec, output_path="UNET2P " + datetime.now().strftime('%Y-%m-%d %H_%M') + " step-" + str(step) + " epoch-" + str(epoch) + ".onnx")
model_proto2, external2 = tf2onnx.convert.from_keras(model2, input_signature=spec, output_path="UNET2P " + datetime.now().strftime('%Y-%m-%d %H_%M') + " step-" + str(step) + " epoch-" + str(epoch) + "_nchw.onnx", inputs_as_nchw=['input'])