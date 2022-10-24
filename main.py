import os

from utils import DataGenerator, read_annotation_lines
from models import Yolov4
import tensorflow as tf
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback


def train():
    run = neptune.init(
        project="tywithings/aircraft-spot-and-box",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZGIzODg1NS1mMDhmLTRjNTgtOTEzOC00NmM4N2NmNmFiNjgifQ==",
    )  # your credentials

    neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

    train_lines, val_lines = read_annotation_lines('./dataset/horizontal/txt/anno.txt',
                                                   test_size=0.2)
    FOLDER_PATH = './dataset/images'
    class_name_path = './dataset/horizontal/class_names/classes.txt'
    data_gen_train = DataGenerator(train_lines,
                                   class_name_path,
                                   FOLDER_PATH)
    data_gen_val = DataGenerator(val_lines,
                                 class_name_path,
                                 FOLDER_PATH)

    dir4saving = 'path2checkpoint/checkpoints'
    os.makedirs(dir4saving, exist_ok=True)

    logdir = 'path4logdir/logs'
    os.makedirs(logdir, exist_ok=True)

    name4saving = 'epoch_{epoch:02d}-val_loss-{val_loss:.4f}.hdf5'

    filepath = os.path.join(dir4saving, name4saving)

    rLrCallBack = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                    factor=0.1,
                                                    patience=5,
                                                    verbose=1)

    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                             histogram_freq=0,
                                             write_graph=False,
                                             write_images=False)

    mcCallBack_loss = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=False,
                                                      mode='auto',
                                                      period=1)

    esCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               verbose=1,
                                               patience=10)

    model = Yolov4(weight_path=None,
                   class_name_path=class_name_path)

    # epochs=10000, TODO: FIX EPOCHS FOR REAL TRAINING
    model.fit(data_gen_train,
              initial_epoch=0,
              epochs=5,
              val_data_gen=data_gen_val,
              callbacks=[rLrCallBack,
                         tbCallBack,
                         mcCallBack_loss,
                         esCallBack,
                         neptune_cbk]
              )

    run.stop()

    # model.fit(data_gen_train,
    #           initial_epoch=0,
    #           epochs=10000,
    #           val_data_gen=data_gen_val,
    #           callbacks=[])


def test():
    model = Yolov4(weight_path='yolov4.weights',
                   class_name_path='./dataset/horizontal/class_names/classes.txt')
    model.predict('./dataset/images/1.jpg')
    model.predict('./dataset/images/99.jpg')
    model.predict('./dataset/images/876.jpg')


def quicktest():
    model = Yolov4(weight_path='yolov4.weights',
                   class_name_path='class_names/coco_classes.txt')
    model.predict('quicktest/coldbeer.jpg')
    model.predict('quicktest/pretoriastreet.jpg')
    model.predict('quicktest/redhead.jpeg')


if __name__ == '__main__':
    # quicktest()
    test()