# Code from group 1
# author Fredrik Gustafsson

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
p4 = __import__("Project 4")


learning = 0.001
adam = Adam(lr=learning, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
epochs = 100
batch_size = 32

def myGetModel(data):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=data.input_dim))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(data.num_classes, activation='softmax'))

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def myFitModel(model, data):
    callback = [EarlyStopping(monitor='val_loss', patience=2),
                ModelCheckpoint(filepath='weights-best.hdf5', monitor='val_loss', save_best_only=True)]
    fitted_cnn = model.fit(data.x_train, data.y_train, batch_size=batch_size, epochs=epochs, callbacks=callback,
                               verbose=1,
                               validation_data=(data.x_valid, data.y_valid))
    fitted_cnn = load_model('weights-best.hdf5')

    return fitted_cnn

# Test model
p4.runImageClassification(myGetModel,myFitModel)