# Cifar-10 Classifier Using CNN

Access and run the notebook on [kaggle](https://www.kaggle.com/mumarwaseem/cnn-cifar10-classifier)

## Dataset

Dataset Link: https://www.cs.toronto.edu/~kriz/cifar.html

## Accuracy acheived after training

#### For 120 Epochs
- Training Acc: `93.43%`
- Validation Acc: `90.01%`
- Test Acc: `89.39%`

## Model Architecture

```python
def get_cnn_model(activation = "relu"):
    model = Sequential()
    model.add(Conv2D(96, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(96, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(192, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(192, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(192, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(128, activation=activation, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    return model

```



<img src="https://github.com/Umar-Waseem/cnn-cifar10-classifier/blob/main/images/accuracy_ss.png"  />

## Accuracy, Loss and Precision

<img src="https://github.com/Umar-Waseem/cnn-cifar10-classifier/blob/main/images/accuracy_plot.png" />

## Model Prediciton

<img src="https://github.com/Umar-Waseem/cnn-cifar10-classifier/blob/main/images/prediction_ss.png" />
