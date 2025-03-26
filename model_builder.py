from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_model(input_size=(128, 128, 1)):
    inputs = Input((input_size))
    X = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(seed=0))(inputs)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D((3, 3))(X)

    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Flatten()(X)

    dense_1 = Dense(256, activation='relu')(X)
    dropout_1 = Dropout(0.4)(dense_1)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(dropout_1)

    dense_2 = Dense(256, activation='relu')(X)
    dense_3 = Dense(128, activation='relu')(dense_2)
    dropout_2 = Dropout(0.4)(dense_3)
    age_output = Dense(1, activation='relu', name='age_output')(dropout_2)

    model = Model(inputs=inputs, outputs=[gender_output, age_output])
    model.compile(optimizer='adam', loss=['binary_crossentropy', 'mse'], metrics=[['accuracy'], ['mae']])
    return model