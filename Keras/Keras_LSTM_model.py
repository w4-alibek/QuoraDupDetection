import tensorflow as tf
import glove_embedding as embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Dropout
from keras.layers.core import Lambda
from keras.layers.merge import concatenate, add, multiply
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers

tf.flags.DEFINE_integer("lstm_out_dimantion", 100,
                       "Where the train data with computed features are stored.")
tf.flags.DEFINE_float("recurrent_dropout", 0.2,
                       "Where the word_to_id is stored.")
tf.flags.DEFINE_string("raw_train_data", None,
                       "Where the raw train data is stored.")
tf.flags.DEFINE_string("glove_path", None,
                       "Where the raw train data is stored.")
tf.flags.DEFINE_string("raw_test_data", None,
                       "Where the raw train data is stored.")
tf.flags.DEFINE_integer("batch_size", 100,
                       "Where the raw train data is stored.")
tf.flags.DEFINE_integer("max_sequence_length", 1000,
                       "Where the raw train data is stored.")
tf.flags.DEFINE_integer("number_epochs", 15,
                       "Where the raw train data is stored.")
tf.flags.DEFINE_integer("glove_dimension", 50,
                       "Where the raw train data is stored.")

FLAGS = tf.flags.FLAGS

LSTM_OUT_D = 100
RNN_DROPOUT = 0.2
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 1000
BATCH_SIZE = 100

embedding_layer, labels, question1s, question2s, tokenizer = embedding\
    .process_data(FLAGS.glove_path,
                  FLAGS.raw_train_data,
                  FLAGS.glove_dimension,
                  FLAGS.max_sequence_length)

lstm_layer = LSTM(FLAGS.lstm_out_dimantion, recurrent_dropout=FLAGS.recurrent_dropout)

sequence_1_input = Input(shape=(FLAGS.max_sequence_length,), dtype="int32")
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(FLAGS.max_sequence_length,), dtype="int32")
embedded_sequences_2 = embedding_layer(sequence_2_input)
y1 = lstm_layer(embedded_sequences_2)



sequences1 = tokenizer.texts_to_sequences(question1s)
sequences2 = tokenizer.texts_to_sequences(question2s)

data_1 = pad_sequences(sequences1, maxlen=FLAGS.max_sequence_length)
data_2 = pad_sequences(sequences2, maxlen=FLAGS.max_sequence_length)

print('Shape of data tensor:', data_1.shape)

# split the data into a training set and a validation set
num_validation_samples = int(VALIDATION_SPLIT * data_1.shape[0])

q1_train = data_1[:-num_validation_samples]
q2_train = data_2[:-num_validation_samples]
labels_train = labels[:-num_validation_samples]

q1_validation = data_1[-num_validation_samples:]
q2_validation = data_2[-num_validation_samples:]
labels_validation = labels[-num_validation_samples:]

addition = add([x1, y1])
minus_y1 = Lambda(lambda x: -x)(y1)
merged = add([x1, minus_y1])
merged = multiply([merged, merged])
merged = concatenate([merged, addition])
merged = Dropout(0.4)(merged)

merged = BatchNormalization()(merged)
merged = GaussianNoise(0.1)(merged)

out = Dense(1, activation="sigmoid")(merged)

model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=out)
optimizer_nadam = optimizers.Nadam(lr=0.002)

model.compile(loss="binary_crossentropy", optimizer=optimizer_nadam, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor="val_loss", patience=5)
best_model_path = "best_model.h5"
model_checkpoint = ModelCheckpoint(best_model_path,
                                   save_best_only=True,
                                   save_weights_only=True)

history = model.fit([q1_train, q2_train],
                labels_train,
                validation_data=([q1_validation, q2_validation], labels_validation),
                epochs=15,
                batch_size=FLAGS.batch_size,
                shuffle=True,
                callbacks=[early_stopping, model_checkpoint],
                verbose=1)

# evaluate model
train_score = model.evaluate([q1_train, q2_train], labels_train, verbose=True)
print("Training:  ", train_score)
print("--------------------")
print("First 5 samples validation:", history.history["val_acc"][0:5])
print("First 5 samples training:", history.history["acc"][0:5])
print("--------------------")
print("Last 5 samples validation:", history.history["val_acc"][-5:])
print("Last 5 samples training:", history.history["acc"][-5:])

