import os
import keras
from keras import layers
import pandas as pd
import numpy as np

def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input([input_size], name="embedding_inputs"),
            layers.Dense(input_size, activation="relu", name="hidden"),
            layers.Dense(num_classes, activation="softmax", name="output_probs"),
        ]
    )

df_train = pd.read_parquet("day2/store/df_train.parquet")
df_test = pd.read_parquet("day2/store/df_test.parquet")

embeddings_test_path = 'day2/store/embeddings_test.npz'
if os.path.exists(embeddings_test_path):
    print("Loading embeddings test ...")
    data = np.load(embeddings_test_path)
    x_test = data["x"]
    y_test = data["y"]

embeddings_train_path = 'day2/store/embeddings_train.npz'
if os.path.exists(embeddings_train_path):
    print("Loading embeddings train ...")
    data = np.load(embeddings_train_path)
    x_train = data["x"]
    y_train = data["y"]

# Derive the embedding size from observing the data. The embedding size can also be specified
# with the `output_dimensionality` parameter to `embed_content` if you need to reduce it.

embedding_size = len(x_train[0])
print(embedding_size)


classifier = build_classification_model(
    embedding_size, len(df_train["Class Name"].unique())
)

print(classifier.summary())

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

NUM_EPOCHS = 20
BATCH_SIZE = 32

# Specify that it's OK to stop early if accuracy stabilises.
early_stop = keras.callbacks.EarlyStopping(monitor="accuracy", patience=3)

# Train the model for the desired number of epochs.
history = classifier.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
)
