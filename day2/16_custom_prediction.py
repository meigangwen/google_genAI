import os
import keras
from keras import layers
import pandas as pd
import numpy as np
from google import genai
from google.genai import types


GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

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

classifier.evaluate(x=x_test, y=y_test, return_dict=True)

def embed_fn(text: str) -> list[float]:
    # You will be performing classification, so set task_type accordingly.
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="classification",
        ),
    )

    return response.embeddings[0].values

def make_prediction(text: str) -> list[float]:
    """Infer categories from the provided text."""
    # Remember that the model takes embeddings as input, so calculate them first.
    embedded = embed_fn(new_text)

    # And recall that the input must be batched, so here they are wrapped as a
    # list to provide a batch of 1.
    inp = np.array([embedded])

    # And un-batched here.
    [result] = classifier.predict(inp)
    return result

new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""

result = make_prediction(new_text)

for idx, category in enumerate(df_test["Class Name"].cat.categories):
    print(f"{category}: {result[idx] * 100:0.2f}%")
