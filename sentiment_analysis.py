from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

texts = ["I love this product!", "This is the worst experience ever."]
labels = [1, 0]  # 1=positive, 0=negative

encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="tf")

dataset = tf.data.Dataset.from_tensor_slices((
    dict(encodings),
    labels
)).batch(2)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(dataset, epochs=2)

# Predict
inputs = tokenizer("I hate waiting in lines.", return_tensors="tf")
outputs = model(inputs)
probs = tf.nn.softmax(outputs.logits, axis=-1)
print("Negative prob:", probs[0][0].numpy(), "Positive prob:", probs[0][1].numpy())
