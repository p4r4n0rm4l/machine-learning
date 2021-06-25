import tensorflow_datasets as tfds


# (test_dataset, train_dataset, validation_dataset), dataset_info = tfds.load('voc', split=["test", "train", "validation"], with_info=True)

# print(test_dataset)

ds, ds_info = tfds.load('DOTA', split="train", with_info=True)
fig = tfds.show_examples(ds, ds_info)
df = tfds.as_dataframe(ds.take(5), ds_info)
