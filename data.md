# Data download guide

## Downloading the data

The data used for the paper come in four sets:

1. `train`: The train data, including predictions on the train data, as multi-channel images for easy viewing.
1. `train_single_channel_images`: The train data, including predictions on the train data, as single-channel images. The filenames of these images identify the transmitted light modality or fluorescent label.
1. `test`: The test data, including predictions on the test data, as multi-channel images for easy viewing.
1. `test_single_channel_images`: The test data, including predictions on the test data, as single-channel images. The filenames of these images identify the transmitted light modality or fluorescent label.

These images may be viewed in a browser or downloaded in bulk using [`gsutil`](https://cloud.google.com/storage/docs/gsutil).
The browser links are:
[`train`](https://storage.googleapis.com/in-silico-labeling/paper_data/train),
[`train_single_channel_images`](https://storage.googleapis.com/in-silico-labeling/paper_data/train_single_channel_images),
[`test`](https://storage.googleapis.com/in-silico-labeling/paper_data/test), and
[`test_single_channel_images`](https://storage.googleapis.com/in-silico-labeling/paper_data/test_single_channel_images).

To download in bulk, first install [`gsutil`](https://cloud.google.com/storage/docs/gsutil) and then execute command like this:

    gsutil -m cp -r gs://in-silico-labeling/paper_data/train
    gsutil -m cp -r gs://in-silico-labeling/paper_data/train_single_channel_images
    gsutil -m cp -r gs://in-silico-labeling/paper_data/test
    gsutil -m cp -r gs://in-silico-labeling/paper_data/test_single_channel_images

## Understanding the data

TODO
