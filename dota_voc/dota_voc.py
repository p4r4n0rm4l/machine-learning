"""PASCAL DOTA dataset."""

import os
import xml.etree.ElementTree

import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_VOC_CITATION = """\
@misc{ding2021object,
title={Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges},
author={Jian Ding and Nan Xue and Gui-Song Xia and Xiang Bai and Wen Yang and Micheal Ying Yang and Serge Belongie and Jiebo Luo and Mihai Datcu and Marcello Pelillo and Liangpei Zhang},
year={2021},
eprint={2102.12219},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
"""

_VOC_DESCRIPTION = """
This dataset contains the data from the PASCAL Visual Object Classes Challenge,
corresponding to the Classification and Detection competitions.
In the Classification competition, the goal is to predict the set of labels
contained in the image, while in the Detection competition the goal is to
predict the bounding box and label of each individual object.
WARNING: As per the official dataset, the test set of VOC2012 does not contain
annotations.
"""

_VOC_CONFIG_DESCRIPTION = """
Created by Crowley
"""
_VOC_URL = "Oh Mr.Crowley"
# Original site, it is down very often.
# _VOC_DATA_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/"
# Data mirror:

_VOC_LABELS = (
    "sv",
    "lv",
    "ship",
    "plane",
)


def _get_example_objects(annon_filepath):
  """Function to get all the objects from the annotation XML file."""
  with tf.io.gfile.GFile(annon_filepath, "r") as f:
    root = xml.etree.ElementTree.parse(f).getroot()

    # Disable pytype to avoid attribute-error due to find returning
    # Optional[Element]
    # pytype: disable=attribute-error
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    for obj in root.findall("object"):
      # Get object's label name.
      label = obj.find("name").text.lower()
      # Get object's bounding box
      bndbox = obj.find("bndbox")
      xmax = float(bndbox.find("xmax").text)
      xmin = float(bndbox.find("xmin").text)
      ymax = float(bndbox.find("ymax").text)
      ymin = float(bndbox.find("ymin").text)
      yield {
          "label": label,
          "bbox": tfds.features.BBox(
              ymin / height, xmin / width, ymax / height, xmax / width),
      }


class Dota_VocConfig(tfds.core.BuilderConfig):

    """BuilderConfig for Voc."""

    def __init__(self, year=None, filenames=None, has_test_annotations=True, **kwargs):
        self.year = '2020'
        self.filenames = filenames
        self.has_test_annotations = has_test_annotations
        super(Dota_VocConfig, self).__init__(
            name='DOTA',
            version=tfds.core.Version("1.0.0"),
            **kwargs)


class Dota_Voc(tfds.core.GeneratorBasedBuilder):
    """DOTA dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Register into https://example.org/login to get the data. Place the `data.zip`
    file in the `manual_dir/`.
    """

    BUILDER_CONFIGS = [
        Dota_VocConfig(
            description=_VOC_CONFIG_DESCRIPTION.format(
                num_images=4771, num_objects=24640),
            filenames={
                "dota_train": "dota_train.zip",
                "dota_test": "dota_test.zip",
            },
        ),
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_VOC_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "image/filename": tfds.features.Text(),
                "objects": tfds.features.Sequence({
                    "label": tfds.features.ClassLabel(names=_VOC_LABELS),
                    "bbox": tfds.features.BBoxFeature(),
                }),
                "labels": tfds.features.Sequence(
                    tfds.features.ClassLabel(names=_VOC_LABELS)),
            }),
            # homepage=_VOC_URL.format(year=self.builder_config.year),
            # citation=_VOC_CITATION.format(year=self.builder_config.year),
        )

    def _split_generators(self, dl_manager):

        archive_path = dl_manager.manual_dir / "data.tar"
        extracted_path = dl_manager.extract(archive_path)

        return{
            'train' : self._generate_examples(
                images_path=extracted_path / 'train_imgs',
                labels_path=extracted_path / 'train_labels',
            ),
            'test' : self._generate_examples(
                images_path=extracted_path / 'test_imgs',
                labels_path=extracted_path / 'test_labels',
            ),
        }

    def _generate_examples(self, images_path, labels_path):

        for labels_file in tf.io.gfile.listdir(labels_path):
            image_id = labels_file[:-4]

            image_path = os.path.join(images_path, image_id + '.jpg')
            label_path = os.path.join(labels_path, labels_file)

            example = self._generate_example(image_id, image_path, label_path)
            yield image_id, example


    def _generate_example(self, image_id, image_filepath, annon_filepath):

        objects = list(_get_example_objects(annon_filepath))
        labels = sorted(set(obj["label"] for obj in objects))
        
        return {
            "image": image_filepath,
            "image/filename": image_id + ".jpg",
            "objects": objects,
            "labels": labels,
        }