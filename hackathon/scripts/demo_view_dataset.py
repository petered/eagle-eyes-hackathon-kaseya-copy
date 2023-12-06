from hackathon.data_utils.data_loading import AnnotatedImageDataLoader, get_default_dataset_folder
from hackathon.ui_utils.ui_view_dataset import open_annotation_database_viewer

"""
This demo opens up a GUI to view the dataset, including annotations.
"""


if __name__ == '__main__':
    open_annotation_database_viewer(AnnotatedImageDataLoader.from_folder(get_default_dataset_folder()))
