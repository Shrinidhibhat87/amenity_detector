"""Python script that creates a dataloder for the SUN RGB-D dataset. We are currently focusing only on 2D images."""
import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.data import Dataset
from PIL import Image
from utilities.general_utils import load_json, save_plot


class SUNRGBDDataset(Dataset):
    def __init__(self, root_dir, debug=False):
        """
        Args:
            root_dir (str): Root directory of the SUN RGB-D dataset.
        """
        self.root_dir = root_dir
        self.debug = debug
        self.data = self._load_data()

    def _load_data(self):
        """
        Recursively loads the dataset and stores relevant information.

        Returns:
            list: A list of dictionaries containing folder name, image path, annotation path, and scene.txt path.
        """
        dataset = []
        count = 0
        for data_folder in os.listdir(self.root_dir):
            data_folder_path = os.path.join(self.root_dir, data_folder)
            if not os.path.isdir(data_folder_path):
                continue

            image_dir = os.path.join(data_folder_path, "image")
            annotation_dir = os.path.join(data_folder_path, "annotation2Dfinal")
            scene_path = os.path.join(data_folder_path, "scene.txt")

            if os.path.exists(image_dir) and os.path.exists(annotation_dir) and os.path.exists(scene_path):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
                if not image_files:
                    continue
                image_path = os.path.join(image_dir, image_files[0])
                annotation_path = os.path.join(annotation_dir, "index.json")

                if os.path.exists(image_path) and os.path.exists(annotation_path):
                    dataset.append({
                        "folder_name": data_folder,
                        "image_path": image_path,
                        "annotation_path": annotation_path,
                        "scene_path": scene_path.strip()
                    })
                    count += 1

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data point.
        Returns:
            dict: A dictionary containing the image, annotations, and scene information.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data[idx]
        image = self._load_image(item["image_path"])
        annotations = load_json(item["annotation_path"])
        with open(item["scene_path"], 'r') as f:
            scene = f.read().strip()

        return {
            "folder_name": item["folder_name"],
            "image": image,
            "annotations": annotations,
            "scene": scene
        }

    def _load_image(self, image_path):
        """
        Loads an image from the given path.
        Args:
            image_path (str): Path to the image file.
        Returns:
            PIL.Image: Loaded image.
        """
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def get_segments_2d(self, annotations):
        """
        Returns the segments in 2D.
        """
        segments = []
        labels = []
        for polygon in annotations['frames'][0]['polygon']:
            x = polygon["x"]
            y = polygon["y"]
            obj_pointer = polygon["object"]
            points = np.transpose(np.array([x, y], np.int32))
            segments.append(points)
            labels.append(annotations['objects'][obj_pointer]["name"])
        return labels, segments

    def show_annotations(self, idx):
        """
        Displays the image and prints its annotations and class.
        Args:
            idx (int): Index of the data point.
        """
        data = self[idx]
        image = data["image"]
        annotations = data["annotations"]
        labels, segments = self.get_segments_2d(annotations)
        
        if self.debug:
            image_np = np.array(image)
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(image_np)

            for label, segment in zip(labels, segments):
                polygon = patches.Polygon(segment, closed=True, edgecolor='red', fill=False, linewidth=2)
                ax.add_patch(polygon)
                x, y = segment[0]
                ax.text(x, y, label, color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            plt.axis('off')
            plt.show()
            save_plot(fig, data["folder_name"])


# To test the dataset
if __name__ == "__main__":
    dataset_path = "/home/s.bhat/Datasets/SUNRGB2DATA"
    dataset = SUNRGBDDataset(dataset_path, debug=True)

    # Test the dataset
    print(f"Total data points: {len(dataset)}")
    if len(dataset) > 0:
        dataset.show_annotations(0)
