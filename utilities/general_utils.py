"""Python files that contain some general utility functions."""
import os
import json


# Helper functions to load json file
def load_json(json_path: str):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

#Helper function to save the plot.
def save_plot(fig, folder_name):
    """
    Saves the plot with the name of the folder/file.
    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        folder_name (str): The folder name to use in the file name.
    """
    output_dir = os.path.join("/home/s.bhat/Coding/amenity_detection/", "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{folder_name}.png")
    fig.savefig(output_path)
    print(f"Plot saved to {output_path}")