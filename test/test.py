import requests
import os
import pandas as pd

current_dir = os.path.dirname(__file__)

test_images = [
    x
    for x in os.listdir(os.path.join(current_dir, "test_sampled_imgs"))
    if x.endswith(".jpg")
]
labels_df = pd.read_csv("test_sampled.csv").set_index("filename")


def send_request(image_path: str) -> str:
    resp = requests.post(
        "http://localhost:5000/predict", files={"file": open(image_path, "rb")}
    )
    return resp.text


def main():
    for image in test_images:
        print("\nPredicting for image:", image)
        print(send_request(os.path.join("test_sampled_imgs", image)))
        print("Actual label:")
        print(labels_df.loc[image])


if __name__ == "__main__":
    main()
