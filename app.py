from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import ResNetWithDualFC
import io

model_base_name = "resnet18"
model = ResNetWithDualFC()
PATH = f"weights/best_{model_base_name}.pth"
model.load_state_dict(torch.load(PATH))
ALLOWED_FILE_EXTENSIONS = {"jpg", "jpeg", "png"}


def transform_image(image_bytes):
    inference_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return inference_transform(image).unsqueeze(0)


def predict_for_image(image_tensor):
    with torch.no_grad():
        model.eval()
        pred = model(image_tensor)
        return pred


def verify_file_extension(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_FILE_EXTENSIONS
    )


app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "Welcome to the image classification service. Please send a POST request to /predict with an image file."


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if file is None or file.filename == "":
            return jsonify({"error": "File not found"})
        if not verify_file_extension(file.filename):
            return jsonify({"error": "File extension not supported"})
        try:
            img_bytes = file.read()
            image_tensor = transform_image(img_bytes)
            prediction = predict_for_image(image_tensor).squeeze().tolist()
            result = {
                "pred_hood": prediction[0],
                "pred_backdoor_left": prediction[1],
            }
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})
    return jsonify({"error": "Invalid request method. Only POST is allowed."})


if __name__ == "__main__":
    app.run(debug=True)
