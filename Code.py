import requests, math, io, os
from PIL import Image
import joblib
import numpy as np

flag = 0
scale_model = joblib.load("scale_model.pkl")

# === Function to Compress Image In-Memory ===
def compress_image_in_memory(image_path, max_size_mb=2):
    """
    Compress the image only if it exceeds max_size_mb.
    Returns a file-like object (BytesIO) for upload.
    """
    global flag
    flag = 1

    max_bytes = max_size_mb * 1024 * 1024
    original_size = os.path.getsize(image_path)

    # If already under the limit, just open and return file object
    if original_size <= max_bytes:
        flag = 0
        return open(image_path, "rb")

    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    quality = 95
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)

    # Compress iteratively if still too large
    while buffer.getbuffer().nbytes > max_bytes and quality > 10:
        quality -= 5
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)

    buffer.seek(0)
    return buffer


# === Function to Compute Measurements ===
def compute_cow_measurements(pred, scale_factor=None):
    """
    Compute cow body length and heart girth from adjusted keypoints.
    """
    global flag
    if flag ==1 :
        kps = {kp["class"]: (kp["x"]/2, kp["y"]/2) for kp in pred["keypoints"]}
    else :
        kps = {kp["class"]: (kp["x"], kp["y"]) for kp in pred["keypoints"]}
    flag = 0
    def euclidean(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Body length = Shoulder1 <-> Tail-base
    pixel_body_length = euclidean(kps["Shoulder1"], kps["Tail-base"])
    
    # Heart girth = Shoulder2 <-> Withers    
    pixel_heart_dia = euclidean(kps["Shoulder2"], kps["withers"])

    # Bounding box features
    bbox_w = pred["width"]
    bbox_h = pred["height"]

    # Prepare features for ML model
    feature_vector = np.array([[
        pixel_body_length,
        bbox_w,
        bbox_h,
        kps["Shoulder1"][0], kps["Shoulder1"][1],
        kps["Tail-base"][0], kps["Tail-base"][1],
        kps["withers"][0],   kps["withers"][1],
        kps["Shoulder2"][0], kps["Shoulder2"][1]
    ]])

    # Predict scale factor (inches/pixel)
    scale_factor = scale_model.predict(feature_vector)[0]

    # Convert pixel distances → inches
    real_body_length = pixel_body_length * scale_factor
    real_heart_dia = pixel_heart_dia * scale_factor

    def ramanujan_girth(D1, D2):
        a = D1 / 2.0
        b = D2 / 2.0
        term = 3*(a + b) - ((3*a + b)*(a + 3*b))**0.5
        return math.pi * term

    heart_girth = ramanujan_girth(real_heart_dia, 1.08*real_heart_dia)

    weight = ((real_body_length) * (heart_girth ** 2)) / 660

    return {
        "pixel_body_length": pixel_body_length,
        "scale_factor": scale_factor,
        "body_length_in": real_body_length,
        "heart_girth_in": heart_girth,
        "weight": weight,
    }


# === Roboflow API Inference with Cropping and Resizing ===
def run_inference(image_path, api_key, model_id, scale_factor=None, fixed_size=(512, 512)):
    original_image = Image.open(image_path).convert("RGB")

    # Reopen the file for uploading after compressing it
    compressed_image_file = compress_image_in_memory(image_path)

    response = requests.post(
        f"https://detect.roboflow.com/{model_id}?api_key={api_key}&format=json",
        files={"file": ("image.jpg", compressed_image_file, "image/jpeg")}
    )

    result = response.json()

    # print("DEBUG RESULT:", result)
    
    if "predictions" not in result or len(result["predictions"]) == 0:
        return {"error": "No cow detected in image!"}

    pred = result["predictions"][0]
    measures = compute_cow_measurements(pred, scale_factor)

    return measures


# === MAIN ===
if __name__ == "__main__":
    # Take dynamic input
    image_path = input("Enter image path: ").strip()
    API_KEY = "wUHnmbcsjGFnk0qkplqR"
    MODEL_ID = "cattle-body-measurements-wtegw/1"
    
    # Run inference
    measures = run_inference(image_path, API_KEY, MODEL_ID, scale_factor= None)
    print("Measurements:", measures)
