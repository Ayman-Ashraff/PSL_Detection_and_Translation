from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from pytorchvideo.models.resnet import create_resnet
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os

app = FastAPI()
print("heyyy")

# ================== Recreate model architecture ==================
def build_model():
    model = create_resnet(
        input_channel=3,        # RGB input
        model_depth=50,         # ResNet-50 backbone
        model_num_class=40,     # Updated for 40 total categories (based on the 8 main categories with 5 subcategories each)
        norm=nn.BatchNorm3d,
    )
    # Replace final projection layer with 40 output classes (based on categories)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, 40)  # 40 output classes
    return model

# Load model
MODEL_PATH = r"c:\Users\Micro\Desktop\fyp_model_ayma\model.pth"  # Change to the correct model path
model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True), strict=False)
model.eval()

# ================== Transformation for input video frames ==================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Match training normalization
])

# ================== Frame extraction function (16 frames) ==================
def extract_frames_from_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        raise ValueError("No frames in video")

    interval = max(1, total_frames // num_frames)
    frames = []

    for i in range(num_frames):
        frame_id = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        tensor = transform(image)
        frames.append(tensor)

    cap.release()

    # Pad with last frame if less than 16
    while len(frames) < num_frames:
        frames.append(frames[-1])

    video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    return video_tensor


# ================== Categories Mapping ==================
categories = {
    0: "Bear", 1: "Bird", 2: "Cat", 3: "Cow", 4: "Deer", 5: "Dog", 6: "Fish", 7: "Horse", 8: "Lion",
    9: "Coat", 10: "Dress", 11: "Gloves", 12: "Hat", 13: "Jacket", 14: "Pants", 15: "Scarf", 16: "Shirt", 
    17: "Shoes", 18: "Skirt", 19: "Socks", 20: "Sweater",
    21: "Angry", 22: "Bored", 23: "Confused", 24: "Excited", 25: "Fear", 26: "Happy", 27: "Lonely", 28: "Love", 
    29: "Nervous", 30: "Sad", 31: "Surprised", 32: "Tired",
    33: "Bicycle", 34: "Boat", 35: "Bus", 36: "Car", 37: "Helicopter", 38: "Motorcycle", 39: "Rickshaw"
    , 40: "Monkey", 41: "Rabbit" , 42: "Sheep", 43: "Tiger"
}
    

# ================== FastAPI endpoint ==================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded video temporarily
        temp_video_path = r"C:\Users\Micro\Desktop\fyp_model_ayma\temp_video.mp4"
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract frames and process
        input_tensor = extract_frames_from_video(temp_video_path)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_index = predicted.item()

        # Map class index to category
        predicted_category = categories.get(class_index, "Unknown")

        # Clean up
        os.remove(temp_video_path)

        return JSONResponse(content={"predicted_class": predicted_category})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})