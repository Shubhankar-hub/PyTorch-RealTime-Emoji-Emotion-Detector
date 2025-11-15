import cv2
import torch
import numpy as np
from model import EmotionCNN
from torchvision import transforms
import os

EMOJI_MAP = {
    0: "angry.png",
    1: "disgusted.png",
    2: "fearful.png",
    3: "happy.png",
    4: "neutral.png",
    5: "sad.png",
    6: "surprised.png"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN().to(device)
model.load_state_dict(torch.load("emotion.pth", map_location=device))
model.eval()

emoji_dir = "assets"
emojis = {}

for idx, file in EMOJI_MAP.items():
    path = os.path.join(emoji_dir, file)
    if os.path.exists(path):
        emojis[idx] = cv2.imread(path, cv2.IMREAD_UNCHANGED)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        img = transform(roi).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.argmax(model(img)).item()

        if pred in emojis:
            emoji = emojis[pred]
            emoji = cv2.resize(emoji, (w, w))

            ex = x
            ey = y - w if y - w > 0 else 0

            alpha = emoji[:, :, 3] / 255.0 if emoji.shape[2] == 4 else None
            emoji_rgb = emoji[:, :, :3]

            for c in range(3):
                if alpha is not None:
                    frame[ey:ey+w, ex:ex+w, c] = (
                        alpha * emoji_rgb[:, :, c] +
                        (1 - alpha) * frame[ey:ey+w, ex:ex+w, c]
                    )
                else:
                    frame[ey:ey+w, ex:ex+w, c] = emoji_rgb[:, :, c]

    cv2.imshow("Emojify", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print("Loaded emojis:", emojis.keys())
cam.release()
cv2.destroyAllWindows()

