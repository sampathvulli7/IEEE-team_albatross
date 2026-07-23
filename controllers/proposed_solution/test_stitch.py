import cv2
import sys

print("Opening video...")
cap = cv2.VideoCapture('../../recordings/small_world_flyover.mp4')
frames = []
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # sample 1 frame per second (30fps)
    if count % 30 == 0:
        frames.append(frame)
    count += 1
cap.release()

print(f"Extracted {len(frames)} frames. Stitching...")
stitcher = cv2.Stitcher_create()
status, pano = stitcher.stitch(frames)
if status == cv2.Stitcher_OK:
    print("Stitching successful!")
    cv2.imwrite('pano.png', pano)
else:
    print(f"Stitching failed with status: {status}")
