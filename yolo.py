from ultralytics import YOLO
model = YOLO('models\best.pt')
results = model.predict('F:\JNL\Projects\Football Analysis\input_videos',save = True)
print(results[0])
print('=================================')
for box in results[0].boxes:
    print(box)  