from ultralytics import YOLO
import os


input_folder = r"C:\Users\lamsa\Desktop\nfl project\nflTrimmedAll22"
output_folder = r"C:\Users\lamsa\Desktop\nfl project\nflOutput"


os.makedirs(output_folder, exist_ok=True)

model = YOLO("yolov8n.pt")


for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")
        
        model.predict(
            source=input_path,
            save=True,
            project=output_folder,
            name=filename.replace(".mp4", ""),
            classes=[0], 
            conf=0.3,     
            show=False
        )

print("All done! Check nflOutput folder.")