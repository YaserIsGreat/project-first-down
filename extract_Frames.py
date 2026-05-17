import os
import subprocess

input_folder = r""
output_folder = r"" 

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        clip_name = filename.replace(".mp4", "")
        input_path = os.path.join(input_folder, filename)
        output_pattern = os.path.join(output_folder, f"{clip_name}_%04d.png")
        
        command = [
            "ffmpeg",
            "-i", input_path,
            "-vf", "fps=2",
            output_pattern
        ]
        
        print(f"Extracting frames from: {filename}")
        subprocess.run(command)

print("All frames extracted!")
