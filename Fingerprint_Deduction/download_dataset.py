import kagglehub
import shutil
import os

print("Downloading SOCOFing dataset...")

path = kagglehub.dataset_download("ruizgara/socofing")

print("Downloaded to:")
print(path)

destination = "dataset"

if os.path.exists(destination):
    print("Dataset folder already exists.")
else:
    shutil.copytree(path, destination)

print("Dataset copied successfully.")