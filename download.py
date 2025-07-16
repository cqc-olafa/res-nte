import kagglehub

path = kagglehub.dataset_download("deadskull7/fer2013", path="/.data")

print("Path to dataset files:", path)