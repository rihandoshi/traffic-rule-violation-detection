from roboflow import Roboflow
rf = Roboflow(api_key="GXkwPcAfj5qrQvmorDXT")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(1)
dataset = version.download("yolov8")
print(dataset.location)
    