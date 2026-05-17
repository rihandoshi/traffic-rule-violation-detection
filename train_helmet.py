import shutil
from ultralytics import YOLO

def main():
    # Fine tune yolo for helmet detection
    
    print("Loading base YOLOv8n model for fine-tuning...")
    model = YOLO("models/yolov8s.pt")
    
    print("Starting training process...")
    model.train(
        data="helmet_dataset.yaml",  
        epochs=25,                   
        imgsz=224,                   
        batch=16,                   
        device="0",                 
        project="training_runs",  
        name="helmet_model" 
    )
    
    print("Training complete! Running validation...")
    metrics = model.val()
    print("Validation metrics:", metrics)

    try:
        shutil.copy("training_runs/helmet_model/weights/best.pt", "models/helmet_yolov8s.pt")
        print("Saved best model to models/helmet_yolov8s.pt")
    except Exception as e:
        print("Could not copy weights (this is normal if you don't have the dataset):", e)

if __name__ == "__main__":
    main()
