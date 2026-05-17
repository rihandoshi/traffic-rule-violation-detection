from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="./models")

image_path = "test_images/7.png"

results = detector.predict(image_path)

print(results)
