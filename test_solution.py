from solution import TrafficViolationDetector
from pathlib import Path
import json

def test():
    print("Initializing TrafficViolationDetector...")
    detector = TrafficViolationDetector(model_dir="./models")
    
    image_path = "test_images/9.jpeg"
    print(f"\nRunning prediction on {image_path}...")
    
    result = detector.predict(image_path)
    
    print("\nResult:")
    print(json.dumps(result, indent=2))
    
    print("\nChecking format...")
    assert "violations" in result, "Missing 'violations' key"
    for v in result["violations"]:
        assert "num_riders" in v, "Missing 'num_riders'"
        assert "helmet_violations" in v, "Missing 'helmet_violations'"
        assert "license_plate" in v, "Missing 'license_plate'"
        assert isinstance(v["num_riders"], int), "num_riders must be int"
        assert isinstance(v["helmet_violations"], int), "helmet_violations must be int"
        assert isinstance(v["license_plate"], str), "license_plate must be string"
    print("Format check passed! Matches guidelines perfectly.")

if __name__ == "__main__":
    test()
