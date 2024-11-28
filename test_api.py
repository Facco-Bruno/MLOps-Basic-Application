import unittest
import requests

class TestAPIPrediction(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8000"

    def test_health_check(self):
        response = requests.get(f"{self.BASE_URL}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())

    def test_prediction(self):
        payload = {
            "MedInc": 8.5,
            "HouseAge": 10,
            "AveRooms": 6,
            "AveBedrms": 1.5,
            "Population": 300,
            "AveOccup": 3.2,
            "Latitude": 37.5,
            "Longitude": -120.5
        }
        response = requests.post(f"{self.BASE_URL}/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        self.assertIsInstance(response.json()["prediction"], float)

    def test_feedback(self):
        payload = {
            "MedInc": 8.5,
            "HouseAge": 10,
            "AveRooms": 6,
            "AveBedrms": 1.5,
            "Population": 300,
            "AveOccup": 3.2,
            "Latitude": 37.5,
            "Longitude": -120.5,
            "actual_value": 450.0
        }
        response = requests.post(f"{self.BASE_URL}/feedback", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("success", response.json()["status"])

if __name__ == "__main__":
    unittest.main()
