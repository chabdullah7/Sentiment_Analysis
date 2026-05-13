import unittest
from fastapi.testclient import TestClient

# import your FastAPI app
from fastapi_app.app import app


class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    
    # Home page test
    def test_home_page(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        # check HTML contains expected title
        self.assertIn(
            b"Sentiment Analysis",
            response.content
        )

    
    # Predict test
    def test_predict_page(self):
        response = self.client.post(
            "/predict",
            data={"text": "I love this!"}
        )

        self.assertEqual(response.status_code, 200)

        # check output contains result
        self.assertTrue(
            b"0" in response.content or b"1" in response.content,
            "Response should contain prediction 0 or 1"
        )


if __name__ == "__main__":
    unittest.main()