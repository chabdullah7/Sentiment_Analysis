import unittest
from fastapi.testclient import TestClient

from fastapi_app.app import app


class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Sentiment", response.content)

    def test_predict(self):
        response = self.client.post(
            "/predict",
            data={"text": "I love this product"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b"0" in response.content or b"1" in response.content
        )


if __name__ == "__main__":
    unittest.main()