import boto3
import pandas as pd
import logging
import os
from io import StringIO
from dotenv import load_dotenv

from src.logger import logging

# Load .env file
load_dotenv()


class s3_operations:
    def __init__(self, bucket_name=None, region_name=None):
        """
        Initialize S3 connection using environment variables.
        """

        # Read from .env if not passed
        self.bucket_name = bucket_name or os.getenv("AWS_BUCKET_NAME")
        region_name = region_name or os.getenv("AWS_REGION", "us-east-1")

        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not all([self.bucket_name, aws_access_key, aws_secret_key]):
            raise ValueError("Missing AWS credentials in environment variables")

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )

        logging.info("S3 connection initialized successfully")

    def fetch_file_from_s3(self, file_key):
        """
        Fetch CSV file from S3 and return DataFrame
        """
        try:
            logging.info(f"Fetching {file_key} from bucket {self.bucket_name}")

            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )

            df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

            logging.info(f"Loaded {len(df)} records from S3")
            return df

        except Exception as e:
            logging.exception(f"Failed to fetch from S3: {e}")
            return None