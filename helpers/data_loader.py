from typing import Tuple, Dict, Union, List

import pandas as pd
import boto3


class DataLoader:
    def __init__(self, use_s3: bool = False) -> None:
        self.use_s3 = use_s3
        self.s3_client = boto3.client('s3')

    def load_csv(self, file_path):
        if self.use_s3:
            return self.s3_file_loader(file_path)
        else:
            return self.local_file_loader(file_path)

    def local_file_loader(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")

    def s3_file_loader(self, file_path: str) -> pd.DataFrame:
        try:
            bucket_name, key = self.parse_s3_file_path(file_path)
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(response['Body'])
            return df
        except Exception as e:
            print(f"Error: Failed to load CSV from S3 - {e}")

    @staticmethod
    def parse_s3_file_path(file_path: str) -> Tuple[str, str]:
        # if S3 file path is like s3://bucket-name/path/to/file.csv
        parts: List[str] = file_path.replace('s3://', '').split('/')
        bucket_name: str = parts[0]
        key: str = '/'.join(parts[1:])
        return bucket_name, key
