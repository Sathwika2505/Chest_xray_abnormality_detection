import io
from io import BytesIO
import pandas as pd
import boto3

def read_csv_from_s3(bucket_name, csv_file_key):
    try:
        print("Accessing CSV file from S3")
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=csv_file_key)
        data = response['Body'].read()
        df = pd.read_csv(io.BytesIO(data))
        print("CSV file loaded into DataFrame")
        print(f"Total number of entries in CSV file: {df['image_id'].nunique()}")
        return df
    except Exception as e:
        print(f"Error downloading CSV file from S3: {e}")
        return None

bucket_name = 'deeplearning-mlops-demo'
csv_file_key = 'train.csv'    
df = read_csv_from_s3(bucket_name, csv_file_key)
print(df)
