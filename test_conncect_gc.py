from google.cloud import storage
from pathlib import Path

def list_files_in_bucket(bucket_name):
    """Lists all the blobs in the bucket."""
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # List blobs in the bucket
    blobs = bucket.list_blobs()
    print(f"Files in bucket {bucket_name}:")
    for blob in blobs:
        print(blob.name)

def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    # Initialize a storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(source_blob_name)

    print(f"Blob {source_blob_name}.")

# Example usage
if __name__ == "__main__":
    # Replace 'your-bucket-name' with your actual bucket name
    BUCKET_NAME = "mlops_data_reddit_bucket"
    raw_data_path: str = Path("data/raw")

    # List files in the bucket
    list_files_in_bucket(BUCKET_NAME)

    # Download a specific file from the bucket
    SOURCE_BLOB_NAME = raw_data_path / "data.csv"
    download_blob(BUCKET_NAME, SOURCE_BLOB_NAME)
