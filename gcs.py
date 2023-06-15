import os
import time

from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'cert/cert-google-cloud-storage.json'

class GCStorage:
    def __init__(self):
        self.storage_client = storage.Client('farm-genius')
        self.bucket_name = "farm-genius.appspot.com"

    def upload_file(self, file):
        bucket = self.storage_client.bucket(self.bucket_name)
        file_path = 'disease/' + str(time.time()) + "-" + file.filename
        blob = bucket.blob(file_path)
        blob.upload_from_file(file.file, content_type='image/jpeg')

        return f'https://storage.cloud.google.com/{self.bucket_name}/{file_path}'
