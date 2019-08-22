import io
import json
import os
import boto3


class Storage():
    def get_file(self):
        pass

    def save_file(self):
        pass


class LocalStorage(Storage):
    def get_file(self):
        pass

    def save_file(self):
        pass


class S3Storage(Storage):

    def __init__(self):
        # Get file from S3
        self.access_key_id = os.getenv("ACCESS_KEY_ID", None)
        self.bucket = os.getenv("BUCKET", None)
        self.region = os.getenv("REGION", None)
        self.secret_access_key = os.getenv("SECRET_ACCESS_KEY", None)

        # If cloud.gov, needs to override it with that
        vcap_services = os.getenv("VCAP_SERVICES", None)
        if vcap_services:
            # This is somewhat hardcoded to the 1st item in S3 service
            vcap_services_json = json.loads(vcap_services)
            s3_cred = vcap_services_json["s3"][0]["credentials"]
            self.access_key_id = s3_cred["access_key_id"]
            self.bucket = s3_cred["bucket"]
            self.region = s3_cred["region"]
            self.secret_access_key = s3_cred["secret_access_key"]

        self.s3 = boto3.client("s3", aws_access_key_id=self.access_key_id,
                               aws_secret_access_key=self.secret_access_key, region_name=self.region)


class S3File():
    def __init__(self, s3_storage, filename):
        self.s3_storage = s3_storage
        self.data = io.BytesIO()
        s3_storage.s3.download_fileobj(self.s3_storage.bucket, filename, self.data)
        self.data.seek(0)

    def __enter__(self):
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.data.close()
