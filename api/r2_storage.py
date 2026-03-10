"""
Cloudflare R2 Storage for SpeakFit Audio Files

Provides S3-compatible storage for audio files with zero egress fees.
"""

import os
import boto3
import logging
from botocore.config import Config
from typing import Optional, BinaryIO, Union
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# R2 Configuration - loaded from environment or config
R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY", "6be392c28d74794055de085e5a1a1484")
R2_SECRET_KEY = os.environ.get("R2_SECRET_KEY", "ba57bc4d15f329968ee39f48ba0785263906034c7d290731a162e54f754fd605")
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID", "35a7141c3204f967cb6eb831a75f71fc")
R2_BUCKET = os.environ.get("R2_BUCKET", "speakfit-audio")
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Public access URL (if configured with public bucket or custom domain)
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL", None)


class R2Storage:
    """Cloudflare R2 storage client for audio files."""
    
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='auto'
        )
        self.bucket = R2_BUCKET
    
    def _generate_key(self, user_id: int, speech_id: int, filename: str, format: str = "opus") -> str:
        """Generate a unique storage key for an audio file."""
        # Structure: audio/{user_id}/{year}/{month}/{speech_id}_{hash}.{format}
        now = datetime.utcnow()
        file_hash = hashlib.md5(f"{speech_id}:{filename}:{now.isoformat()}".encode()).hexdigest()[:8]
        return f"audio/{user_id}/{now.year}/{now.month:02d}/{speech_id}_{file_hash}.{format}"
    
    def upload(
        self, 
        data: Union[bytes, BinaryIO], 
        user_id: int, 
        speech_id: int, 
        filename: str,
        format: str = "opus",
        content_type: str = "audio/opus"
    ) -> dict:
        """
        Upload audio file to R2.
        
        Returns:
            dict with 'key', 'url', 'size_bytes'
        """
        key = self._generate_key(user_id, speech_id, filename, format)
        
        # Get size
        if isinstance(data, bytes):
            size_bytes = len(data)
        else:
            data.seek(0, 2)  # Seek to end
            size_bytes = data.tell()
            data.seek(0)  # Reset
        
        # Upload
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={
                'user_id': str(user_id),
                'speech_id': str(speech_id),
                'original_filename': filename,
            }
        )
        
        logger.info(f"Uploaded {key} ({size_bytes} bytes) to R2")
        
        return {
            'key': key,
            'url': self.get_url(key),
            'size_bytes': size_bytes,
        }
    
    def download(self, key: str) -> bytes:
        """Download audio file from R2."""
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()
    
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Get a URL for accessing the file.
        
        If R2_PUBLIC_URL is configured, returns a public URL.
        Otherwise, generates a presigned URL valid for expires_in seconds.
        """
        if R2_PUBLIC_URL:
            return f"{R2_PUBLIC_URL.rstrip('/')}/{key}"
        
        # Generate presigned URL
        return self.client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expires_in
        )
    
    def delete(self, key: str) -> bool:
        """Delete a file from R2."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
            logger.info(f"Deleted {key} from R2")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if a file exists in R2."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False
    
    def list_user_audio(self, user_id: int, limit: int = 100) -> list:
        """List audio files for a user."""
        prefix = f"audio/{user_id}/"
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
            MaxKeys=limit
        )
        return [obj['Key'] for obj in response.get('Contents', [])]


# Singleton instance
_r2_storage: Optional[R2Storage] = None

def get_r2_storage() -> R2Storage:
    """Get the R2 storage singleton."""
    global _r2_storage
    if _r2_storage is None:
        _r2_storage = R2Storage()
    return _r2_storage
