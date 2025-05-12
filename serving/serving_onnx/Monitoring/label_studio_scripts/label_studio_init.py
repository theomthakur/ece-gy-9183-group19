import requests
import os
import time

# Configuration from environment variables
LABEL_STUDIO_URL = os.getenv("LABEL_STUDIO_URL", "http://label-studio:8080")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_USER_TOKEN", "ab9927067c51ff279d340d7321e4890dc2841c4a")
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "PROSadmin19")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "PROSadmin19")
BUCKET_NAME = "production"

# Labeling configuration for bounding box annotations
LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" maxWidth="800px"/>
  <RectangleLabels name="label" toName="image">
    <Label value="0" background="#FFA39E"/>
    <Label value="1" background="#D4380D"/>
    <Label value="2" background="#FFC069"/>
    <Label value="3" background="#AD8B00"/>
    <Label value="4" background="#D3F261"/>
    <Label value="5" background="#389E0D"/>
    <Label value="6" background="#5CDBD3"/>
    <Label value="7" background="#096DD9"/>
    <Label value="8" background="#ADC6FF"/>
    <Label value="9" background="#9254DE"/>
    <Label value="10" background="#F759AB"/>
    <Label value="11" background="#FFA39E"/>
    <Label value="12" background="#D4380D"/>
    <Label value="13" background="#FFC069"/>
  </RectangleLabels>
</View>
"""

def create_label_studio_project():
    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
    
    # Check if project already exists
    response = requests.get(f"{LABEL_STUDIO_URL}/api/projects", headers=headers)
    if response.status_code == 200:
        projects = response.json()
        for project in projects:
            if project["title"] == "Chest X-Ray Detection":
                print(f"Project 'Chest X-Ray Detection' already exists with ID {project['id']}")
                return project["id"]
    
    # Create new project
    project_config = {
        "title": "Chest X-Ray Detection",
        "description": "Annotate chest X-ray images for object detection",
        "label_config": LABEL_CONFIG
    }
    
    response = requests.post(f"{LABEL_STUDIO_URL}/api/projects", json=project_config, headers=headers)
    if response.status_code == 201:
        project_id = response.json()["id"]
        print(f"Created new project: Chest X-Ray Detection (ID {project_id})")
        
        # Configure MinIO storage
        storage_config = {
            "title": "MinIO Storage",
            "type": "s3",
            "bucket": BUCKET_NAME,
            "aws_access_key_id": MINIO_ACCESS_KEY,
            "aws_secret_access_key": MINIO_SECRET_KEY,
            "s3_endpoint": MINIO_URL,
            "recursive_scan": True,
            "treat_every_object_as_file": True
        }
        
        response = requests.post(
            f"{LABEL_STUDIO_URL}/api/storages/s3",
            json=storage_config,
            headers=headers,
            params={"project": project_id}
        )
        if response.status_code == 201:
            print("MinIO storage configured successfully")
            # Sync storage
            response = requests.post(
                f"{LABEL_STUDIO_URL}/api/storages/s3/{response.json()['id']}/sync",
                headers=headers
            )
            if response.status_code == 200:
                print("MinIO storage synced successfully")
            else:
                print(f"Failed to sync storage: {response.text}")
        else:
            print(f"Failed to configure MinIO storage: {response.text}")
        
        return project_id
    else:
        raise Exception(f"Failed to create project: {response.text}")

if __name__ == "__main__":
    # Wait for Label Studio to be fully up
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{LABEL_STUDIO_URL}/health", timeout=5)
            if response.status_code == 200:
                print("Label Studio is up")
                break
        except requests.exceptions.RequestException:
            print(f"Waiting for Label Studio... Attempt {attempt + 1}/{max_attempts}")
            time.sleep(5)
    else:
        raise Exception("Label Studio failed to start within the expected time")
    
    # Create the project
    create_label_studio_project()