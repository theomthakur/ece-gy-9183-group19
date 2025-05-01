# Installation Steps

## Pre-requsites

- Python 3.11
- Git
- Docker

### 1. Clone the repository
Begin by cloning the project repository and navigating into the project directory:
```
https://github.com/theomthakur/ece-gy-9183-group19.git
cd ece-gy-9183-group19/
```

### 2. Set up a virtual environment
Create and activate a virtual environment for the project:
```
# Create virtual environment
python3.11 -m venv .venv --prompt env-MLOps

# Activate virtual environment
# For Unix/macOS:
source .venv/bin/activate

# For Windows:
# .venv\Scripts\activate
```

### 3. Install Project Dependencies
Install the required dependencies:
```
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```