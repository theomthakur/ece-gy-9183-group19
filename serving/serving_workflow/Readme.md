# Clone the repo with read-only token
git clone https://rigvedrs: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@github.com/rigvedrs/serving-monitoring.git

# Pull the model from git lfs
sudo apt-get install git-lfs -y
git lfs install
cd serving-monitoring
git lfs pull
cd ..

# Run docker compose
docker compose -f serving-monitoring/serving/serving_workflow/docker-compose.yml up -d