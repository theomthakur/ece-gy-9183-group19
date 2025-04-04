# ECE-GY-9183 - GROUP 19 - Chest X-ray Abnormality Detection System for Radiology Departments

## Value Proposition

Our system is an AI-assisted chest X-ray analysis system that uses YOLOv11-L to both classify pathologies AND localize abnormalities with bounding boxes. The system works with the VinDr-CXR dataset, which contains annotations from multiple radiologists, enabling robust evaluation of AI assistance benefits.

Our system enhances radiologist workflow in hospital settings by:

1. Reducing missed pathologies: The system identifies and highlights potential abnormalities that might be overlooked during routine reads, particularly during high-volume or fatigue periods.

2. Improving reading efficiency: By pre-highlighting regions of concern, the system allows radiologists to focus their attention on suspicious areas, potentially reducing reading time.

3. Serving as a "second opinion": The visual indicators of potential abnormalities provide radiologists with an automated consultation that can confirm findings or prompt reconsideration.

The system maintains the radiologist as the ultimate decision-maker while providing assistive insights that improve diagnostic accuracy and workflow efficiency. Our evaluation will specifically measure how the AI assistance reduces pathology miss rates compared to unassisted radiologist performance.

- **Status Quo (Non-ML):** In hospital radiology departments, radiologists manually interpret 50-100 CXRs daily, spending an average of a few minutes per image. This process relies entirely on human expertise, with no computational assistance. Fatigue and high workloads lead to missed pathologies, leading to an increase in error rates.
- **Methods for addressing over-dependence :** For addressing the biases that might occur when doctors may get too dependent on AI to make their decisions, we will make use of auditing and training. There will also be warninigs reminding the doctors about this in the middle. 


- **Business Metrics:** Our chest X-ray AI assistant aims to improve two key business metrics:
	- **Radiologist efficiency** - Help radiologists spend less time reviewing each CXR
	- **Diagnostic accuracy** - Help radiologists miss fewer pathologies

	- Following our manager Fraida Fund's (:) guidance, while we cannot implement a full business evaluation in our academic setting, we can evaluate "radiologists missing fewer pathologies" using the VinDr-CXR dataset:
	
	 - We'll use images in our test set that were labeled by specific radiologists, compute their accuracy relative to consensus labels, and then simulate AI-assisted accuracy in two scenarios:
	
		- Assuming radiologists follow all AI suggestions
		- Assuming radiologists only follow AI input when they initially would have labeled the image as "no finding" but the AI detected a pathology
	
	 - This approach allows us to estimate potential improvements in pathology detection without requiring clinical deployment.
	
	 - For "reduced interpretation time," we would define a measurement plan for future implementation in a production environment, comparing baseline reading times against AI-assisted reading times.
	 - We recognize that in practice, studies have shown radiologists sometimes ignore AI assistance and sometimes trust it too much even when incorrect. Our evaluation considers whether our system provides genuine improvement over the status quo of human-only interpretation.

## Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->


| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                | System Diagram, Planning |                                    |
| Prashant Shihora                   | Continuous X pipeline | [Commits](https://github.com/theomthakur/ece-gy-9183-group19/commits/main/README.md?author=Mario928)                                   |
| Rigved Shirvalkar                   | Model Serving and Monitoring | [Commits](https://github.com/theomthakur/ece-gy-9183-group19/commits/main/README.md?author=rigvedrs)                                    |
| Om Thakur               | Data Pipeline | [Commits](https://github.com/theomthakur/ece-gy-9183-group19/commits/main/README.md?author=theomthakur)                                   |
| Sohith Bandari | Model Training    | [Commits](https://github.com/theomthakur/ece-gy-9183-group19/commits/main/README.md?author=Billa-Man)                                   |



### System diagram

![](https://github.com/theomthakur/ece-gy-9183-group19/blob/main/systemdesign.png)

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. 
Must include: all the hardware, all the containers/software platforms, all the models, 
all the data. -->


<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

### Summary of outside materials

|                                      | How it was created | Conditions of use |
|--------------------------------------|--------------------|-------------------|
| VinBigData Chest X-Ray dataset | [Paper Link](https://arxiv.org/pdf/2012.15029) | See Usage Notes section in the [research paper](https://arxiv.org/pdf/2012.15029) |
| YOLOv11-L | [Documentation](https://docs.ultralytics.com/models/yolo11/) | [AGPL 3.0 Software License](https://www.ultralytics.com/legal/agpl-3-0-software-license) |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `compute_liqid` | 4x A100 GPUs                     | Required for training models           |
| `gpu_p100`     | 1                        | Model Inferencing  |
| Floating IPs    | 2 | 1 for model serving API, 1 for monitoring |
| Object Storage | 1 volume - 250GB | Storing the dataset |
| Persistent Volume | 1 volume - 10GB | Model checkpoints, Retrained Models & Logs |


## Detailed design plan

## UNIT 3: DevOps

- **Infrastructure-as-Code (IaC):** We will define all infrastructure in code and store it in Git for version control and reproducibility. Terraform will be used to manage infrastructure on Chameleon, replacing manual configuration. This will include compute resources for YOLOv11-L training and inference, networking, and storage for the VinDr-CXR dataset.

- **Automated Setup:** We will use Ansible to automatically set up software on our systems, including CUDA drivers, PyTorch, and other dependencies for YOLOv11-L. All configurations will be stored in Git to ensure consistency across environments.


- **Cloud-Native Design:** The project will follow cloud-native principles. Infrastructure will be immutable, meaning updates will be made by changing the code in Git and deploying new infrastructure. The system will use a microservices architecture, separating components like data preprocessing, model inference, and result storage into independent services that communicate through APIs. All services will run in Docker containers for consistent deployment.

- **Staged Deployment:** We will implement three deployment environments: staging, canary, and production. Staging will be used for initial testing with sample X-ray images. Canary deployment will test changes on a small percentage of simulated X-ray traffic. Production will handle full-scale X-ray image analysis.

- **Continuous Training:** Our pipeline will automatically train new YOLOv11-L models when enough new data is collected. Models will be evaluated based on accuracy and false positive/negative rates. Promising models will be deployed to staging for further testing. We will use an MLflow server on Chameleon to track experiments.

- **Testing Integration:** The pipeline will test performance with batches of X-rays in staging and compare results with previous models. Canary testing will simulate hospital workflows to validate performance under realistic conditions. Feedback from radiologists in production will trigger retraining if needed.

- **Integration with Dashboard:** A Grafana dashboard will provide real-time monitoring of the system. It will display pipeline status, model performance metrics, and deployment history to help track the system’s progress and performance.

## UNIT 4: Model training at scale

- **Train and re-train:** Our project will involve training a YOLOv11-L on the VinBigData chest X-ray dataset. We will establish a pipeline for re-training this model using new, labeled production data. While we do not anticipate a real-time production data stream, we will simulate receiving and labeling a portion (10%) of our existing dataset to demonstrate the re-training capability.

- **Modeling:** We have chosen YOLOv11-L model for the task of object detection in chest X-rays, specifically for identifying the abnormalities present in the VinBigData dataset. YOLO is known for real-time object detection models, and the "L" variant typically signifies a larger, more capable model within the YOLOv11 series. This choice is justified by the need for accurate localization of findings in medical images, to improve the overall detection performance, robustness, and reduce uncertainty which is important for our medical business use-case.

### EXTRA DIFFICULTY POINTS:

- **Use distributed training to increase velocity:** To accelerate the training of our YOLOv11-L, we will leverage the Chameleon infrastructure to implement distributed training across multiple GPUs. We will focus on implementing Distributed Data Parallelism (DDP) where each GPU will have a full copy of the model and process a subset of the VinBigData dataset in parallel. We will experiment with different numbers of GPUs and appropriate batch sizes to determine the optimal configuration for reducing the total model training time.
	
We will conduct experiments comparing the total model training time using a single GPU versus training with multiple GPUs (e.g: 2 and 4) of the same type available on Chameleon. We will include a discussion of the chosen batch size and its rationale. Our final report will include a plot of training time versus the number of GPUs, backed up by our experimental results.

## UNIT 5: Model training infrastructure and platform

- **Experiment tracking:** We will deploy an MLflow tracking server on Chameleon Cloud to centralize all experiment monitoring and management. The training code will log system metrics, model development artifacts (code versions, model checkpoints), and performance metrics (training/validation losses, accuracies). The MLflow UI will provide visualizations for comparing experiments, while a structured tagging system will organize runs by model architecture, dataset version, and key hyperparameters, creating a searchable history of all training experiments.

- **Scheduling training jobs:** The Ray cluster on Chameleon Cloud will consist of one head node for orchestration and multiple worker nodes for computation, with configurable resource allocation per training job. We'll implement a continuous training pipeline that will trigger new training jobs when a request is sent. A Ray Dashboard for real-time monitoring of resource utilization and job status. Custom scripts will handle environment setup, data preparation, and post-training evaluation, creating a fully automated workflow from development to model evaluation.

### EXTRA DIFFICULTY POINTS:

- **Using Ray Train:** We will leverage Ray Train to implement robust distributed training with fault tolerance mechanisms that automatically restart failed training jobs on available workers. Our implementation will include configurable checkpointing with integration to MiniIO deployed on Chameleon for efficient model persistence. We'll utilize Ray Train's scaling capabilities to distribute training across multiple nodes.

- **Scheduling hyperparameter tuning jobs:**  We will use Ray Tune's for hyperparameter search in comprehensive search spaces covering learning rates and optimization parameters (momentum, weight decay), network architecture choices and data augmentation configurations. 

## UNIT 6: Model serving

- **Serving from an API endpoint:** We will wrap our trained YOLOv11-L in a FastAPI framework due to its ease of use and performance in building asynchronous APIs, which can be beneficial for handling multiple concurrent requests. This API will accept chest X-ray images as input and return the detected abnormalities with their bounding boxes and confidence scores. If we implement a front-end (which is not a strict requirement of this unit), it would use this API.

- **Identify requirements:** Based on the business use case of assisting medical professionals in analyzing chest X-rays, we identify the following preliminary requirements:

  - **Model Size:** While high accuracy is necessity, a reasonably sized model (after potential optimizations) is desirable for efficient deployment and potential future consideration of edge deployment. We aim to keep the final serving model under 10MB if possible, depending on the impact of optimizations on accuracy.

  - **Latency for Online (Single Sample) Inference:** For a clinician needing a quick analysis of a single X-ray, we aim for a median inference latency of under 200 milliseconds on a server-grade GPU.

  - **Throughput for Batch Inference:** For scenarios involving the analysis of multiple historical scans or processing a queue of images, we aim for a batch throughput of at least 30 frames per second when serving on a GPU.

  - **Concurrency:** We anticipate a need to handle multiple simultaneous requests from different users or systems. We will aim to support a concurrency of at least 5-10 simultaneous requests without significant degradation in latency.

- **Model optimizations to satisfy requirements:** To meet the identified performance requirements, we will explore the following model-level optimizations:

  - **Conversion to ONNX:** We will convert our PyTorch-based YOLOv11-L models to the ONNX format. This facilitates the use of various optimization techniques and hardware-specific execution providers.

  - **Quantization:** We will investigate post-training quantization techniques offered by tools like ONNX Runtime or OpenVINO. This may include:

    - **Dynamic Quantization:** Applying quantization to the weights and activations dynamically during inference.

    - **Static Quantization:** Using a calibration dataset (a subset of our VinBigData training data) to determine optimal quantization parameters for weights and activations. We will experiment with different quantization levels and tolerance for accuracy loss.

    - **Hardware-Specific Execution Providers:** We will leverage hardware-specific execution providers available in ONNX Runtime, such as CUDAExecutionProvider and TensorrtExecutionProvider, to utilize the parallel processing capabilities of the GPUs on the Chameleon infrastructure. We will compare the performance gains achieved with these providers against CPU-based inference.

- **System optimizations to satisfy requirements:** For deployment on the Chameleon cloud, we will explore system-level optimizations:

  - **Concurrency Management with a Serving Framework:** We plan to use a dedicated model serving framework like the Triton Inference Server. Triton allows for managing multiple model instances, handling concurrent requests efficiently, and supports dynamic batching.

  - **Dynamic Batching:** We will configure Triton to use dynamic batching to aggregate incoming single requests into batches before feeding them to the model for inference. This can significantly improve throughput without drastically increasing latency under moderate load. We will experiment with different preferred batch sizes and queue delay settings.

  - **Scaling Model Instances:** We will explore scaling our model deployment by running multiple instances of the model on the available GPUs within our Chameleon instance. We will analyze the trade-offs between increased throughput and potential resource contention or increased latency per request.

Our final report will detail the system-level optimizations implemented, the configurations used, and the observed effects on concurrency handling, latency, and throughput, including any bottlenecks identified.

### EXTRA DIFFICULTY POINTS:

- **Develop multiple options for serving:** Since our YOLOv11-L model benefits from GPU acceleration, we will develop and evaluate optimized serving options for:

  - **Server-grade GPU (on Chameleon):** We will focus on achieving the latency and throughput targets outlined above using ONNX Runtime with CUDA/TensorRT execution providers and Triton Inference Server on a Chameleon GPU instance (e.g., an instance with NVIDIA A100 or P100 GPUs, depending on availability).

  - **Server-grade CPU (on Chameleon):** We will evaluate the inference performance of our ONNX-optimized model (potentially with CPU-specific optimizations or execution providers like OpenVINO if feasible) on a server-grade CPU instance on Chameleon (e.g., an instance with an AMD EPYC processor). We will measure the achievable latency and throughput and compare them to the GPU-based serving.

  - **On-device (Conceptual Evaluation):** While a full deployment on a low-resource edge device like a Raspberry Pi 5 might be beyond the scope of our project due to resource and time constraints, we will evaluate the feasibility of on-device serving. This will involve:

  - **Further Model Size Reduction:** Exploring more aggressive quantization techniques or model pruning to meet the stringent size requirements for edge devices (e.g., < 5MB as mentioned in the lab).

  - **Benchmarking on Simulated Edge Environments:** We might use software emulators or smaller Chameleon instances to simulate the performance of a low-resource ARM Cortex A76 processor using our most size-efficient model variant.

  - **Discussion of Potential Edge Deployment:** In our report, we will discuss the potential benefits and challenges of on-device serving for chest X-ray analysis in specific scenarios (e.g., remote clinics with limited network connectivity).

## UNIT 7: Evaluation and monitoring 

- **Offline Evaluation:**: YOLOv11 will be evaluated offline using VinBigData’s test set of around 3,600 images. The evaluation will include standard metrics like mAP@0.5 for detection accuracy, domain-specific metrics such as sensitivity for each pathology (e.g., pneumothorax), and performance analysis by pathology and image quality (e.g., noisy vs. clear images). Template-based unit tests will also be conducted to check robustness, such as ensuring Gaussian noise does not affect major detections and occlusion reduces confidence scores. The process will be automated using Python and logged in MLFlow.

- **Load Test in Staging:** In staging, the system will be tested by simulating chest X-rays (CXRs). Prometheus will log throughput and latency metrics, which will be visualized in Grafana dashboards to ensure the system can handle the workload effectively.

- **Online Evaluation in Canary:** For canary evaluation, artificial requests will simulate radiologists uploading CXRs to test the new system. Prometheus will track latency (targeting less than 100ms), throughput, and simulated sensitivity using VinBigData labels. These metrics will be displayed in Grafana dashboards for real-time monitoring.

- **Feedback Loop:** A feedback loop will flag predictions with confidence below a certain threshold for simulated human review by radiologists. Additionally, some part of production data will be saved and labeled for retraining, ensuring continuous improvement of the model.
  
- **Business-Specific Evaluation:** Two business-specific metrics will be defined, time saved and missed pathologies. Prometheus will log baseline processing times compared to AI-assisted times. Inference times will support this goal and will be visualized in Grafana. Missed pathologies will be measured by comparing radiologist sensitivity with AI-assisted sensitivity, using VinBigData labels. These results will also be displayed in Grafana.

- **Monitoring:** Prometheus will collect latency, throughput, and confidence scores during production to support load testing, canary evaluations, and feedback loops. Grafana will provide visualizations for these metrics and business evaluation results.

### EXTRA DIFFICULTY POINTS:

- **Monitor for Model Degradation:** Model degradation will be monitored by logging simulated sensitivity over time using feedback loop labels in Prometheus. Grafana dashboards will detect performance drops and trigger automatic retraining with newly labeled data to maintain prediction quality.

## UNIT 8: Data Pipeline

- **Persistent Storage:** Persistent storage will be set up on Chameleon, following the approach outlined in Lab 8. This storage will be used to house models, training results, and the VinDr-CXR dataset. The storage will be organized into separate folders for raw data, processed data, and models to maintain clarity and ease of access. Additionally, it will be designed to attach seamlessly to various infrastructure components as needed.

- **Offline Data Management:** For offline data management, the VinDr-CXR dataset, which includes chest X-rays and radiologist annotations, will be stored and organized. A simple folder structure will be created to separate the data into training, validation, and test splits. To ensure traceability and consistency, the data will be version-controlled to track any preprocessing changes or updates.

- **ETL Pipeline:** The ETL pipeline will handle the end-to-end processing of the VinDr-CXR dataset. This pipeline will download and extract the dataset, resize images to match the format required by YOLOv11-L, and convert radiologist annotations into a YOLO-compatible format. It will also filter out corrupted or unusable images and save the processed data in a format ready for training. Additionally, a mechanism will be implemented to incorporate feedback data into the pipeline for retraining purposes.

- **Online Data:** To simulate online data usage, a simple script will be developed that sends X-ray images to the system at a realistic rate, mimicking real hospital scenarios. The script will include a mix of normal and abnormal cases. These images will undergo the same preprocessing steps as those used for training data to maintain consistency.

### EXTRA DIFFICULTY POINTS:
- **Interactive Data Dashboard:** An interactive data dashboard will be built using Grafana and Prometheus to provide insights into various aspects of the system. The dashboard will display basic dataset statistics such as the number of images and types of findings, along with data quality metrics like missing annotations or image quality issues. It will also track model performance metrics over time and show simulated inference requests along with their processing times. This dashboard will serve as a critical tool for monitoring and analysis.
