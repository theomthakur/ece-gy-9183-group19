# Serving and Monitoring

Original code here https://github.com/rigvedrs/serving-monitoring/tree/main

# Implemented the following

## UNIT 6: Model serving

- **Serving from an API endpoint:** We have wrapped our framework in Triton and have tested and tried all configs for GPU, TensorRT, ONNX, OpenVino. There is an additional testing branch for data-drift detection as well. 

Our model has high enough throughput on CPU for our use case and fits the business use case of prioritising accuracy over everything. Along with the server being deployable on CPU grade servers as well without the need of a CPU. But options are open to serve with GPUs as well.

- **Model optimizations to satisfy requirements:** To meet the identified performance requirements, we explored the following model-level optimizations:

  - **Conversion to ONNX:** 
  - **Quantization:** 
  - **TensorRT**

    - **Hardware-Specific Execution Providers:** We leveraged hardware-specific execution providers available in ONNX Runtime, such as CUDAExecutionProvider and TensorrtExecutionProvider, to utilize the parallel processing capabilities of the GPUs on the Chameleon infrastructure. We compared the performance gains achieved with these providers against CPU-based inference.

- **System optimizations to satisfy requirements:** For deployment on the Chameleon cloud, we explored system-level optimizations:

  - **Concurrency Management with a Serving Framework:**  the Triton Inference Server. Triton allows for managing multiple model instances, handling concurrent requests efficiently, and supports dynamic batching.

  - **Dynamic Batching:** We configured Triton to use dynamic batching as well

  - **Scaling Model Instances:** We explored scaling our model deployment by running multiple instances of the model on the available GPUs within our Chameleon instance. 


### EXTRA DIFFICULTY POINTS:

- **Developed multiple options for serving:** Since our YOLOv11-L model benefits from GPU acceleration, we developed and evaluated optimized serving options for:

  - **Server-grade GPU (on Chameleon):** 

  - **Server-grade CPU (on Chameleon):** 

  - **Further Model Size Reduction:** Like half-precision


## UNIT 7: Evaluation and monitoring 

- **Offline Evaluation:**: with the predictions being obtained from Triton

- **Load Test in Staging:** 

- **Online Evaluation in Canary:** 

- **Feedback Loop:** There is an option to track feedback through streamlit and they get uploaded to MinIO which is automatically uploaded to LabelStudio through one script. There is also low confidence imaeg bucketing
  
- **Business-Specific Evaluation:** Two business-specific metrics will be defined, time saved and missed pathologies. The missed pathalogies are being tracked through feedback

- **Monitoring:** Prometheus collects system metrics to support load testing, canary evaluations, and feedback loops. Grafana provides visualizations for these metrics and business evaluation results.

