# YOLOv5-PDLPR algorithm

## Task
The objective of this project is to design and implement a deep learning-based system for license plate recognition, following the methodology outlined in [[1](https://www.mdpi.com/1424-8220/24/9/2791)]. The proposed solution is structured as a two-stage pipeline, leveraging the strengths of different neural network architectures to address the distinct subtasks involved in the recognition process.

- In the **first stage**, a **YOLOv5** model is employed for license plate detection, allowing for fast and accurate localization of the plate region within vehicle images, even under challenging environmental conditions.

- In the **second stage**, the cropped plate region is passed to a specialized recognition model based on the **PDLPR** architecture. This model is responsible for decoding the sequence of alphanumeric characters on the plate, effectively treating the task as a sequence prediction problem.

The integration of these two components aims to deliver a robust and efficient system for plate recognition and reconstruction suitable for deployment in real-world scenarios.

---

## Main Objectives

- **Baseline implementation, training and evaluation**  
  Implement a simple baseline, train and evaluate it with the metrics used in [[1](https://www.mdpi.com/1424-8220/24/9/2791)].

- **YOLOv5 and PDLPR model implementation and evaluation**  
  Implement the proposed model in [1], composed of the YOLOv5 and PDLPR models, and evaluate it.

- **Comparison with the baseline**  
  Compare the performance of the proposed model with the baseline, underlining why the proposed model works better or not on recognizing and reconstructing the car plates.

---

## References

1. Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). [A Real-Time License Plate Detection and Recognition Model in Unconstrained Scenarios](https://www.mdpi.com/1424-8220/24/9/2791). *Sensors*, 24(9), 2791.


2. Xu, Z.; Yang, W.; Meng, A.; Lu, N.; Huang, H.; Ying, C.; Huang, L. *Towards end-to-end license plate detection and recognition: A large dataset and baseline*. In Proceedings of the European Conference on Computer Vision (ECCV), Munich, Germany, 8–14 September 2018.