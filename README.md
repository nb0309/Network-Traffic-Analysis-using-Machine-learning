
# Anomaly Detection In Computer Network
In the ever-evolving landscape of cybersecurity, safeguarding computer networks from malicious activities and unusual behavior has become paramount. Anomaly detection plays a pivotal role in identifying and mitigating potential threats in real-time. This description explores the concept of anomaly detection in computer networks, with a focus on how it can be optimized using Intel's extension for TensorFlow.

Anomaly detection is a sophisticated technique used to identify patterns and behaviors that deviate significantly from normal network activity. In computer networks, normal behavior is typically established by observing historical data, which is then used as a benchmark for detecting anomalies. Anomalies can be indicative of security breaches, network failures, or other irregularities that warrant immediate attention.



## Data
Wireshark captures various data when monitoring network traffic:
```
Packet Information: Individual network packets with details like source/destination IP addresses, port numbers, packet length, and payload content.
Protocol Analysis: Identification and decoding of network protocols (e.g., TCP, UDP, HTTP) for deeper inspection.
Decoding: Human-readable representation of packet data structures (e.g., DNS queries, HTTP requests).
Statistics: Captured packet statistics, including duration, data rates, and potential issues like packet loss.
Filtering: Tools to focus on specific traffic types or search for keywords.
Visualization: Packet flow visualization for understanding network communication patterns.
Expert Info: Flags warnings or errors related to packet issues.
Hierarchy: Displays the protocol breakdown in the capture.
Export: Allows saving captures for further analysis or reporting in different formats.
```
Features:
```
#   Column       Non-Null Count   Dtype  
---  ------       --------------   -----  
 0   No.          117084 non-null  int64  
 1   Time         117084 non-null  float64
 2   Source       117084 non-null  object 
 3   Destination  117084 non-null  object 
 4   Protocol     117084 non-null  object 
 5   Length       117084 non-null  int64  
 6   Info         117084 non-null  object 
 ```

## Tech Stack:
![App Screenshot](https://imgs.search.brave.com/3xxa-ZJZeey5h_Czsj0lckx9eJ_irq7jN5oO680hyCM/rs:fit:560:320:1/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvZW4vdGh1bWIv/Zi9mYS9PbmVBUEkt/cmdiLTMwMDAucG5n/LzUxMnB4LU9uZUFQ/SS1yZ2ItMzAwMC5w/bmc)


#



Intel Extension for Tensorflow*:
1. Plug into Tensorflow 2.10 or late to accelerate training and inference on Intel GPU hardware with no code changes.
2. accelerate AI performance with Intel oneAPI Deep Neural Network Library(oneDNN) features such as graph optimizations and memory pool allocation.
3. Automatically use Intel Deep Learning Boost instruction set features to parallelize and accelerate AI workloads.
4. Enable optimizations by setting the environment variable by
```
TF_ENABLE_ONEDNN_OPTS=1
```
##
Intel Distribution for Python*:
1. The distribution is designed to scale efficiently across multiple CPU cores and threads. This scalability is essential for applications that required high-performance computing.
2. Essential Python bindings for easing integration of Intel native tools with the python project. It seamlessly works with Intel software and libraries.
3. Intel Distribution for python maintains compatibility with the standard python distribution(cpython). This means that most existing python packages and libraries can be used seamlessly with this distribution.

##


Intel Extension for scikit-learn*:
1. Intel extension can accelerate scikit-learn algorithms by up to 100x, which can significantly reduce the time it takes to train and deploy machine learning models.
2. The extension is seamlessly integrated with scikit-learn, so you can continue to use the same API and code.
3. The intel extension supports multiple devices, including CPUs, GPUs, and FPGAs. This allows you to choose the best device for your specific applicatino and workload.
   
Add two lines of code to patch all compatible algorithms in your Python script.
```
from sklearnex import patch_sklearn
patch_sklearn()
```    


Wireshark:
Data packet sniffing tool
![App Screenshot](https://imgs.search.brave.com/eZPcDy6jX155eTNG-TC_-d6jzFp5rparfpL5l_zuycM/rs:fit:560:320:1/g:ce/aHR0cHM6Ly91cGxv/YWQud2lraW1lZGlh/Lm9yZy93aWtpcGVk/aWEvY29tbW9ucy90/aHVtYi9jL2NmL1dp/cmVzaGFya18zLjZf/c2NyZWVuc2hvdC5w/bmcvNTEycHgtV2ly/ZXNoYXJrXzMuNl9z/Y3JlZW5zaG90LnBu/Zw)

## Model:
In the pursuit of robust anomaly detection in computer network data, a combination of two powerful techniques has been employed: Isolation Forest and Autoencoders. This dual approach harnesses the strengths of both methodologies to enhance the precision and effectiveness of anomaly detection in complex network environments.

Autoencoders Architecture:

Input Layer:
```
Neurons: Number of input features (determined by input_dim).

Activation: None (raw input).
```
Encoding Layers:
```
Layer 1: Dense layer with 64 neurons and ReLU activation.

Dropout: 20% dropout for regularization.
Layer 2: Dense layer with 32 neurons and ReLU activation.

Encoding Bottleneck: Dense layer with encoding_dim (10) neurons and ReLU activation.
```
Decoding Layers:
```
Layer 1: Dense layer with 32 neurons and ReLU activation.

Dropout: 20% dropout for regularization.

Layer 2: Dense layer with 64 neurons and ReLU activation.

Output Layer: Dense layer with the same number of neurons as input features (specified by input_dim) and sigmoid activation.
```
Model Compilation:
```
Optimizer: Adam optimizer.
Loss Function: Mean Squared Error (MSE) for reconstruction loss.
Training:

Input and Target: Scaled input data (X_scaled) used as both input and target.
Epochs: 20.
Batch Size: 32.
```
Anomaly Detection:
```
After training, the model calculates MSE between original data and its reconstruction.
Anomaly threshold is set at the 99.9th percentile of MSE values.
```
Identifying Anomalies:
```
Data points with MSE above the threshold are considered
```

Ensemble Method:

Combinig both randomforestclassifier and isolation forest.

Isolation Forest (IsolationForest):
```
Isolation Forest is used for initial anomaly score estimation.
contamination is set to 0.0045, and random_state is 42.
Anomaly scores are predicted for data points, where -1 indicates anomalies and 1 indicates normal data.
```
Random Forest Classifier (RandomForestClassifier):
```
Random Forest Classifier refines the anomaly detection process.
n_estimators is 100, and random_state is 42.
It is trained on features and anomaly labels derived from the Isolation Forest.
Anomaly predictions are made, and anomalies are identified where the prediction is 0 (anomaly).
```
## Anomaly points:

```Anomaly points:
           No.         Time                                   Source  \
11901    11902   379.582211  2409:40f4:100b:c1b6:b9fb:3ec3:5675:a236   
19689    19690   424.147757             2405:200:1630:a03::312c:c5b3   
19831    19832   424.232626             2405:200:1630:a03::312c:c5b3   
20090    20091   424.369842             2405:200:1630:a03::312c:c5b3   
20118    20119   424.388097             2405:200:1630:a03::312c:c5b3   
...        ...          ...                                      ...   
100885  100886  5232.917759  2409:40f4:100b:c1b6:b9fb:3ec3:5675:a236   
100888  100889  5232.938174  2409:40f4:100b:c1b6:b9fb:3ec3:5675:a236   
100897  100898  5233.018599  2409:40f4:100b:c1b6:b9fb:3ec3:5675:a236   
115756  115757  6767.014347  2409:40f4:100b:c1b6:5a68:c2f9:5206:af46   
115757  115758  6767.014478                           192.168.239.25   

                                    Destination  Protocol  Length  \
11901                  2404:6800:4007:816::2002         5    1294   
19689   2409:40f4:100b:c1b6:b9fb:3ec3:5675:a236        12    2662   
19831   2409:40f4:100b:c1b6:b9fb:3ec3:5675:a236        12    2662   
20090   2409:40f4:100b:c1b6:b9fb:3ec3:5675:a236        12    2662   
20118   2409:40f4:100b:c1b6:b9fb:3ec3:5675:a236        12    2662   
...                                         ...       ...     ...   
100885                 2404:6800:4007:810::2002         5    1294   
100888                 2404:6800:4007:810::2002         5    1294   
100897                 2404:6800:4007:810::2002         5    1294   
115756                                 ff02::fb         7     221   
115757                              224.0.0.251         7     201   

                                                     Info  anomaly  
11901          Destination Unreachable (Port unreachable)        1  
19689                   Encrypted Data, Continuation Data        1  
19831                   Encrypted Data, Continuation Data        1  
20090                   Encrypted Data, Continuation Data        1  
20118                   Encrypted Data, Continuation Data        1  
...                                                   ...      ...  
100885         Destination Unreachable (Port unreachable)        1  
100888         Destination Unreachable (Port unreachable)        1  
100897         Destination Unreachable (Port unreachable)        1  
115756  Standard query 0x0000 PTR _nfs._tcp.local, "QM...        1  
115757  Standard query 0x0000 PTR _nfs._tcp.local, "QM...        1  
```
## Epoch:
Without Intel Extension for Tensorflow:
![image](https://github.com/nb0309/Network-Traffic-Analysis-using-Machine-learning/assets/93106796/f985f3b9-d78f-472d-8ef4-99b9871a1f66)


With using Intel Extension for Tensorflow:
![image](https://github.com/nb0309/Network-Traffic-Analysis-using-Machine-learning/assets/93106796/f6c8fe7f-30cd-4412-bf53-8e341ba3a609)



## Contributors:

- [@navabhaarathi](https://github.com/nb0309)
- [@balasuriya](https://github.com/balasuriyaranganathan/balasuriyaranganathan)


## Acknowledgements

 - [Computer Network Intrusion and anomaly detection](https://www.hindawi.com/journals/misy/2022/6576023/)
 - [Intel Distribution for Python](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
 - [Intel oneAPI](https://www.oneapi.io/)
 - [Wireshark](https://www.wiresharp.org/)
