**CSM02-Cricket-Video-to-Text-Summarization-Using-Neural-Networks-MajorProject**

**RESULTS:**

![image](https://github.com/SaiPrashanthMaddhe/CSM02-Cricket-Video-to-Text-Summarization-Major/assets/101933141/59528ab2-a8aa-4ded-8fa0-59363049fdbc)

![image](https://github.com/SaiPrashanthMaddhe/CSM02-Cricket-Video-to-Text-Summarization-Major/assets/101933141/7db0e289-180c-4db2-be4d-37856be06d56)

![image](https://github.com/SaiPrashanthMaddhe/CSM02-Cricket-Video-to-Text-Summarization-Major/assets/101933141/dd62fe70-284c-4eae-a304-cc8e28841359)

![image](https://github.com/SaiPrashanthMaddhe/CSM02-Cricket-Video-to-Text-Summarization-Major/assets/101933141/483b5bbb-fe0f-492c-a1c2-991820600b25)

![image](https://github.com/SaiPrashanthMaddhe/CSM02-Cricket-Video-to-Text-Summarization-Major/assets/101933141/eed8da02-2ad3-49ab-bac2-11f974e1460a)

Our project aims to generate textual summaries from the uploaded cricket video as input utilizing neural networks YOLO, CNN VGG16, LSTM, Paddle OCR, and Distilbart. Frames are extracted from the cricket video and then representative frames are generated. YOLOV8 is performed on representative frames to extract the scoreboard from the frames. CNN and LSTM are utilized to predict the action going in the frame. OCR is performed on the scoreboard extracted from the frame to extract the team, batsman, and bowler information. This data is stored in csv file.


**SOFTWARE REQUIREMENTS:**
1. Operating System: Compatible with Windows 10, macOS Mojave (10.14) or later, or
popular Linux distributions such as Ubuntu 18.04 LTS or newer versions.
2. Python Environment: Python 3.x is installed with essential libraries such as
TensorFlow, Keras, scikit-learn, NumPy, NLTK, and OpenCV for machine learning, natural language processing, and image processing tasks.
3. Development Tools: Integrated Development Environments (IDEs) such as PyCharm, Jupyter Notebook, or VSCode for coding, debugging, and experimentation.
4. Version Control: Git installed for version control management, facilitating
collaboration and tracking changes in code and project files.
5. External Libraries and Models: Installation of additional libraries and models such as
Paddle OCR, BART (Bidirectional and Auto-Regressive Transformers), and pretrained models like VGG16 for image processing and text summarization tasks.
6. Internet Connectivity: High-speed internet connection for accessing online resources, downloading additional datasets, and cricket match footage.

**HARDWARE REQUIREMENTS**
1. CPU: A modern multi-core processor (Intel Core i5 or equivalent) to handle
computational tasks efficiently.
2. GPU: A dedicated graphics processing unit (NVIDIA GeForce GTX 1060 or equivalent)
with CUDA support for accelerated deep learning computations, especially for training
large neural network models.
3. RAM: A minimum of 8GB of RAM (16GB recommended) to ensure smooth processing
of large datasets and model training operations.
4. Storage: Adequate storage space (at least 500GB HDD or SSD) for storing video datasets, image frames, trained models, and intermediate data files.


**PIP commands for installing the libraries:**


torch
ultralytics
opencv-python
numpy
pillow
pandas
paddlepaddle
paddleocr
matplotlib
transformers
nltk
sentencepiece


**HOW TO IMPLEMENT THIS PROJECT:**


allmodules.py file consists all the code to generate the summary of input cricket video 

1)Run app.py (flask program)
2)It renders index.html
3)Click Upload button to upload a cricket video as input
4)Then click Submit buttom to process the summary for the input video
5)After the processing, click the Generate button to get the textual summary of the uploaded video

