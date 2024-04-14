**CSM02-Cricket-Video-to-Text-Summarization-Using-Neural-Networks-MajorProject**


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
pip install torch
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pillow
pip install pandas
pip install paddlepaddle paddleocr
pip install matplotlib
pip install transformers
pip install nltk
pip install sentencepiece

**HOW TO IMPLEMENT THIS PROJECT:**


allmodules.py file consists all the code to generate the summary of input cricket video 

1)Run app.py (flask program)
2)It renders index.html
3)Click Upload button to upload a cricket video as input
4)Then click Submit buttom to process the summary for the input video
5)After the processing, click the Generate button to get the textual summary of the uploaded video

