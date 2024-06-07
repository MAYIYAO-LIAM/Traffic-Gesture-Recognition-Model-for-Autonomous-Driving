# Traffic Gesture Recognition Model for Autonomous Driving

## Introduction
The Traffic Gesture Recognition Model for Autonomous Driving is designed to enable autonomous vehicles to interpret and react to traffic police gestures in real-time, enhancing safety and efficiency in complex traffic scenarios where conventional signaling is inadequate. This system integrates advanced computer vision and deep learning techniques, such as MediaPipe and RNNs with LSTM cells, to process live camera feeds and accurately identify nuanced gestures, offering a vital interpretative tool for autonomous driving systems amidst the variable and unpredictable conditions of urban traffic.

## Features

- **Hybrid MediaPipe-LSTM Architecture**: Merges MediaPipe's keypoint detection with LSTM's sequential data processing for accurate, real-time gesture recognition.

- **Data Preprocessing and Augmentation**: Utilizes custom classes for image processing, keypoint extraction, and noise addition to enhance model robustness against real-world variances.

- **Smart Data Handling**: Leverages a tailored `SkeletonDataset` class for efficient data loading and preprocessing, ensuring smooth training and testing cycles.

- **Attention-Enhanced LSTM**: Incorporates an attention mechanism for nuanced gesture understanding, with a network design that balances depth and complexity to prevent overfitting.

- **Streamlined Training Workflow**: Features a `RNNTrainer` class for systematic model training and evaluation, with metrics visualization for ongoing performance assessment.

- **Performance-Optimized Model**: Automatically preserves the best-performing model weights, facilitating reliable deployment and consistent gesture recognition.


## System Requirements

To run the Traffic Gesture Recognition Model, your system needs to meet the following requirements:

### Software
- Python 3.6 or later
- OpenCV (cv2) - for image processing
- MediaPipe - for pose estimation and keypoint detection
- PyTorch - for machine learning model construction and evaluation
- NumPy - for numerical computing
- SciPy - for scientific and technical computing
- Pandas - for data manipulation and analysis
- Matplotlib - for creating static, interactive, and animated visualizations in Python
- Seaborn - for statistical data visualization
- Scikit-learn - for machine learning and statistical modeling including classification, regression, clustering, and dimensionality reduction

### Hardware
- CPU: x86-64 architecture with SSE2 instruction set support
- GPU: (Optional) CUDA-capable GPU for PyTorch acceleration

### Operating System
- Compatible with Windows, macOS, or Linux operating systems that can run the software requirements mentioned above.

It is also recommended to use a virtual environment for Python to manage dependencies more effectively.

### Installation of Dependencies
To install the required libraries, use the following command:
    ```bash 
    pip install numpy opencv-python mediapipe torch pandas matplotlib seaborn scikit-learn

## Usage
### Model Training
#### 1. File Structure

- gesture_data: Dataset folder
- image_skeleton_data: Folder containing - image frames and skeleton data in .pkl format
- rnn_model: Folder for storing trained model parameters and evaluation visualizations
- data_augmentation.py: Data augmentation script
- get_skeleton.py: Script to get the skeleton from images
- Model_train.py: Main model training script
- RNN_data.py: RNN data processing script
- RNN_model.py: RNN model architecture script
- train_test_function.py: Functions for training and testing the model

#### 2. Initial Setup

Make sure you have all the necessary dependencies installed. If not, refer to the system requirements section.

### Frontend
#### 1. File Structure

- pycache: Python's byte-compiled cache folder
- front.css: CSS file defining appearance and styling
- front.html: HTML file defining the structure and content of the web page
- music.wav: Sound or music file in WAV format
- python.py: Backend logic or server-side Python script
- rnn_epoch73_loss0.19.pth: Saved model or weights of a trained RNN
- RNN_model.py: Script defining the RNN model architecture
- script.js: JavaScript file for frontend interactivity

#### 2. Initial Setup

Make sure you have all the necessary dependencies installed. If not, refer to the system requirements section.

#### 3. Launching the Frontend

- Open a terminal or command prompt in the directory containing the frontend files.
- Run the `python.py` script to start the server:
  ```bash
  python python.py

#### 4. Accessing the Web Interface
- Open your preferred web browser.
- Navigate to the address displayed in the terminal (usually something like `http://localhost:PORT`). If no address is displayed, try `http://localhost:5000` or the default port your Python script uses.

#### 5. Using the Web Interface
##### Main Interface
- The `front.html` file serves as the main interface. It will likely present buttons, input fields, and other UI elements.

##### Static Assets
- `111.png` and `button1.png` are static image assets used in the interface. They might be background images, buttons, or icons.
- `time.gif` is a gif file that could be an animation or loader displayed in the frontend.
- `music.wav` could be a background sound or an audio notification.

##### Styling
- The appearance of the web interface is defined by `front.css`. This CSS file contains styles for various elements present in `front.html`.

##### Frontend Logic
- The `script.js` file contains JavaScript code that manages the frontend logic, such as handling button clicks, form submissions, and updating UI elements dynamically.

#### 6. Using the Gesture Recognition Model
- While the specific details aren't mentioned in the files list, the frontend might have options or buttons to:
    - Upload an image or video for gesture recognition.
    - Start real-time gesture recognition using a webcam.
    - Display results or feedback from the recognition model.

#### 7. Shutdown
- To stop the server, go back to the terminal or command prompt and press `Ctrl + C`.

##### Notes
- If any error occurs, ensure all dependencies are correctly installed.
- If you make changes to `front.css`, `script.js`, or `front.html`, you might need to restart the server for changes to take effect.
- Ensure the RNN model (`rnn_epoch73_loss0.19.pth`) is in the correct directory path as specified in the `python.py` script.

## Dataset
[Describe the dataset used, how to access it, and any preprocessing steps required.]


