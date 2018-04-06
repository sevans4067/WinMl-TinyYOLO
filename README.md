# WinMl-TinyYOLO
Windows Machine Learning implementation using TinyYOLO (You-Only-Look-Once).  For details on YOLO, check out the inventor's site - https://pjreddie.com/darknet/yolo/

# Acknowledgements
This sample project was based on the Windows Machine Learning examples located at https://github.com/Microsoft/Windows-Machine-Learning. A special thanks to http://machinethink.net/blog/yolo-coreml-versus-mps-graph/ for providing a CoreML model along with example code.

# Setup
Before running this on your local system, you will need the Windows SDK - Build 17110+ (see https://docs.microsoft.com/en-us/windows/uwp/machine-learning/get-started for more information) and Visual Studio 15.7 Preview 1.  If running on desktop Windows, you'll need to set your configuration to x64.

# Model
The TinyYOLO.onnx model embedded with this project was created from the model located at https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/TinyYOLO-CoreML/TinyYOLO-CoreML/TinyYOLO.mlmodel.  This model represents YOLOv2, which was built off the classes in the Pascal VOC project (http://host.robots.ox.ac.uk/pascal/VOC/).  The model was converted from CoreML to ONNX using the tools from https://github.com/onnx/onnxmltools.
