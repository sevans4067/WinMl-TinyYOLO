using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.AI.MachineLearning.Preview;
using Windows.Graphics.Imaging;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Media.Imaging;
using Windows.UI.Xaml.Navigation;
using Windows.UI.Xaml.Shapes;
using Windows.UI.Text;
using Windows.Media.Capture;
using Windows.Media.MediaProperties;
using Windows.System.Threading;
using System.Threading;
using System.Diagnostics;
using Windows.Media.Devices;
using Windows.Devices.Enumeration;

namespace TinyYOLO
{
    public sealed partial class MainPage : Page
    {
        private const string MODEL_FILENAME = "TinyYOLO.onnx";

        private readonly SolidColorBrush lineBrush = new SolidColorBrush(Windows.UI.Colors.Yellow);
        private readonly SolidColorBrush fillBrush = new SolidColorBrush(Windows.UI.Colors.Transparent);
        private readonly double lineThickness = 2.0;

        private MediaCapture _captureManager;
        private VideoEncodingProperties _videoProperties;
        private ThreadPoolTimer _frameProcessingTimer;
        private SemaphoreSlim _frameProcessingSemaphore = new SemaphoreSlim(1);

        private ImageVariableDescriptorPreview inputImageDescription;
        private TensorVariableDescriptorPreview outputTensorDescription;
        private LearningModelPreview model = null;

        private IList<YoloBoundingBox> boxes = new List<YoloBoundingBox>();
        private YoloWinMlParser parser = new YoloWinMlParser();

        public MainPage()
        {
            this.InitializeComponent();
        }

        protected override async void OnNavigatedTo(NavigationEventArgs e)
        {
            base.OnNavigatedTo(e);

            await LoadModelAsync();
        }

        private async void ButtonRun_Click(object sender, RoutedEventArgs e)
        {
            ButtonRun.IsEnabled = false;

            try
            {
                // Load the model
                await Task.Run(async () => await LoadModelAsync());

                // Trigger file picker to select an image file
                var fileOpenPicker = new FileOpenPicker();
                fileOpenPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
                fileOpenPicker.FileTypeFilter.Add(".jpg");
                fileOpenPicker.FileTypeFilter.Add(".png");
                fileOpenPicker.ViewMode = PickerViewMode.Thumbnail;
                var selectedStorageFile = await fileOpenPicker.PickSingleFileAsync();

                SoftwareBitmap softwareBitmap;

                using (IRandomAccessStream stream = await selectedStorageFile.OpenAsync(FileAccessMode.Read))
                {
                    // Create the decoder from the stream 
                    var decoder = await BitmapDecoder.CreateAsync(stream);

                    // Get the SoftwareBitmap representation of the file in BGRA8 format
                    softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                    softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                }

                // Encapsulate the image within a VideoFrame to be bound and evaluated
                var inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

                await Task.Run(async () =>
                {
                    // Evaluate the image
                    await EvaluateVideoFrameAsync(inputImage);
                });

                await DrawOverlays(inputImage);
            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
                ButtonRun.IsEnabled = true;
            }
        }

        private async Task DrawOverlays(VideoFrame inputImage)
        {
            this.OverlayCanvas.Children.Clear();

            // Render output
            if (this.boxes.Count > 0)
            {
                // Remove overalapping and low confidence bounding boxes
                var filteredBoxes = this.parser.NonMaxSuppress(this.boxes, 5, .5F);

                foreach (var box in filteredBoxes)
                    await this.DrawYoloBoundingBoxAsync(inputImage.SoftwareBitmap, box);
            }
        }

        private async void OnWebCameraButtonClicked(object sender, RoutedEventArgs e)
        {
            if (_captureManager == null || _captureManager.CameraStreamState != CameraStreamState.Streaming)
            {
                await StartWebCameraAsync();
            }
            else
            {
                await StopWebCameraAsync();
            }
        }

        private async void OnDeviceToggleToggled(object sender, RoutedEventArgs e)
        {
            await LoadModelAsync(DeviceToggle.IsOn);
        }

        /// <summary>
        /// Event handler for camera source changes
        /// </summary>
        private async Task StartWebCameraAsync()
        {
            try
            {
                if (_captureManager == null ||
                    _captureManager.CameraStreamState == CameraStreamState.Shutdown ||
                    _captureManager.CameraStreamState == CameraStreamState.NotStreaming)
                {
                    if (_captureManager != null)
                    {
                        _captureManager.Dispose();
                    }

                    // Workaround since my home built-in camera does not work as expected, so have to use my LifeCam
                    MediaCaptureInitializationSettings settings = new MediaCaptureInitializationSettings();
                    var allCameras = await DeviceInformation.FindAllAsync(DeviceClass.VideoCapture);
                    var selectedCamera = allCameras.FirstOrDefault(c => c.Name.Contains("LifeCam")) ?? allCameras.FirstOrDefault();
                    if (selectedCamera != null)
                    {
                        settings.VideoDeviceId = selectedCamera.Id;
                    }
                    //settings.PreviewMediaDescription = new MediaCaptureVideoProfileMediaDescription()

                    _captureManager = new MediaCapture();
                    await _captureManager.InitializeAsync(settings);

                    WebCamCaptureElement.Source = _captureManager;
                }

                if (_captureManager.CameraStreamState == CameraStreamState.NotStreaming)
                {
                    if (_frameProcessingTimer != null)
                    {
                        _frameProcessingTimer.Cancel();
                        _frameProcessingSemaphore.Release();
                    }

                    TimeSpan timerInterval = TimeSpan.FromMilliseconds(66); //15fps
                    _frameProcessingTimer = ThreadPoolTimer.CreatePeriodicTimer(new TimerElapsedHandler(ProcessCurrentVideoFrame), timerInterval);

                    _videoProperties = _captureManager.VideoDeviceController.GetMediaStreamProperties(MediaStreamType.VideoPreview) as VideoEncodingProperties;

                    await _captureManager.StartPreviewAsync();

                    WebCamCaptureElement.Visibility = Visibility.Visible;
                }
            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
            }
        }

        public async Task StopWebCameraAsync()
        {
            try
            {
                if (_frameProcessingTimer != null)
                {
                    _frameProcessingTimer.Cancel();
                }

                if (_captureManager != null && _captureManager.CameraStreamState != CameraStreamState.Shutdown)
                {
                    await _captureManager.StopPreviewAsync();
                    WebCamCaptureElement.Source = null;
                    _captureManager.Dispose();
                    _captureManager = null;

                    WebCamCaptureElement.Visibility = Visibility.Collapsed;
                }
            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
            }
        }

        private async void ProcessCurrentVideoFrame(ThreadPoolTimer timer)
        {
            if (_captureManager.CameraStreamState != CameraStreamState.Streaming || !_frameProcessingSemaphore.Wait(0))
            {
                return;
            }

            try
            {
                const BitmapPixelFormat InputPixelFormat = BitmapPixelFormat.Bgra8;
                VideoFrame previewFrame = new VideoFrame(InputPixelFormat, (int)_videoProperties.Width, (int)_videoProperties.Height);
                await _captureManager.GetPreviewFrameAsync(previewFrame);
                await EvaluateVideoFrameAsync(previewFrame);
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, async () => 
                {
                    await DrawOverlays(previewFrame);
                    previewFrame.Dispose();
                });

            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
            }
            finally
            {
                _frameProcessingSemaphore.Release();
            }
        }

        private async Task LoadModelAsync(bool isGpu = true)
        {
            if (this.model != null)
                return;

            await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"Loading { MODEL_FILENAME } ... patience ");

            try
            {
                // Load Model
                var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/{ MODEL_FILENAME }"));
                this.model = await LearningModelPreview.LoadModelFromStorageFileAsync(modelFile);
                this.model.InferencingOptions.ReclaimMemoryAfterEvaluation = true;
                this.model.InferencingOptions.PreferredDeviceKind = isGpu == true ? LearningModelDeviceKindPreview.LearningDeviceGpu : LearningModelDeviceKindPreview.LearningDeviceCpu;

                // Retrieve model input and output variable descriptions (we already know the model takes an image in and outputs a tensor)
                var inputFeatures = this.model.Description.InputFeatures.ToList();
                var outputFeatures = this.model.Description.OutputFeatures.ToList();

                this.inputImageDescription =
                    inputFeatures.FirstOrDefault(feature => feature.ModelFeatureKind == LearningModelFeatureKindPreview.Image)
                    as ImageVariableDescriptorPreview;

                this.outputTensorDescription =
                    outputFeatures.FirstOrDefault(feature => feature.ModelFeatureKind == LearningModelFeatureKindPreview.Tensor)
                    as TensorVariableDescriptorPreview;

                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"Loaded { MODEL_FILENAME }. Press the camera button to start the webcam...");

            }
            catch (Exception ex)
            {
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
                model = null;
            }
        }

        private async Task EvaluateVideoFrameAsync(VideoFrame inputFrame)
        {
            if (inputFrame != null)
            {
                try
                {
                    // Create bindings for the input and output buffer
                    var binding = new LearningModelBindingPreview(this.model as LearningModelPreview);

                    // R4 WinML does needs the output pre-allocated for multi-dimensional tensors
                    var outputArray = new List<float>();
                    outputArray.AddRange(new float[21125]);  // Total size of TinyYOLO output

                    binding.Bind(this.inputImageDescription.Name, inputFrame);
                    binding.Bind(this.outputTensorDescription.Name, outputArray);

                    // Process the frame with the model
                    var stopwatch = Stopwatch.StartNew();
                    var results = await this.model.EvaluateAsync(binding, "TinyYOLO");
                    stopwatch.Stop();
                    var resultProbabilities = results.Outputs[this.outputTensorDescription.Name] as List<float>;

                    // Use out helper to parse to the YOLO outputs into bounding boxes with labels
                    this.boxes = this.parser.ParseOutputs(resultProbabilities.ToArray(), .3F);

                    await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
                    {
                        Duration.Text = $"{1000f / stopwatch.ElapsedMilliseconds,4:f1} fps";
                        StatusBlock.Text = "Model Evaluation Completed";
                    });
                }
                catch (Exception ex)
                {
                    await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
                }

                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => ButtonRun.IsEnabled = true);
            }
        }

        private async Task DrawYoloBoundingBoxAsync(SoftwareBitmap inputImage, YoloBoundingBox box)
        {
            // Scale is set to stretched 416x416 - Clip bounding boxes to image area
            var x = (uint)Math.Max(box.X, 0);
            var y = (uint)Math.Max(box.Y, 0);
            var w = (uint)Math.Min(this.OverlayCanvas.Width - x, box.Width);
            var h = (uint)Math.Min(this.OverlayCanvas.Height - y, box.Height);

            var brush = new ImageBrush();

            var bitmapSource = new SoftwareBitmapSource();
            await bitmapSource.SetBitmapAsync(inputImage);

            brush.ImageSource = bitmapSource;
            brush.Stretch = Stretch.Fill;

            this.OverlayCanvas.Background = brush;

            var r = new Rectangle();
            r.Tag = box;
            r.Width = w;
            r.Height = h;
            r.Fill = this.fillBrush;
            r.Stroke = this.lineBrush;
            r.StrokeThickness = this.lineThickness;
            r.Margin = new Thickness(x, y, 0, 0);

            var tb = new TextBlock();
            tb.Margin = new Thickness(x + 4, y + 4, 0, 0);
            tb.Text = $"{box.Label} ({Math.Round(box.Confidence, 4).ToString()})";
            tb.FontWeight = FontWeights.Bold;
            tb.Width = 126;
            tb.Height = 21;
            tb.HorizontalTextAlignment = TextAlignment.Center;

            var textBack = new Rectangle();
            textBack.Width = 134;
            textBack.Height = 29;
            textBack.Fill = this.lineBrush;
            textBack.Margin = new Thickness(x, y, 0, 0);

            this.OverlayCanvas.Children.Add(textBack);
            this.OverlayCanvas.Children.Add(tb);
            this.OverlayCanvas.Children.Add(r);
        }
    }
}
