using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;

namespace ObjectRecogntionWebApp.Model
{
    public class FasterRCNN : IDetector
    {
        readonly InferenceSession Session;

        public FasterRCNN()
        {
            Session = new InferenceSession("C:/Users/tremb/source/repos/ObjectRecogntionWebApp/ObjectRecogntionWebApp/wwwroot/onnx/FasterRCNN-10.onnx");
        }

        /// <summary>
        /// Performs object detection on an image.
        /// </summary>
        /// <param name="path">The file path to the image.</param>
        /// <param name="classes">The HashSet containing class labels for detection filtering.</param>
        /// <param name="threshold">The score threshold used for detection filtering.</param>
        /// <returns>The output Image</returns>
        public Image DetectObjects(string path, HashSet<string> classes, float threshold)
        {
            // Setup input
            Image<Bgr24> image = Image.Load<Bgr24>(path);
            var input = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", PreProcessImage(image))
            };

            // Run inference
            var results = Session.Run(input);

            // Post process results
            var resultsArray = results.ToArray();
            float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
            long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
            float[] scores = resultsArray[2].AsEnumerable<float>().ToArray();
            var detections = new List<Detection>();
            var minScore = threshold;

            // Iterating by 4 because every box has 4 sequenced float values (xmin, ymin, xmax, ymax)
            for (int i = 0; i < boxes.Length - 4; i += 4)
            {
                // Divide by 4 to get appropriate bounding box index
                var index = i / 4;
                string label = LabelMap.Labels[labels[index]];

                if (classes.Contains(label) && (scores[index] >= minScore))
                {
                    // Passed the accepted confidence threshold, add to Prediction list
                    detections.Add(new Detection
                    {
                        Box = new Box(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]),
                        Label = label,
                        Score = scores[index]
                    });

                }
            }
            //string fileName = imageUrl.Split('\\').Last();
            return DrawDetections(image, detections);
        }

        /// <summary>
        /// Uses detection data to draw over an image for output visualization
        /// </summary>
        /// <param name="image">The image that went through detection</param>
        /// <param name="detections">The detections that were inferred from the image</param>
        /// <returns></returns>
        public Image DrawDetections(Image image, List<Detection> detections)
        {
            // Set the font for text to be drawn on image
            Font font = SystemFonts.CreateFont("Arial", 13);

            // Get the ratio for resizing boxes to original image
            float ratio = 800f / Math.Min(image.Width, image.Height);

            foreach (var d in detections) 
            {
                // Resize the bounding box
                d.Box.Xmin /= ratio;
                d.Box.Ymin /= ratio;
                d.Box.Ymax /= ratio;
                d.Box.Xmax /= ratio;

                // Apply transformations to image
                image.Mutate(x =>
                {
                    // Draw bounding box over the image
                    x.DrawLines(Color.Red, 2f, new PointF[]
                    {
                        new PointF(d.Box.Xmin, d.Box.Ymin),
                        new PointF(d.Box.Xmax, d.Box.Ymin),

                        new PointF(d.Box.Xmax, d.Box.Ymin),
                        new PointF(d.Box.Xmax, d.Box.Ymax),

                        new PointF(d.Box.Xmax, d.Box.Ymax),
                        new PointF(d.Box.Xmin, d.Box.Ymax),

                        new PointF(d.Box.Xmin, d.Box.Ymax),
                        new PointF(d.Box.Xmin, d.Box.Ymin)
                    }); 
                    // Draw the class label and accuracy score over the image
                    x.DrawText($"{d.Label}, {d.Score:0.00}", font, Color.White, new PointF(d.Box.Xmin, d.Box.Ymin - 15));
                });
            }
            return image;
        }

        /// <summary>
        /// Pre processes an image into the proper format for inference with Faster RCNN models.
        /// </summary>
        /// <param name="image">Image object to pre process</param>
        /// <returns>Processed image tensor object</returns>
        private Tensor<float> PreProcessImage(Image<Bgr24> image)
        {
            var clonedImage = image.Clone();

            // Resize image
            float ratio = 800f / Math.Min(clonedImage.Width, clonedImage.Height);
            clonedImage.Mutate(x => x.Resize((int)(ratio * clonedImage.Width), (int)(ratio * clonedImage.Height)));

            // Pad to be divisiable by 32
            var paddedHeight = (int)(Math.Ceiling(clonedImage.Height / 32f) * 32f);
            var paddedWidth = (int)(Math.Ceiling(clonedImage.Width / 32f) * 32f);

            var processedInput = new DenseTensor<float>(new[] { 3, paddedHeight, paddedWidth });

            // Normalize image
            var mean = new[] { 102.9801f, 115.9465f, 122.7717f };
            clonedImage.ProcessPixelRows(accessor =>
            {
                for (int y = paddedHeight - clonedImage.Height; y < clonedImage.Height; y++)
                {
                    var pixelRow = accessor.GetRowSpan(y);
                    for (int x = paddedWidth - clonedImage.Width; x < clonedImage.Width; x++)
                    {
                        processedInput[0, y, x] = pixelRow[x].B - mean[0];
                        processedInput[1, y, x] = pixelRow[x].G - mean[1];
                        processedInput[2, y, x] = pixelRow[x].R - mean[2];
                    }
                }
            });
            return processedInput;
        }
    }

    /// <summary>
    /// Stores the coordinates of a bounding box
    /// </summary>
    public class Box
    {
        public float Xmin { get; set; }
        public float Ymin { get; set; }
        public float Xmax { get; set; }
        public float Ymax { get; set; }

        public Box(float xmin, float ymin, float xmax, float ymax)
        {
            Xmin = xmin;
            Ymin = ymin;
            Xmax = xmax;
            Ymax = ymax;

        }
    }

    /// <summary>
    /// Stores information about a detection
    /// </summary>
    public class Detection
    {
        public Box Box { get; set; }
        public string Label { get; set; }
        public float Score { get; set; }
    }
}