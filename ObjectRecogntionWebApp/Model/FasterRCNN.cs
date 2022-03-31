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
            Session = new InferenceSession("/wwwroot/onnx/FasterRCNN-10.onnx");
        }
        public void DetectObjects(string imageUrl, HashSet<string> labels, float confidence)
        {
            Image image = Image<Bgr24>.Load(imageUrl);
        }

        public Tensor<float> PreProcessImage(Image<Bgr24> image)
        {
            var processedImage = image.Clone();
            // Resize image
            float ratio = 800f / Math.Min(processedImage.Width, processedImage.Height);
            processedImage.Mutate(x => x.Resize((int)(ratio * processedImage.Width), (int)(ratio * processedImage.Height)));

            // Pad to be divisiable by 32
            var paddedHeight = (int)(Math.Ceiling(processedImage.Height / 32f) * 32f);
            var paddedWidth = (int)(Math.Ceiling(processedImage.Width / 32f) * 32f);

            var input = new DenseTensor<float>(new[] { 3, paddedHeight, paddedWidth });

            // Normalize image
            var mean = new[] { 102.9801f, 115.9465f, 122.7717f };
            processedImage.ProcessPixelRows(accessor => {
                for (int y = paddedHeight - processedImage.Height; y < processedImage.Height; y++)
                {
                    var pixelRow = accessor.GetRowSpan(y);
                    for (int x = paddedWidth - processedImage.Width; x < processedImage.Width; x++)
                    {
                        input[0, y, x] = pixelRow[x].B - mean[0];
                        input[1, y, x] = pixelRow[x].G - mean[1];
                        input[2, y, x] = pixelRow[x].R - mean[2];
                    }
                }
            });
            return input;
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
        public float Confidence { get; set; }
    }
}
