using Microsoft.ML.OnnxRuntime;

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
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Stores the coordinates for a bounding box
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
}
