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
}
