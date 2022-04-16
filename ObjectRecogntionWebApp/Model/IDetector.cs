using SixLabors.ImageSharp;

namespace ObjectRecogntionWebApp.Model
{
    public interface IDetector 
    {
        public List<Detection> DetectObjects(string imageUrl, HashSet<string> labels);
    }
}
