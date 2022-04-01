using SixLabors.ImageSharp;

namespace ObjectRecogntionWebApp.Model
{
    public interface IDetector 
    {
        public Image DetectObjects(string imageUrl, HashSet<string> labels, float confidence);
    }
}
