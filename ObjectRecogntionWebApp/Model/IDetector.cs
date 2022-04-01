namespace ObjectRecogntionWebApp.Model
{
    public interface IDetector 
    {
        public List<Detection> DetectObjects(string imageUrl, HashSet<string> labels, float confidence);
    }
}
