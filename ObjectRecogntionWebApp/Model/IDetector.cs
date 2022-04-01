namespace ObjectRecogntionWebApp.Model
{
    public interface IDetector 
    {
        public void DetectObjects(string imageUrl, HashSet<string> labels, float confidence);
    }
}
