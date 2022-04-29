namespace ObjectRecogntionWebApp.Model
{
    /// <summary>
    /// Class for storing the configuration state of the application.
    /// </summary>
    public class AppState
    {
        // Default properties of the app
        public FasterRCNN Detector { get; set; } = new FasterRCNN();
        public float Threshold { get; set; } = 0.7f;
        public HashSet<string> Classes { get; set; } = new HashSet<string>{"person", "car"};
    }
}
