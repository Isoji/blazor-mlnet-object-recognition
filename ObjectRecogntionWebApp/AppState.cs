namespace ObjectRecogntionWebApp.Model
{
    public class AppState
    {
        private FasterRCNN Detector { get; set; }
        private float Threshold { get; set; }
        private bool DetectPerson { get; set; }
        private bool DetectCar { get; set; }
        private string TestString { get; set; }
    }
}
