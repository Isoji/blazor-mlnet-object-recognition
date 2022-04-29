namespace ObjectRecogntionWebApp.Model
{
    /// <summary>
    /// Interface for object detector classes.
    /// </summary>
    public interface IDetector 
    {
        /// <summary>
        /// Method that performs detections on image input data.
        /// </summary>
        /// <param name="imageUrl">The file path to the image.</param>
        /// <param name="labels">The HashSet containing class labels for detection filtering.</param>
        /// <param name="threshold">The score threshold used for detection filtering.</param>
        /// <returns>A list of detections.</returns>
        public List<Detection> DetectObjects(string imageUrl, HashSet<string> labels, float threshold);
    }
}
