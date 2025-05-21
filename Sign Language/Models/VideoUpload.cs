using System.ComponentModel.DataAnnotations;

namespace Sign_Language.Models
{
    public class VideoUpload
    {
        [Required]
        public IFormFile VideoFile { get; set; }

        public string TranslatedText { get; set; }
    }
} 