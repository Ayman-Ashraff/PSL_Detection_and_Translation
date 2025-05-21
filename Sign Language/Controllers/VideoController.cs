using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using Sign_Language.Models;
using Sign_Language.Services;
using System.Diagnostics;

namespace Sign_Language.Controllers
{
    [Authorize]
    public class VideoController : Controller
    {
        private readonly IWebHostEnvironment _environment;
        private readonly ILogger<VideoController> _logger;
        private readonly SignLanguageService _signLanguageService;

        public VideoController(IWebHostEnvironment environment, ILogger<VideoController> logger, SignLanguageService signLanguageService)
        {
            _environment = environment;
            _logger = logger;
            _signLanguageService = signLanguageService;
        }

        public IActionResult Upload()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Upload(VideoUpload model)
        {
                   _logger.LogInformation("Upload POST called");
            Console.WriteLine("upload called");

            if (model.VideoFile == null)
            {
                Console.WriteLine("VideoFile is null!");
            }
            else { Console.WriteLine("tjis is error model validdd"); }

            try
            {
                Console.WriteLine("try block called");
                model.TranslatedText = await _signLanguageService.PredictSignLanguageAsync(model.VideoFile);

                return View("Result", model);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                _logger.LogError(ex, "Error processing video upload");
                ModelState.AddModelError("", $"An error occurred while processing your video: {ex.Message}");
                return View(model);
            }
        }

        public IActionResult Result(VideoUpload model)
        {
            Console.WriteLine($"[{model.VideoFile}]");
            return View(model);
        }
    }
}