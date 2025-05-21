using System.Text.Json;

namespace Sign_Language.Services
{
    public class SignLanguageService
    {
        private readonly HttpClient _httpClient;

        public SignLanguageService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<string> PredictSignLanguageAsync(IFormFile videoFile)
        {
            using var form = new MultipartFormDataContent();
            using var streamContent = new StreamContent(videoFile.OpenReadStream());
            
            streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(videoFile.ContentType);
            form.Add(streamContent, "file", videoFile.FileName);

            var response = await _httpClient.PostAsync("predict", form);
            var jsonString = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode)
            {
                var result = JsonSerializer.Deserialize<PredictionResponse>(jsonString);
                return result.predicted_class;
            }
            else
            {
                var error = JsonSerializer.Deserialize<ErrorResponse>(jsonString);
                throw new Exception($"API Error: {error.error}");
            }
        }

        public async Task<string> PredictSignLanguageAsync(string videoFilePath)
        {
            using var form = new MultipartFormDataContent();
            using var fileStream = File.OpenRead(videoFilePath);
            using var streamContent = new StreamContent(fileStream);
            
            streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("video/mp4");
            form.Add(streamContent, "file", Path.GetFileName(videoFilePath));

            var response = await _httpClient.PostAsync("predict", form);
            var jsonString = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode)
            {
                var result = JsonSerializer.Deserialize<PredictionResponse>(jsonString);
                return result.predicted_class;
            }
            else
            {
                var error = JsonSerializer.Deserialize<ErrorResponse>(jsonString);
                throw new Exception($"API Error: {error.error}");
            }
        }
    }

    public class PredictionResponse
    {
        public string predicted_class { get; set; }
    }

    public class ErrorResponse
    {
        public string error { get; set; }
    }
}