# 🇵🇰 Pakistan Sign Language Detection & Translation System

A web-based application that detects and translates **Pakistan Sign Language (PSL)** gestures into text using machine learning. This project aims to improve communication for the hearing-impaired community in Pakistan by offering an accessible and region-specific translation system.

## 💡 Project Overview

This application allows users to upload a video of a PSL gesture. The system uses trained deep learning models (such as I3D and CNN-LSTM) to recognize and translate the sign into readable text. It is developed using **ASP.NET Core MVC (C#)** and focuses on **offline PSL recognition**, targeting accessibility in schools, clinics, and public services.

---

## 🔍 Features

- Upload PSL gesture videos and receive translated text
- Region-specific support for **Pakistan Sign Language**
- Uses custom-collected PSL dataset
- Deep learning-based gesture recognition using I3D
- Web interface built with **ASP.NET Core MVC**
- Easy-to-use UI and adaptable deployment
- Optimized for small datasets without data augmentation

---

## 🧰 Tech Stack

- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** ASP.NET Core MVC (C#)
- **Machine Learning:** I3D, CNN, LSTM (trained separately in Python)
- **Database (optional):** SQL Server for user/session data

---

## 📁 Project Structure

<pre> ```bash /PSL-Translation-App │ ├── Controllers/ ├── Views/ │ ├── Home/ │ └── Shared/ ├── Models/ ├── wwwroot/ │ ├── css/ │ └── js/ ├── MLModels/ # Trained ML model files ├── demo/ # Demo videos (if any) ├── Program.cs ├── Startup.cs └── README.md ``` </pre>

## 🧠 How It Works

1. User uploads a video of a PSL word.
2. The backend sends the video to the I3D model.
3. The model processes the video and returns the recognized sign.
4. The result is displayed on the website.

---

## 📦 Setup Instructions

### Prerequisites
- [.NET 6 SDK](https://dotnet.microsoft.com/en-us/download)
- Visual Studio 2022 or VS Code
- Python (only needed for training models, not running the web app)

### Steps
```bash
git clone https://github.com/your-username/psl-translation-system.git
cd psl-translation-system

Open the solution in Visual Studio.

Restore packages and build the solution.

Place your trained model files inside /MLModels/.

Run the application.

🧪 Dataset
We collected our own Pakistan Sign Language dataset consisting of video samples recorded manually. Due to the absence of a large public PSL dataset, this custom dataset played a crucial role in training and testing our models.

```
👥 Authors
Developed by students at Punjab University College of Information Technology (PUCIT) under the supervision of Dr. Muhammad Farooq.

Nabiha Hamid – BCSF21M026

Ayman Ashraf – BCSF21M031

Maryam Rasool Qaisrani – BCSF21M055

📃 License
This project is for academic and research purposes only.
For reuse or citations, please contact the authors.
