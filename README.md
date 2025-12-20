# 🏥 Parkinson's Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Advanced AI-Powered Voice Analysis Platform for Parkinson's Disease Detection and Stage Classification

## 📋 Overview

This professional-grade diagnostic tool utilizes **15 voice biomarkers** and machine learning algorithms to detect Parkinson's Disease and classify disease progression stages. The system employs weighted voting mechanisms and advanced signal processing for accurate analysis.

### Key Features

- 🎤 **Real-time Audio Recording** - Record voice samples directly in the browser
- 📁 **Audio File Upload** - Support for WAV, MP3, and FLAC formats
- 🔬 **15 Voice Biomarkers Analysis** - Comprehensive feature extraction
- 🏥 **Disease Detection** - Binary classification (Healthy vs Parkinson's)
- 📈 **4-Stage Classification** - Early, Mild, Moderate, and Severe stages
- 🖼️ **Visual References** - Clinical and anatomical reference images
- 📊 **Detailed Reports** - Comprehensive analysis with recommendations

## 🧬 Voice Biomarkers Analyzed

### Jitter Features (4)
- **Jitter(%)** - Frequency variation percentage
- **Jitter:RAP** - Relative Average Perturbation
- **Jitter:PPQ5** - Five-point Period Perturbation Quotient
- **Jitter:DDP** - Difference of Differences of Periods

### Shimmer Features (6)
- **Shimmer** - Amplitude variation
- **Shimmer(dB)** - Amplitude variation in decibels
- **Shimmer:APQ3** - 3-point Amplitude Perturbation Quotient
- **Shimmer:APQ5** - 5-point Amplitude Perturbation Quotient
- **Shimmer:APQ11** - 11-point Amplitude Perturbation Quotient
- **Shimmer:DDA** - Difference of Differences of Amplitudes

### Harmonicity Features (2)
- **NHR** - Noise-to-Harmonics Ratio
- **HNR** - Harmonics-to-Noise Ratio

### Nonlinear Features (3)
- **RPDE** - Recurrence Period Density Entropy
- **DFA** - Detrended Fluctuation Analysis
- **PPE** - Pitch Period Entropy

## 🚀 Installation

### Google Colab Setup

1. Upload the notebook to Google Colab
2. Mount Google Drive when prompted
3. Run all cells - dependencies will be installed automatically

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/parkinsons-detection.git
cd parkinsons-detection
```

2. **Install system dependencies** (Linux/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
parkinsons-detection/
├── parkinsons_detection.py    # Main application code
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
│
├── models/                     # Trained model files
│   ├── parkinsons_ensemble_model.pkl
│   ├── feature_scaler.pkl
│   └── feature_info.json
│
├── images/                     # Stage reference images
│   ├── stage1a.png
│   ├── stage1b.png
│   ├── stage2a.png
│   ├── stage2b.png
│   ├── stage3a.png
│   ├── stage3b.png
│   ├── stage4a.png
│   └── stage4b.png
│
└── results/                    # Analysis results (generated)
    └── [timestamp]_analysis.json
```

## 💻 Usage

### Running in Google Colab

1. Upload the main Python file to Colab
2. Run the notebook
3. Click on the generated Gradio link
4. Use the interface to analyze voice samples

### Running Locally

```bash
python parkinsons_detection.py
```

### Using the Interface

#### Tab 1: Disease Detection

1. **Record Audio** or **Upload File**
   - Record: Click microphone button and speak for 5-10 seconds
   - Upload: Select a WAV/MP3/FLAC file
   
2. **Configure Settings**
   - Choose noise reduction strength (light/medium/heavy)
   
3. **Analyze**
   - Click "🔬 Analyze Audio" button
   - View disease detection results
   - Review extracted features

#### Tab 2: Stage Classification

1. **Automatic Analysis**
   - After disease detection, stage classification runs automatically
   
2. **View Results**
   - Stage information (1-4)
   - Clinical symptoms
   - Voice characteristics
   - Medical recommendations
   - Reference images
   - Feature contribution analysis

### Recording Tips

- 🔇 Find a quiet environment
- 🗣️ Speak clearly for 5-10 seconds
- 🎵 Sustain a vowel sound (e.g., "Aaaah")
- 📊 Maintain consistent volume

## 🏗️ Model Architecture

### Preprocessing Pipeline

1. **DC Offset Removal** - Eliminates DC bias
2. **Bandpass Filter** - 80-4000 Hz for Parkinson's features
3. **Spectral Subtraction** - Advanced noise reduction
4. **RMS Normalization** - Consistent amplitude
5. **Voice Activity Detection** - Silence trimming

### Classification Models

- **Disease Detection**: Ensemble model (Random Forest/XGBoost)
- **Stage Classification**: Weighted voting system with 15 features

### Weighting System

| Weight | Features |
|--------|----------|
| 5 (High) | NHR, HNR, RPDE |
| 4 | PPE, DFA, Jitter(%), Shimmer |
| 3 | Shimmer:APQ3, APQ5, Jitter:RAP, PPQ5 |
| 2 | Shimmer(dB), DDA, Jitter:DDP |
| 1 | Shimmer:APQ11 |

## 📊 Stage Classification

### Stage 1: Early Stage (LOW Severity)
- Minimal voice alterations
- Unilateral motor symptoms
- Early intervention recommended

### Stage 2: Mild Stage (MILD Severity)
- Noticeable voice instability
- Bilateral motor involvement
- Speech therapy initiation

### Stage 3: Moderate Stage (MODERATE Severity)
- Significant voice monotony
- Moderate motor disability
- Multidisciplinary care required

### Stage 4: Severe Stage (SEVERE Severity)
- Severe voice degradation
- Complete motor dependency
- Palliative care consideration

## 🔬 Scientific Background

This system is based on research in voice biomarker analysis for Parkinson's Disease detection, utilizing established acoustic features documented in peer-reviewed medical literature.

## ⚠️ Disclaimer

**IMPORTANT**: This tool is for **research and educational purposes only**. It is **NOT** a replacement for professional medical diagnosis.

- Always consult qualified healthcare professionals
- Clinical diagnosis requires comprehensive neurological evaluation
- Results should be interpreted by movement disorder specialists
- Use as a screening tool, not definitive diagnosis

## 🛠️ Configuration

### Model Directory Setup

In Google Colab, directories are set to:

```python
MODEL_DIR = '/content/drive/MyDrive/ParkinsonsModel'
OUTPUT_DIR = '/content/drive/MyDrive/ParkinsonsResults'
IMAGES_DIR = '/content/drive/MyDrive/ParkinsonsImages'
```

### Required Model Files

Place these files in the MODEL_DIR:
- `parkinsons_ensemble_model.pkl` - Trained classification model
- `feature_scaler.pkl` - Feature normalization scaler
- `feature_info.json` - Feature metadata

Place stage images (stage1a.png through stage4b.png) in IMAGES_DIR.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- Parkinson's Disease research community
- Open-source audio processing libraries
- Gradio team for the excellent UI framework
- Medical professionals for domain expertise

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🔄 Version History

- **v1.0.0** (2024-12-20)
  - Initial release
  - 15 feature extraction
  - 4-stage classification
  - Premium medical interface

---

<div align="center">

**Made with ❤️ for Parkinson's Disease Research**

</div>
