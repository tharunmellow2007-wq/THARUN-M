import os
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import librosa
import soundfile as sf
import scipy.signal as signal
from scipy.signal import butter
import pickle
import json
import streamlit as st
from datetime import datetime
from PIL import Image
import warnings
import tempfile
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Parkinson's Disease Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
    }
    .main-header p {
        color: #e0e7ff;
        margin-top: 12px;
        font-size: 1.2em;
    }
    .info-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0ea5e9;
        margin-bottom: 25px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 17px;
        font-weight: 600;
        padding: 14px 28px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION - RENDER OPTIMIZED
# ============================================================================

# Use environment variable for model directory or default to ./models
MODEL_DIR = os.getenv('MODEL_DIR', './models')
OUTPUT_DIR = './results'
IMAGES_DIR = './images'

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Initialize session state
if 'current_features' not in st.session_state:
    st.session_state.current_features = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_info' not in st.session_state:
    st.session_state.feature_info = None

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def remove_dc_offset(audio):
    """Remove DC bias from audio signal"""
    return audio - np.mean(audio)

def bandpass_filter(audio, sr, lowcut, highcut, order=4):
    """Butterworth Bandpass Filter"""
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    return filtered_audio

def estimate_noise_profile(audio, sr, noise_duration=0.5):
    """Estimate noise characteristics from initial portion"""
    noise_samples = int(noise_duration * sr)
    if len(audio) < noise_samples:
        noise_samples = max(len(audio) // 10, 100)
    noise_segment = audio[:noise_samples]
    stft_noise = librosa.stft(noise_segment, n_fft=2048, hop_length=512)
    noise_profile = np.mean(np.abs(stft_noise), axis=1)
    return noise_profile

def spectral_subtraction(audio, sr, noise_profile=None, strength='medium'):
    """Spectral Subtraction (Boll, 1979)"""
    params = {
        'light': {'alpha': 1.0, 'beta': 0.1},
        'medium': {'alpha': 2.0, 'beta': 0.05},
        'heavy': {'alpha': 3.0, 'beta': 0.02}
    }
    alpha = params[strength]['alpha']
    beta = params[strength]['beta']

    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    if noise_profile is None:
        noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
    else:
        noise_profile = noise_profile.reshape(-1, 1)

    magnitude_clean = magnitude - alpha * noise_profile
    magnitude_clean = np.maximum(magnitude_clean, beta * magnitude)

    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=512)
    return audio_clean

def normalize_rms(audio, target_rms=0.1):
    """RMS Normalization"""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        scaling_factor = target_rms / current_rms
        audio = audio * scaling_factor
    max_val = np.max(np.abs(audio))
    if max_val > 0.99:
        audio = audio * (0.99 / max_val)
    return audio

def trim_silence(audio, sr, top_db=20):
    """Voice Activity Detection & Trimming"""
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=512)
    if len(intervals) == 0:
        return audio, []

    voiced_segments = []
    for start, end in intervals:
        voiced_segments.append(audio[start:end])

    silence_duration = int(0.05 * sr)
    silence = np.zeros(silence_duration)

    trimmed_audio = []
    for i, segment in enumerate(voiced_segments):
        trimmed_audio.append(segment)
        if i < len(voiced_segments) - 1:
            trimmed_audio.append(silence)

    trimmed_audio = np.concatenate(trimmed_audio)
    return trimmed_audio, intervals

def preprocess_parkinsons_audio(audio_path=None, audio_data=None, sr=None,
                                noise_reduction_strength='medium',
                                preserve_parkinsons_features=True):
    """Comprehensive audio preprocessing for Parkinson's voice analysis"""
    if audio_path:
        y, original_sr = librosa.load(audio_path, sr=None, mono=True)
    elif audio_data:
        y, original_sr = audio_data
    else:
        raise ValueError("Either audio_path or audio_data must be provided")

    target_sr = sr if sr else 16000
    if original_sr != target_sr:
        y = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)

    y = remove_dc_offset(y)
    noise_profile = estimate_noise_profile(y, target_sr)
    y = spectral_subtraction(y, target_sr, noise_profile, strength=noise_reduction_strength)

    if preserve_parkinsons_features:
        y = bandpass_filter(y, target_sr, lowcut=80, highcut=4000, order=4)
    else:
        y = bandpass_filter(y, target_sr, lowcut=300, highcut=3400, order=4)

    target_rms = 0.1 if preserve_parkinsons_features else 0.15
    y = normalize_rms(y, target_rms=target_rms)
    y, intervals = trim_silence(y, target_sr, top_db=20)

    min_duration = 1.0
    if len(y) / target_sr < min_duration:
        y = np.pad(y, (0, int(min_duration * target_sr) - len(y)))

    quality_metrics = {
        'duration': len(y) / target_sr,
        'sample_rate': target_sr
    }

    return y, target_sr, quality_metrics

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_jitter_shimmer_features(voice):
    """Extract jitter and shimmer features using Parselmouth"""
    try:
        point_process = call(voice, "To PointProcess (periodic, cc)", 75, 600)

        jitter_percent = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
        jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = jitter_rap * 3

        shimmer_local = call([voice, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = call([voice, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = call([voice, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = call([voice, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = call([voice, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = shimmer_apq3 * 3

        return {
            'Jitter(%)': jitter_percent,
            'Jitter:RAP': jitter_rap,
            'Jitter:PPQ5': jitter_ppq5,
            'Jitter:DDP': jitter_ddp,
            'Shimmer': shimmer_local,
            'Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': shimmer_apq3,
            'Shimmer:APQ5': shimmer_apq5,
            'Shimmer:APQ11': shimmer_apq11,
            'Shimmer:DDA': shimmer_dda
        }
    except Exception as e:
        raise Exception(f"Error extracting jitter/shimmer: {e}")

def extract_harmonicity_features(voice):
    """Extract NHR and HNR features"""
    try:
        harmonicity = call(voice, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 1.0 / (hnr + 1e-6) if hnr > 0 else 1.0

        return {
            'NHR': nhr,
            'HNR': hnr
        }
    except Exception as e:
        raise Exception(f"Error extracting harmonicity: {e}")

def extract_nonlinear_features(y, sr):
    """Extract RPDE, DFA, and PPE features"""
    try:
        spec = np.abs(librosa.stft(y))
        spec_norm = spec / (np.sum(spec, axis=0) + 1e-6)
        spec_entropy = -np.sum(spec_norm * np.log2(spec_norm + 1e-6), axis=0)
        rpde = np.mean(spec_entropy) / 10.0

        autocorr = librosa.autocorrelate(y)
        dfa = np.sum(autocorr[:min(100, len(autocorr))]) / len(autocorr)

        f0 = librosa.yin(y, fmin=75, fmax=600, sr=sr)
        f0_valid = f0[f0 > 0]
        ppe = np.std(f0_valid) / (np.mean(f0_valid) + 1e-6) if len(f0_valid) > 0 else 0.0

        return {
            'RPDE': rpde,
            'DFA': dfa,
            'PPE': ppe
        }
    except Exception as e:
        raise Exception(f"Error extracting nonlinear features: {e}")

def extract_all_features(audio_path=None, audio_data=None, sr=None):
    """Extract all 15 required features"""
    try:
        # Use tempfile for cross-platform compatibility
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            temp_wav = tmp_file.name
        
        if audio_path:
            audio_array, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
            sf.write(temp_wav, audio_array, sample_rate)
            voice = parselmouth.Sound(temp_wav)
        elif audio_data:
            audio_array, sample_rate = audio_data
            sf.write(temp_wav, audio_array, sample_rate)
            voice = parselmouth.Sound(temp_wav)
        else:
            raise ValueError("Either audio_path or audio_data must be provided")

        jitter_shimmer = extract_jitter_shimmer_features(voice)
        harmonicity = extract_harmonicity_features(voice)
        nonlinear = extract_nonlinear_features(audio_array, sample_rate)

        # Clean up temp file
        try:
            os.remove(temp_wav)
        except:
            pass

        all_features = {
            **jitter_shimmer,
            **harmonicity,
            **nonlinear
        }

        return all_features

    except Exception as e:
        raise Exception(f"Feature extraction failed: {e}")

# ============================================================================
# STAGE CLASSIFICATION
# ============================================================================

def classify_parkinsons_stage_weighted(features):
    """15-feature weighted voting stage classification"""

    weights = {
        'NHR': 5, 'HNR': 5, 'RPDE': 5,
        'PPE': 4, 'DFA': 4, 'Jitter(%)': 4, 'Shimmer': 4,
        'Shimmer:APQ3': 3, 'Shimmer:APQ5': 3, 'Jitter:RAP': 3, 'Jitter:PPQ5': 3,
        'Shimmer(dB)': 2, 'Shimmer:DDA': 2, 'Jitter:DDP': 2,
        'Shimmer:APQ11': 1
    }

    thresholds = {
        'Jitter(%)': [(0, 0.005), (0.005, 0.008), (0.008, 0.012), (0.012, float('inf'))],
        'Jitter:RAP': [(0, 0.003), (0.003, 0.005), (0.005, 0.008), (0.008, float('inf'))],
        'Jitter:PPQ5': [(0, 0.003), (0.003, 0.005), (0.005, 0.008), (0.008, float('inf'))],
        'Jitter:DDP': [(0, 0.009), (0.009, 0.015), (0.015, 0.024), (0.024, float('inf'))],
        'Shimmer': [(0, 0.035), (0.035, 0.045), (0.045, 0.055), (0.055, float('inf'))],
        'Shimmer(dB)': [(0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, float('inf'))],
        'Shimmer:APQ3': [(0, 0.018), (0.018, 0.023), (0.023, 0.028), (0.028, float('inf'))],
        'Shimmer:APQ5': [(0, 0.019), (0.019, 0.024), (0.024, 0.029), (0.029, float('inf'))],
        'Shimmer:APQ11': [(0, 0.021), (0.021, 0.026), (0.026, 0.031), (0.031, float('inf'))],
        'Shimmer:DDA': [(0, 0.054), (0.054, 0.069), (0.069, 0.084), (0.084, float('inf'))],
        'NHR': [(0, 0.04), (0.04, 0.06), (0.06, 0.08), (0.08, float('inf'))],
        'HNR': [(20.0, float('inf')), (15.0, 20.0), (10.0, 15.0), (0, 10.0)],
        'RPDE': [(0, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, float('inf'))],
        'DFA': [(0.75, float('inf')), (0.65, 0.75), (0.55, 0.65), (0, 0.55)],
        'PPE': [(0, 0.18), (0.18, 0.22), (0.22, 0.28), (0.28, float('inf'))]
    }

    stage_votes = {1: 0, 2: 0, 3: 0, 4: 0}
    feature_contributions = {}
    total_weight = sum(weights.values())

    for feature_name, value in features.items():
        if feature_name not in thresholds:
            continue

        weight = weights.get(feature_name, 1)
        stage_ranges = thresholds[feature_name]

        voted_stage = None
        for stage_idx, (low, high) in enumerate(stage_ranges, start=1):
            if low <= value < high or (stage_idx == 4 and value >= low):
                voted_stage = stage_idx
                break

        if voted_stage:
            stage_votes[voted_stage] += weight
            feature_contributions[feature_name] = {
                'value': value,
                'voted_stage': voted_stage,
                'weight': weight
            }

    max_votes = max(stage_votes.values())
    winning_stages = [stage for stage, votes in stage_votes.items() if votes == max_votes]
    predicted_stage = max(winning_stages)
    confidence = (stage_votes[predicted_stage] / total_weight) * 100

    stage_info = {
        1: {
            'name': 'Early Stage',
            'severity': 'LOW',
            'color': '#10b981',
            'bg_color': '#d1fae5',
            'description': 'Initial manifestation of Parkinson\'s Disease with minimal voice alterations and early motor symptoms.',
            'symptoms': [
                'Subtle tremor in one hand or limb at rest',
                'Mild stiffness or slowness in movements',
                'Slight changes in posture or facial expression',
                'Reduced arm swing on one side while walking',
                'Early sleep disturbances or fatigue'
            ],
            'characteristics': [
                'Minimal voice instability with slight pitch variations',
                'Slight reduction in voice volume (hypophonia)',
                'Early changes in pitch control and modulation',
                'Unilateral motor symptoms affecting one side',
                'Symptoms do not significantly interfere with daily activities'
            ],
            'recommendations': [
                'Schedule comprehensive neurological evaluation with movement disorder specialist',
                'Begin regular monitoring program with quarterly follow-up assessments',
                'Implement lifestyle modifications: regular exercise, balanced diet, adequate sleep',
                'Explore speech therapy evaluation for voice preservation techniques',
                'Join support groups for education and emotional support'
            ]
        },
        2: {
            'name': 'Mild Stage',
            'severity': 'MILD',
            'color': '#f59e0b',
            'bg_color': '#fef3c7',
            'description': 'Progressive symptoms with noticeable voice instability and bilateral motor involvement.',
            'symptoms': [
                'Tremor affecting both sides of the body',
                'Noticeable rigidity and bradykinesia (slowness)',
                'Mild balance issues and postural instability',
                'Reduced facial expressions (masked face)',
                'Difficulty with fine motor tasks like writing'
            ],
            'characteristics': [
                'Noticeable voice tremor and vocal instability',
                'Reduced voice volume requiring effort to speak loudly',
                'Bilateral motor symptoms affecting both body sides',
                'Mild motor disability impacting daily tasks',
                'Speech may become monotonous with reduced prosody'
            ],
            'recommendations': [
                'Initiate pharmacological treatment with dopaminergic medications as prescribed',
                'Active participation in speech therapy for voice strengthening (LSVT LOUD)',
                'Regular physical therapy to maintain mobility and prevent muscle stiffness',
                'Occupational therapy for adaptive strategies in daily activities',
                'Continue regular monitoring with neurologist every 2-3 months'
            ]
        },
        3: {
            'name': 'Moderate Stage',
            'severity': 'MODERATE',
            'color': '#ef4444',
            'bg_color': '#fee2e2',
            'description': 'Significant functional impairment with clear monotonicity, breathiness, and moderate motor disability.',
            'symptoms': [
                'Significant slowness and difficulty initiating movements',
                'Frequent freezing episodes during walking',
                'Notable balance problems with increased fall risk',
                'Difficulty swallowing (dysphagia)',
                'Cognitive changes and mood disturbances'
            ],
            'characteristics': [
                'Significant voice monotony with flattened prosody',
                'Breathy voice quality due to incomplete glottal closure',
                'Moderate motor impairment limiting independence',
                'Difficulty with voice projection and sustained phonation',
                'Speech intelligibility may be compromised in noisy environments'
            ],
            'recommendations': [
                'Comprehensive multidisciplinary care plan with neurology, PT, OT, and SLP',
                'Medication adjustment and optimization by movement disorder specialist',
                'Intensive speech therapy focusing on articulation and voice projection',
                'Consider assistive devices for mobility (walker, cane) and communication',
                'Regular swallowing assessments and dietary modifications for safety'
            ]
        },
        4: {
            'name': 'Severe Stage',
            'severity': 'SEVERE',
            'color': '#dc2626',
            'bg_color': '#fecaca',
            'description': 'Advanced disease with severe voice degradation, frequent breaks, and significant motor disability requiring assistance.',
            'symptoms': [
                'Severe mobility limitations, often wheelchair-dependent',
                'Frequent falls and inability to stand without support',
                'Marked cognitive impairment or dementia',
                'Severe difficulty swallowing with aspiration risk',
                'Complete dependence for activities of daily living'
            ],
            'characteristics': [
                'Severe voice degradation with marked hoarseness',
                'Frequent aphonic breaks during speech production',
                'Highly irregular pitch patterns and prosody',
                'Significant motor disability requiring full-time assistance',
                'Speech may be unintelligible requiring augmentative communication'
            ],
            'recommendations': [
                'Intensive comprehensive care with 24/7 caregiver support or skilled nursing',
                'Palliative care consultation for symptom management and quality of life',
                'Speech-language pathology for augmentative and alternative communication (AAC)',
                'Nutritional support with possible feeding tube consideration for safety',
                'Advanced care planning and discussion of end-of-life preferences'
            ]
        }
    }

    return {
        'stage': predicted_stage,
        'stage_name': stage_info[predicted_stage]['name'],
        'severity': stage_info[predicted_stage]['severity'],
        'color': stage_info[predicted_stage]['color'],
        'bg_color': stage_info[predicted_stage]['bg_color'],
        'description': stage_info[predicted_stage]['description'],
        'symptoms': stage_info[predicted_stage]['symptoms'],
        'characteristics': stage_info[predicted_stage]['characteristics'],
        'recommendations': stage_info[predicted_stage]['recommendations'],
        'confidence': confidence,
        'weighted_votes': stage_votes,
        'feature_contributions': feature_contributions,
        'total_weight': total_weight
    }

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load model components with caching"""
    try:
        model_path = os.path.join(MODEL_DIR, 'parkinsons_ensemble_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
        info_path = os.path.join(MODEL_DIR, 'feature_info.json')
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, info_path]):
            st.error(f"⚠️ Model files not found in {MODEL_DIR}. Please upload model files.")
            st.info("Required files: parkinsons_ensemble_model.pkl, feature_scaler.pkl, feature_info.json")
            return None, None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(info_path, 'r') as f:
            feature_info = json.load(f)
        
        return model, scaler, feature_info
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Parkinson's Disease Detection System</h1>
        <p>Advanced AI-Powered Voice Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p style="margin: 0; color: #0c4a6e; font-size: 1.05em; line-height: 1.8;">
            <strong>📌 System Overview:</strong> This professional-grade diagnostic tool utilizes <strong>15 voice biomarkers</strong> and
            machine learning algorithms to detect Parkinson's Disease and classify disease progression stages.
            The system employs weighted voting mechanisms and advanced signal processing for accurate analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    if st.session_state.model is None:
        with st.spinner("Loading AI models..."):
            model, scaler, feature_info = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.feature_info = feature_info
                st.success("✓ Model loaded successfully!")
            else:
                st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        noise_reduction = st.selectbox(
            "Noise Reduction Strength",
            ["light", "medium", "heavy"],
            index=1,
            help="Applies to recorded/uploaded audio"
        )
        
        st.markdown("---")
        st.markdown("""
        ### 🎙️ Recording Tips
        - Find a quiet environment
        - Speak clearly for 5-10 seconds
        - Sustain a vowel sound (e.g., "Aaaah")
        - Maintain consistent volume
        
        ### 📂 Supported Formats
        WAV, MP3, FLAC
        """)
        
        if st.button("🔄 Clear All", use_container_width=True):
            st.session_state.current_features = None
            st.session_state.current_prediction = None
            st.rerun()

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Disease Detection", "📈 Stage Classification"])

    # TAB 1: Disease Detection
    with tab1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #1e40af; margin: 0 0 10px 0;">🎤 Audio Input</h3>
            <p style="color: #1e3a8a; margin: 0;">Upload an audio file for analysis</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Upload a clear voice recording (5-10 seconds recommended)"
        )

        if uploaded_file is not None:
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔬 Analyze Audio", type="primary", use_container_width=True):
                with st.spinner("Analyzing audio..."):
                    try:
                        # Use tempfile for cross-platform compatibility
                        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded_file.name)[1], delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_path = tmp_file.name
                        
                        # Extract features
                        features = extract_all_features(audio_path=temp_path)
                        
                        # Predict
                        feature_vector = [features[name] for name in st.session_state.feature_info['feature_names']]
                        feature_scaled = st.session_state.scaler.transform([feature_vector])
                        prediction = st.session_state.model.predict(feature_scaled)[0]
                        
                        # Store in session state
                        st.session_state.current_features = features
                        st.session_state.current_prediction = prediction
                        
                        # Display results
                        if prediction == 1:
                            st.error("⚠️ Parkinson's Disease Detected")
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #ef4444; margin-bottom: 20px;">
                                <h3 style="color: #991b1b; margin-top: 0;">⚠️ Parkinson's Disease Detected</h3>
                                <p style="color: #7f1d1d; font-size: 1.1em;">Voice biomarkers indicate presence of Parkinson's Disease</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.info("📋 Please proceed to the **Stage Classification** tab for detailed progression analysis.")
                        else:
                            st.success("✅ Healthy Voice Profile")
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 25px; border-radius: 15px; border-left: 6px solid #10b981; margin-bottom: 20px;">
                                <h3 style="color: #065f46; margin-top: 0;">✅ Healthy Voice Profile</h3>
                                <p style="color: #047857; font-size: 1.1em;">No significant indicators of Parkinson's Disease detected</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Display features
                        with st.expander("🔬 Detailed Feature Analysis", expanded=False):
                            features_df = pd.DataFrame([features]).T
                            features_df.columns = ['Value']
                            features_df.index.name = 'Feature'
                            st.dataframe(features_df.style.format("{:.6f}"), use_container_width=True)
                        
                        # Clean up
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"❌ Analysis Error: {str(e)}")
                        st.info("Please ensure the audio is clear and at least 3 seconds long.")
        else:
            st.info("👆 Please upload an audio file to begin analysis")

    # TAB 2: Stage Classification
    with tab2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #92400e; margin: 0 0 10px 0;">📈 Disease Progression Analysis</h3>
            <p style="color: #78350f; margin: 0; font-size: 1.05em;">
                Detailed stage classification based on weighted feature voting.
                The system classifies disease progression into four stages: Early, Mild, Moderate, and Severe.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.current_prediction is None:
            st.warning("⚠️ Please run disease detection in the first tab before stage classification")
        elif st.session_state.current_prediction == 0:
            st.success("✅ No Stage Classification Required")
            st.info("The audio sample was classified as **HEALTHY**. Stage classification is only performed when Parkinson's Disease is detected.")
        else:
            # Perform stage classification
            with st.spinner("Classifying disease stage..."):
                try:
                    stage_result = classify_parkinsons_stage_weighted(st.session_state.current_features)
                    
                    # Display stage header
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {stage_result['bg_color']} 0%, {stage_result['color']}20 100%); padding: 30px; border-radius: 15px; border-left: 6px solid {stage_result['color']}; margin-bottom: 20px;">
                        <h2 style="color: {stage_result['color']}; margin: 0 0 10px 0;">Stage {stage_result['stage']}: {stage_result['stage_name']}</h2>
                        <p style="color: {stage_result['color']}; font-size: 1.2em; margin: 0;"><strong>Severity: {stage_result['severity']}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Clinical Description
                    with st.container():
                        st.markdown("### 📝 Clinical Description")
                        st.write(stage_result['description'])
                    
                    # Symptoms and Characteristics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🩺 Clinical Symptoms")
                        for symptom in stage_result['symptoms']:
                            st.markdown(f"• {symptom}")
                    
                    with col2:
                        st.markdown("### 🔍 Voice Characteristics")
                        for char in stage_result['characteristics']:
                            st.markdown(f"• {char}")
                    
                    # Recommendations
                    st.markdown("### 💊 Medical Recommendations")
                    for rec in stage_result['recommendations']:
                        st.markdown(f"• {rec}")
                    
                    # Weighted Voting Analysis
                    st.markdown("### 📊 Weighted Voting Analysis")
                    
                    # Create voting visualization
                    vote_data = []
                    for stage, votes in sorted(stage_result['weighted_votes'].items()):
                        percentage = (votes / stage_result['total_weight']) * 100
                        vote_data.append({
                            'Stage': f"Stage {stage}",
                            'Votes': votes,
                            'Percentage': percentage
                        })
                    
                    vote_df = pd.DataFrame(vote_data)
                    st.dataframe(vote_df, use_container_width=True)
                    
                    # Feature contributions
                    with st.expander("📊 Feature Contribution Analysis", expanded=False):
                        contrib_data = []
                        for feat_name, contrib in sorted(stage_result['feature_contributions'].items(),
                                                        key=lambda x: x[1]['weight'], reverse=True):
                            contrib_data.append({
                                'Feature': feat_name,
                                'Value': f"{contrib['value']:.6f}",
                                'Voted Stage': contrib['voted_stage'],
                                'Weight': contrib['weight']
                            })
                        
                        contrib_df = pd.DataFrame(contrib_data)
                        st.dataframe(contrib_df, use_container_width=True)
                        
                        st.info("""
                        **Weighting System:**
                        - Weight 5 (High): NHR, HNR, RPDE
                        - Weight 4: PPE, DFA, Jitter(%), Shimmer
                        - Weight 3: Shimmer:APQ3, APQ5, Jitter:RAP, PPQ5
                        - Weight 2: Shimmer(dB), DDA, Jitter:DDP
                        - Weight 1: Shimmer:APQ11
                        """)
                    
                    # Load and display images
                    st.markdown("### 🖼️ Stage Visual References")
                    img_col1, img_col2 = st.columns(2)
                    
                    with img_col1:
                        img_path = os.path.join(IMAGES_DIR, f'stage{stage_result["stage"]}a.png')
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Clinical Reference Image 1", use_container_width=True)
                        else:
                            st.info("Image 1 not available")
                    
                    with img_col2:
                        img_path = os.path.join(IMAGES_DIR, f'stage{stage_result["stage"]}b.png')
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Clinical Reference Image 2", use_container_width=True)
                        else:
                            st.info("Image 2 not available")
                    
                except Exception as e:
                    st.error(f"❌ Stage Classification Error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 20px;">
        <p>🏥 Parkinson's Disease Detection System | Advanced AI-Powered Voice Analysis</p>
        <p style="font-size: 0.9em;">⚠️ This tool is for research purposes only. Always consult healthcare professionals for medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
