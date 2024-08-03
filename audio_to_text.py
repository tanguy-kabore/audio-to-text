import whisper
import warnings
import torch
import noisereduce as nr
from tqdm import tqdm
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import os

# Filtrer les avertissements spécifiques
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

def reduce_noise(file_path):
    # Charger l'audio avec pydub
    audio = AudioSegment.from_file(file_path)
    
    # Convertir en tableau numpy
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Réduire le bruit
    reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)
    
    # Créer un nouveau segment audio
    reduced_audio = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    
    # Sauvegarder l'audio traité
    cleaned_file_path = "cleaned_" + os.path.basename(file_path)
    reduced_audio.export(cleaned_file_path, format="wav")
    
    return cleaned_file_path

def transcribe_audio(file_path, language):
    # Charger le modèle Whisper
    model = whisper.load_model("base")

    # Lire et traiter l'audio en utilisant tqdm pour afficher la progression
    print("Loading and processing audio...")
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    # Créer un spectrogramme log-Mel et le déplacer sur le même appareil que le modèle
    print("Creating log-Mel spectrogram...")
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Transcrire l'audio avec la langue spécifiée
    print("Transcribing audio...")
    result = model.transcribe(file_path, language=language)
    return result["text"]

def detect_language_and_transcribe(file_path, language):
    # Charger le modèle Whisper
    model = whisper.load_model("base")

    # Charger l'audio et le découper pour qu'il corresponde à 30 secondes
    print("Loading and processing audio...")
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    # Créer un spectrogramme log-Mel et le déplacer sur le même appareil que le modèle
    print("Creating log-Mel spectrogram...")
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Détecter la langue parlée
    print("Detecting language...")
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Décoder l'audio
    print("Decoding audio...")
    options = whisper.DecodingOptions(language=language)
    result = whisper.decode(model, mel, options)

    # Retourner le texte reconnu
    return result.text

def write_transcription_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    # Chemin vers le fichier audio
    audio_path = input("Please enter the path to an audio file (WAV, MP3, M4A, OGG, or FLAC): ").strip()

    # Réduire le bruit de l'audio
    cleaned_audio_path = reduce_noise(audio_path)

    # Spécifier la langue de l'audio
    language = input("Please enter the language code (e.g., en, fr, de): ").strip()

    # Chemin vers le fichier de sortie
    output_path = input("Please enter the path to the output text file: ").strip()

    # Option pour détecter la langue et transcrire
    option = input("Do you want to detect the language and transcribe? (yes/no): ").strip().lower()
    
    try:
        if option == "yes":
            text = detect_language_and_transcribe(cleaned_audio_path, language)
        else:
            text = transcribe_audio(cleaned_audio_path, language)
        
        write_transcription_to_file(text, output_path)
        print(f"Transcription written to {output_path}")
    except Exception as e:
        print(f"Error during transcription: {e}")
