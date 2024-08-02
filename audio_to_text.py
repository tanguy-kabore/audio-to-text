import whisper
import warnings
import torch
from tqdm import tqdm

# Filtrer les avertissements spécifiques
warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

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

    # Spécifier la langue de l'audio
    language = input("Please enter the language code (e.g., en, fr, de): ").strip()

    # Chemin vers le fichier de sortie
    output_path = input("Please enter the path to the output text file: ").strip()

    # Option pour détecter la langue et transcrire
    option = input("Do you want to detect the language and transcribe? (yes/no): ").strip().lower()
    
    try:
        if option == "yes":
            text = detect_language_and_transcribe(audio_path, language)
        else:
            text = transcribe_audio(audio_path, language)
        
        write_transcription_to_file(text, output_path)
        print(f"Transcription written to {output_path}")
    except Exception as e:
        print(f"Error during transcription: {e}")