
# Transcription Audio en Texte

Ce projet fournit un script pour transcrire des fichiers audio en texte en utilisant le modèle Whisper d'OpenAI. Il supporte divers formats audio (WAV, MP3, M4A, OGG, FLAC) et permet de spécifier la langue de l'audio pour une transcription plus précise. Il offre également une option pour détecter automatiquement la langue de l'audio.

## Prérequis

- Python 3.7 ou supérieur
- `torch`
- `whisper`
- `tqdm`

Vous pouvez installer les packages nécessaires avec la commande suivante :

```sh
pip install torch whisper tqdm
```

## Utilisation

1. **Clonez le dépôt :**

    ```sh
    git clone https://github.com/tanguy-kabore/audio-to-text.git
    cd audio-to-text
    ```

2. **Exécutez le script :**

    ```sh
    python speech_to_text.py
    ```

3. **Suivez les instructions :**

    - Entrez le chemin vers un fichier audio (WAV, MP3, M4A, OGG, ou FLAC).
    - Entrez le code de langue (par exemple, `en`, `fr`, `de`).
    - Entrez le chemin vers le fichier de sortie pour le texte.
    - Choisissez si vous souhaitez détecter la langue et transcrire (`oui` ou `non`).

## Explication du Code

Le script fournit les fonctions suivantes :

### `transcribe_audio(file_path, language)`

Transcrit le fichier audio à l'emplacement `file_path` en utilisant la langue spécifiée `language`. Il charge l'audio, le traite, crée un spectrogramme log-Mel et effectue la transcription.

### `detect_language_and_transcribe(file_path, language)`

Détecte la langue du fichier audio à l'emplacement `file_path`, puis le transcrit en utilisant la langue détectée. Il effectue des étapes similaires à celles de `transcribe_audio` avec des étapes supplémentaires pour détecter la langue.

### `write_transcription_to_file(text, output_path)`

Écrit le texte transcrit dans un fichier à l'emplacement `output_path`.

### Script Principal

Le script principal demande à l'utilisateur les chemins d'accès, la langue, et les options, puis appelle les fonctions appropriées pour transcrire l'audio et sauvegarder le résultat dans un fichier.

## Exemple

```sh
python speech_to_text.py
```

```plaintext
Please enter the path to an audio file (WAV, MP3, M4A, OGG, or FLAC): C:\Projects\Python_lab\audio-to-text\input.wav
Please enter the language code (e.g., en, fr, de): fr
Please enter the path to the output text file: C:\Projects\Python_lab\audio-to-text\output.txt
Do you want to detect the language and transcribe? (yes/no): no
```

## Gestion des Avertissements

Le script ignore certains avertissements spécifiques liés à l'utilisation de `torch.load` et au support de FP16 sur CPU pour assurer une exécution fluide sans avertissements inutiles.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour les détails.