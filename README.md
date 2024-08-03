# Projet Audio-to-Text

Ce projet utilise le modèle Whisper pour transcrire des fichiers audio en texte, avec une option pour détecter automatiquement la langue parlée dans l'audio. Avant la transcription, le fichier audio est traité pour supprimer les bruits de fond.

## Prérequis

- Python 3.7+
- `whisper`
- `torch`
- `tqdm`
- `pydub`

Vous pouvez installer les dépendances nécessaires avec pip :

```bash
pip install whisper torch tqdm pydub
```

## Utilisation

1. Clonez ce dépôt :

```bash
git clone https://github.com/tanguy-kabore/audio-to-text.git
cd audio-to-text
```

2. Exécutez le script `audio_to_text.py` :

```bash
python audio_to_text.py
```

3. Suivez les instructions pour fournir le chemin vers le fichier audio, la langue de l'audio et le chemin pour le fichier de sortie.

## Exemple

Lors de l'exécution du script, vous serez invité à entrer le chemin vers un fichier audio (WAV, MP3, M4A, OGG ou FLAC), la langue de l'audio (par exemple, en, fr, de) et le chemin vers le fichier texte de sortie. Vous aurez également la possibilité de détecter automatiquement la langue parlée dans l'audio.

## Fonctionnalités

- Amélioration de la qualité de l'audio en supprimant les bruits de fond.
- Transcription de l'audio dans la langue spécifiée.
- Détection automatique de la langue parlée dans l'audio.
- Sauvegarde de la transcription dans un fichier texte.

## Avertissements

- Ce projet utilise `pydub` et `ffmpeg` pour traiter l'audio. Assurez-vous que `ffmpeg` est installé et accessible depuis votre PATH.
- Les avertissements spécifiques liés à `torch` sont filtrés pour une exécution plus fluide.

## Contributions

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.