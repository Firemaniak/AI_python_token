import whisper


model = whisper.load_model("base")


result = model.transcribe("audio1.mp3")

print("Распознанный текст:")
print(result["text"])
print("Определённый язык:", result["language"])