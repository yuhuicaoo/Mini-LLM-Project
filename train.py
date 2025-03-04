

with open('kendrick_lamar_lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
print(text[:1000])