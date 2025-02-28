import pandas as pd

# Leggi il file .txt; usa il separatore giusto (qui ',' se i campi sono separati da virgola)
df = pd.read_csv(r'C:\Users\Mauro\Downloads\MindBigDataVisualMnist2022-Cap64v0.016M\MindBigData64_Mnist2022-EEGv0.016.txt', sep=',', header=None, engine='python')

# Salva il DataFrame in un file CSV (senza indice)
df.to_csv('output.csv', index=False, header=False)