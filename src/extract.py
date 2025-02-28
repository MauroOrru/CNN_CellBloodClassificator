import pandas as pd

# filepath: /c:/Users/Mauro/Desktop/Mauro/Universita/AI/Progetto/Dataset/test.csv
input_file = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\test.csv"
output_file = r"C:\Users\Mauro\Desktop\Mauro\Universita\AI\Progetto\Dataset\test_raw.csv"

# Legge il dataset completo
df = pd.read_csv(input_file)

# Seleziona le righe 2 e 3 (usando indici 1 e 2)
df_subset = df.iloc[1:]

# Salva il nuovo dataset con solo le righe selezionate
df_subset.to_csv(output_file, index=False, header=True)

print(f"Dataset creato in: {output_file}")