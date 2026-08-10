import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

def train_classifier():
    print("🚀 Inicjalizacja treningu AI Guardrail Classifier...")
    
    # --- Sprawdzenie GPU ---
    if torch.cuda.is_available():
        print(f"✅ Znaleziono kartę graficzną: {torch.cuda.get_device_name(0)}")
        print("Trening odbędzie się na GPU! 🔥")
    else:
        print("⚠️ UWAGA: PyTorch nie widzi karty graficznej (CUDA). Trening działa na CPU!")
        
    # 1. Ścieżki
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "guardrail_dataset.csv")
    output_model_path = os.path.join(current_dir, "model_weights")

    if not os.path.exists(dataset_path):
        print("❌ Nie znaleziono datasetu! Uruchom najpierw dataset_builder.py")
        return

    # 2. Ładowanie danych przy pomocy biblioteki Pandas i konwersja do formatu HuggingFace
    df = pd.read_csv(dataset_path)
    # Tasujemy zbiór danych (shuffle) by upewnić się, że model nie uczy się "po kolei"
    df = df.sample(frac=1).reset_index(drop=True) 
    hf_dataset = Dataset.from_pandas(df)
    
    # 3. Wybór Modelu (Optymalny dla RTX 3070 8GB VRAM)
    # Używamy DistilBERT - jest to lżejsza, szybsza wersja słynnego BERTa od Google.
    # Użycie wersji 'uncased' oznacza, że model ignoruje wielkość liter (wszystko traktuje jako małe),
    # co oszczędza pamięć.
    model_name = "distilbert-base-uncased"
    
    print(f"📦 Pobieranie / ładowanie tokenizera dla modelu: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"🤖 Pobieranie / ładowanie architekutry modelu z 3 etykietami klasyfikacji...")
    # num_labels=3 oznacza nasze 3 klasy: 0=Benign, 1=Prompt Injection, 2=Toxic
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # 4. Tokenizacja
    # Modele AI nie rozumieją tekstu, tylko liczby (tokeny). 
    # Ta funkcja przekształca nasze stringi na wektory liczbowe o stałej długości (max_length=128).
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    print("🔤 Tokenizacja datasetu...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    
    # Do celów testowych używamy całego zbioru do treningu
    # (w prawdziwym projekcie podzielilibyśmy na train i eval)
    train_dataset = tokenized_datasets
    
    # 5. Konfiguracja parametrów treningowych
    print("⚙️ Konfiguracja hiperparametrów dla RTX 3070...")
    training_args = TrainingArguments(
        output_dir=os.path.join(current_dir, "checkpoints"),
        num_train_epochs=5,                  # Ile razy model ma "zobaczyć" cały dataset
        per_device_train_batch_size=16,      # Idealny rozmiar partii (batch) pod 8GB VRAM
        save_strategy="epoch",               # Zapisuj wagi po każdej epoce
        logging_dir=os.path.join(current_dir, "logs"),
        logging_steps=5,
        learning_rate=2e-5,                  # Szybkość uczenia - mała wartość (fine-tuning) zapobiega "zapomnieniu" pierwotnej wiedzy modelu
        fp16=True,                           # Mixed Precision: Wymusza użycie pamięci 16-bit zamiast 32-bit na karcie RTX, potężnie oszczędza VRAM i przyspiesza!
        report_to="none"                     # Wyłączamy raportowanie do zewnętrznych serwisów jak wandb
    )
    
    # 6. Tworzenie obiektu Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # 7. Start uczenia!
    print("🔥 Start treningu na GPU (jeśli dostępne)...")
    trainer.train()
    
    # 8. Zapis wyników
    print(f"💾 Zapisywanie wytrenowanych wag (.pth, config) do {output_model_path}")
    # To zapisze model.safetensors (nowszy format .pth) oraz pliki konfiguracyjne do wybranego folderu
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print("✅ Trening zakończony sukcesem!")

if __name__ == "__main__":
    train_classifier()
