import pandas as pd
import os
from datasets import load_dataset

def build_dataset():
    """
    Tworzy zbiór danych treningowych pobierając profesjonalną bazę ataków JailbreakV-28K z HuggingFace.
    Zbiór ten (zawierający m.in. jailbreaki z LLaMA, ataki na system prompt) zostanie przekonwertowany 
    do struktury akceptowalnej przez nasz skrypt trenujący (0 = Benign, 1 = Prompt Injection).
    """
    print("⏳ Pobieranie potężnego datasetu JailbreakV-28K z chmury (HuggingFace)...")
    print("To może chwilę potrwać za pierwszym razem (pobieranie ~20 tysięcy rekordów).")
    
    # Ładujemy profesjonalny dataset z HuggingFace
    hf_data = load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K", split="JailBreakV_28K")
    print(f"✅ Pobrano pomyślnie {len(hf_data)} rekordów.")
    
    # Konwersja do naszego formatu dla DistilBERTa
    # JailbreakV-28K to głównie ataki (szkodliwe)
    custom_data = []
    
    for item in hf_data:
        # Ten dataset w kolumnie "jailbreak_query" posiada wstrzyknięte złośliwe zapytanie
        query = item.get("jailbreak_query")
        if query and isinstance(query, str) and query.strip():
            custom_data.append({"text": query.strip(), "label": 1}) # 1 = Prompt Injection / Jailbreak
            
    # UWAGA: Ponieważ z pobranego datasetu mamy tylko ataki, musimy dodać bezpieczne (benign) przykłady
    # W produkcyjnym systemie należy pobrać osobny dataset bezpiecznych rozmów (np. lmsys/toxic-chat (split benign))
    # Na ten moment dodaję sztuczne 'benign' żeby model nie klasyfikował wszystkiego jako atak 
    # ze względu na brak zbilansowania.
    print("Dodawanie syntetycznych przykładów bezpiecznych (Benign) dla zbilansowania...")
    benign_examples = [
        "What is the weather today?",
        "Please summarize the history of Rome.",
        "How do I cook pasta?",
        "Write a python script to reverse a string.",
        "What are the benefits of eating healthy?",
        "Explain the theory of relativity simply.",
        "Can you translate 'hello' to Spanish?",
        "I need a recipe for chocolate cake.",
        "How do I change a tire?",
        "Tell me a fun fact about space."
    ] * 500 # Powielamy by zbalansować wagi
    
    for b in benign_examples:
        custom_data.append({"text": b, "label": 0})
        
    df = pd.DataFrame(custom_data)
    
    # Tasujemy (shuffle)
    df = df.sample(frac=1).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guardrail_dataset.csv")
    
    df.to_csv(csv_path, index=False)
    print(f"✅ Zbudowano i zapisano dataset {csv_path} ({len(df)} rekordów). Gotowe do treningu!")

if __name__ == "__main__":
    build_dataset()
