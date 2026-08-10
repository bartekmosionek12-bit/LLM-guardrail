import os
import torch
import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ścieżka do wytrenowanego modelu i datasetu
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model_weights")
dataset_path = os.path.join(current_dir, "guardrail_dataset.csv")

# Mapowanie wyników (etykiet numerycznych na ludzki tekst)
LABELS = {
    0: "✅ BENIGN (Bezpieczny prompt)",
    1: "⚠️ PROMPT INJECTION (Wykryto atak / nadpisywanie instrukcji!)",
    2: "⛔ TOXIC / HARMFUL (Szkodliwe / niedozwolone treści!)"
}

# 1. Sprawdzamy czy model istnieje
if not os.path.exists(model_path):
    raise FileNotFoundError("Nie znaleziono zapisanych wag modelu! Uruchom wpierw 'train.py'.")

print("Ładowanie modelu Guardrail do pamięci...")

# 2. Ładowanie modelu i tokenizera w trybie ewaluacji (inference)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def evaluate_guardrail(text: str):
    """Przepuszcza tekst przez model i zwraca wynik."""
    if not text.strip():
        return "Wpisz jakiś tekst."
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class_id].item()
    
    result_text = LABELS.get(predicted_class_id, "Nieznany")
    return f"{result_text}\n\nPewność modelu: {confidence*100:.2f}%"

def correct_and_save(text: str, correct_label: int):
    """Zapisuje poprawiony tekst i jego etykietę z powrotem do pliku CSV."""
    if not text.strip():
        return "Brak tekstu do zapisania!"
    
    new_row = {"text": text.strip(), "label": correct_label}
    
    # Otwieramy bazę, dodajemy poprawkę na sam koniec i zapisujemy
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
        
    df.to_csv(dataset_path, index=False)
    return f"✅ Zapisano pomyślnie w bazie! Wpis zakodowano jako: {LABELS[correct_label]}"

# 3. Interfejs Gradio (Z użyciem gr.Blocks dla własnych przycisków)
with gr.Blocks(theme="default") as demo:
    gr.Markdown("# 🛡️ Intelligent AI Guardrail Classifier")
    gr.Markdown("Lokalny model AI wytrenowany na karcie RTX 3070. Ocenia bezpieczeństwo promptów.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=5, label="Wklej prompt użytkownika lub odpowiedź LLM...")
            submit_btn = gr.Button("Sprawdź tekst", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Werdykt Guardraila")
            
    gr.Markdown("---")
    gr.Markdown("### 👨‍🏫 Czy model się pomylił? Doucz go!")
    gr.Markdown("Użyj przycisków poniżej, aby wymusić poprawną etykietę dla tekstu, który aktualnie znajduje się w lewym okienku. Zostanie to dopisane na sam dół pliku `guardrail_dataset.csv`. Przy następnym uruchomieniu `train.py` model będzie już to wiedział!")
    
    with gr.Row():
        btn_benign = gr.Button("To jest ✅ BENIGN")
        btn_injection = gr.Button("To jest ⚠️ PROMPT INJECTION")
        btn_toxic = gr.Button("To jest ⛔ TOXIC")
        
    correction_status = gr.Textbox(label="Status operacji", interactive=False)
    
    # Reakcje na przyciski
    submit_btn.click(fn=evaluate_guardrail, inputs=[input_text], outputs=[output_text])
    
    # Lambda pozwala nam przekazać statyczną liczbę (ID klasy) z poziomu przycisku
    btn_benign.click(fn=lambda t: correct_and_save(t, 0), inputs=[input_text], outputs=[correction_status])
    btn_injection.click(fn=lambda t: correct_and_save(t, 1), inputs=[input_text], outputs=[correction_status])
    btn_toxic.click(fn=lambda t: correct_and_save(t, 2), inputs=[input_text], outputs=[correction_status])

if __name__ == "__main__":
    print("🚀 Uruchamianie interfejsu graficznego! Zobacz w przeglądarce...")
    demo.launch(server_name="127.0.0.1", server_port=7860)
