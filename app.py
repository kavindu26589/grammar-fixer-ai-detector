import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
import os
from huggingface_hub import login
import torch

# Authenticate with Hugging Face token
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
login(token=HUGGINGFACE_TOKEN)

# Load Phi-4 Mini
phi_id = "microsoft/phi-4-mini-instruct"
phi_tokenizer = AutoTokenizer.from_pretrained(phi_id, token=HUGGINGFACE_TOKEN)
phi_model = AutoModelForCausalLM.from_pretrained(phi_id, torch_dtype="auto", device_map="auto", token=HUGGINGFACE_TOKEN)
phi_pipe = pipeline("text-generation", model=phi_model, tokenizer=phi_tokenizer)

# Load T5 for paraphrasing
t5_pipe = pipeline("text2text-generation", model="google-t5/t5-base")

# Load AI Detector
detector_id = "openai-community/roberta-base-openai-detector"
detector_tokenizer = AutoTokenizer.from_pretrained(detector_id)
detector_model = AutoModelForSequenceClassification.from_pretrained(detector_id)

# ===== Helper: Circular HTML Visualization =====
def circular_html(ai_percent):
    color = (
        "#4caf50" if ai_percent < 30 else
        "#2196f3" if ai_percent < 60 else
        "#f44336" if ai_percent < 90 else
        "#e91e63"
    )
    text_color = "#fff" if ai_percent > 50 else "#222"

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; margin-top: 15px;">
        <div style="font-size: 20px; font-weight: 500; margin-bottom: 12px; color: {text_color};">
            AI Likelihood
        </div>
        <div style="
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: conic-gradient({color} {ai_percent}%, #33333333 {ai_percent}%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            font-weight: bold;
            color: {text_color};
            border: 6px solid rgba(255,255,255,0.1);
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
        ">
            {ai_percent}%
        </div>
    </div>
    """

# ===== Chunking for Large Input Support =====
def chunk_text(text, max_tokens=300):
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current.split()) + len(para.split()) < max_tokens:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            current = para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks

# ===== Phi Prompt Wrapper =====
def generate_phi_prompt(text, instruction):
    chunks = chunk_text(text)
    outputs = []
    for chunk in chunks:
        prompt = f"{instruction}\n{chunk}\nResponse:"
        result = phi_pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.3)[0]["generated_text"]
        outputs.append(result.split("Response:")[1].strip() if "Response:" in result else result.strip())
    return "\n\n".join(outputs)

# ===== Writing Tools =====
def fix_grammar(text):
    return generate_phi_prompt(text, "Correct all grammar and punctuation errors in the following text. Provide only the corrected version:")

def improve_tone(text):
    return generate_phi_prompt(text, "Rewrite the following text to sound more formal, polite, and professional:")

def improve_fluency(text):
    return generate_phi_prompt(text, "Rewrite the following to improve its clarity, sentence flow, and natural fluency:")

def paraphrase(text):
    chunks = chunk_text(text, max_tokens=60)
    return "\n\n".join(
        t5_pipe("paraphrase this sentence: " + chunk, max_length=128, num_beams=5, do_sample=False)[0]["generated_text"]
        for chunk in chunks
    )

# ===== Apply Enhancements Based on Checkboxes =====
def apply_selected_enhancements(text, fix, tone, fluency, para):
    result = text
    if fix:
        result = fix_grammar(result)
    if tone:
        result = improve_tone(result)
    if fluency:
        result = improve_fluency(result)
    if para:
        result = paraphrase(result)
    return result

# ===== AI Detection and Visualization =====
def detect_ai_percent(text):
    inputs = detector_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = detector_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()
        ai_score = round(probs[1].item() * 100, 2)
        label = "Likely AI-Generated" if ai_score > 50 else "Likely Human"
        return label, circular_html(ai_score)

# ===== Rewrite for Human-Like Text =====
def rewrite_to_human(text):
    return generate_phi_prompt(text, "Rewrite the following text so that it is indistinguishable from human writing and avoids AI detection. Be natural and fluent:")

# ===== File Handling =====
def load_file(file_obj):
    if file_obj is None:
        return ""
    return file_obj.read().decode("utf-8")

def save_file(text):
    path = "/tmp/output.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

# ===== Gradio Interface =====
with gr.Blocks() as demo:
    gr.Markdown("# ✍️ AI Writing Assistant + Circular AI Detector")
    gr.Markdown("Fix grammar, tone, fluency, paraphrase, and detect AI content with a modern circular progress view.")

    with gr.Row():
        file_input = gr.File(label="📂 Upload .txt File", file_types=[".txt"])
        load_btn = gr.Button("📥 Load Text")
        input_text = gr.Textbox(lines=12, label="Input Text")
    load_btn.click(fn=load_file, inputs=file_input, outputs=input_text)

    gr.Markdown("## ✅ Select Enhancements to Apply")
    with gr.Row():
        check_fix = gr.Checkbox(label="✔️ Fix Grammar")
        check_tone = gr.Checkbox(label="🎯 Improve Tone")
        check_fluency = gr.Checkbox(label="🔄 Improve Fluency")
        check_paraphrase = gr.Checkbox(label="🌀 Paraphrase")

    run_selected = gr.Button("🚀 Run Selected Enhancements")
    output_text = gr.Textbox(lines=12, label="Output")
    run_selected.click(
        fn=apply_selected_enhancements,
        inputs=[input_text, check_fix, check_tone, check_fluency, check_paraphrase],
        outputs=output_text
    )

    gr.Markdown("## 🕵️ AI Detection")
    detect_btn = gr.Button("Detect AI Probability")
    ai_summary = gr.Textbox(label="AI Summary", interactive=False)
    ai_circle = gr.HTML()
    detect_btn.click(fn=detect_ai_percent, inputs=input_text, outputs=[ai_summary, ai_circle])

    gr.Markdown("## 🔁 Rewrite to Reduce AI Probability")
    rewrite_btn = gr.Button("Rewrite as Human")
    rewritten_text = gr.Textbox(lines=12, label="Rewritten Text")
    rewrite_btn.click(fn=rewrite_to_human, inputs=input_text, outputs=rewritten_text)

    gr.Markdown("## 📤 Download Output")
    download_btn = gr.Button("💾 Download Output")
    download_file = gr.File(label="Click to download", interactive=True)
    download_btn.click(fn=save_file, inputs=output_text, outputs=download_file)

demo.launch()