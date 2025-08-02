import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify
import PyPDF2

# Initialize Flask app
app = Flask(__name__)

# Load IBM Granite model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ibm-granite/granite-speech-3.3-2b"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name, device_map=device, torch_dtype=torch.bfloat16
)

# Initialize PDF text storage
pdf_text = ""

# Route for home
@app.route('/')
def home():
    return "Voice-Based StudyMate Backend Running"

# Route to upload PDF and extract text
@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    global pdf_text
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    reader = PyPDF2.PdfReader(file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    return jsonify({"message": "PDF uploaded and text extracted successfully."})

# Route to handle speech input and question answering
@app.route('/ask', methods=['POST'])
def ask_question():
    global pdf_text
    if not pdf_text:
        return jsonify({"error": "No PDF uploaded yet!"}), 400

    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided!"}), 400

    # Process speech input
    audio_path = hf_hub_download(repo_id=model_name, filename="10226_10111_000000.wav")
    wav, sr = torchaudio.load(audio_path, normalize=True)
    assert wav.shape[0] == 1 and sr == 16000  # mono, 16khz

    system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    user_prompt = "<|audio|>can you transcribe the speech into a written format?"
    chat = [
        dict(role="system", content=system_prompt),
        dict(role="user", content=user_prompt),
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
    model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)

    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
    output_text = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
    stt_output = output_text[0].upper()

    # Answer the question based on PDF content
    context = pdf_text[:2000]  # limit context for performance
    answer = qa_pipeline(question=question, context=context)

    return jsonify({
        "stt_output": stt_output,
        "answer": answer["answer"],
        "score": answer["score"]
    })

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
