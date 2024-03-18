from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, request, jsonify , make_response

def load_model():
    peft_model_id = "ANWAR101/lora-bart-base-youtube-cnn"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id)
    return model , tokenizer

app = Flask(__name__)

@app.before_request  # Use @app.before_request for middleware-like functionality
def check_cors():
    response = make_response(jsonify({'message': 'CORS is allowed for all origins'}))
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins
    response.headers.add('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, UPDATE')
    return response

@app.route("/summarize", methods=["POST"])
def summarize():
  model , tokenizer = load_model()
  # Get the text from the request body
  data = request.get_json()
  text = data.get("text")

  # Check for missing text
  if not text:
    return jsonify({"error": "Missing text in request body"}), 400

  # Preprocess the text
  inputs = tokenizer(text , truncation = True, return_tensors="pt")

  # Generate summary using the model
  outputs = model.generate(**inputs, max_length=300, min_length=50 , do_sample = True , num_beams = 3 , no_repeat_ngram_size=2 , temperature=0.6 , length_penalty=1.0)
  
  summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

  # Return the summary as JSON
  return jsonify({"summary": summary})

if __name__ == "__main__":
  app.run(debug=False)  # Set debug=False for production