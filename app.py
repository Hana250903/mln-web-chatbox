import requests
from flask import Flask, request, jsonify
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
from flask_cors import CORS
import google.generativeai as genai
import soundfile as sf # Thêm dòng này
import whisper
import os
import tempfile
import torch

app = Flask(__name__)
CORS(app)
app.json_encoder = LazyJSONEncoder

# Swagger cấu hình chuẩn để gửi file
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Whisper API",
        "description": "Upload a .wav audio file to get transcription using OpenAI Whisper",
        "version": "1.0"
    },
    "consumes": [
        "multipart/form-data"
    ],
    "produces": [
        "application/json"
    ]
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,  # tất cả routes
            "model_filter": lambda tag: True   # tất cả models
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)

# Load model Whisper 1 lần
# Kiểm tra xem có GPU không, nếu có thì sử dụng
# if torch.cuda.is_available():
#     device = "cuda"
#     print("Using GPU for Whisper inference.")
#     print(f"GPU Name: {torch.cuda.get_device_name(0)}")
# else:
    # device = "cpu"
    # print("Using CPU for Whisper inference. Consider GPU for faster performance.")

# Medium model 1,5gb, large model 2,88gb, turbo model 1,51gb
# model = whisper.load_model("turbo", device=device)

GEMINI_API_KEY = "AIzaSyAFCZLK0Vr0mDLPcOFyDy8H7SWAl2vSb1A"
genai.configure(api_key=GEMINI_API_KEY)
# Gemini model setup
ai_model = genai.GenerativeModel('gemini-2.0-flash')

# @app.route('/transcribe', methods=['POST'])
# @swag_from({
#     'parameters': [
#         {
#             'name': 'file',
#             'in': 'formData',
#             'type': 'file',
#             'required': True,
#             'description': 'Upload a .wav file',
#         }
#     ],
#     'responses': {
#         200: {
#             'description': 'Transcribed text',
#             'examples': {
#                 'application/json': {
#                     'text': 'Xin chào tôi là AI'
#                 }
#             }
#         },
#         400: {
#             'description': 'Missing file'
#         },
#         500: {
#             'description': 'Server error'
#         }
#     }
# })
# def transcribe():
    # if 'file' not in request.files:
    #     return jsonify({"error": "No file uploaded"}), 400

    # file = request.files['file']
    # if file.filename == '':
    #     return jsonify({"error": "No selected file"}), 400

    # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    #     file.save(tmp.name)
    #     temp_path = tmp.name

    # print(f"Received file: {temp_path}")  # Debug print
    # print(f"File size: {os.path.getsize(temp_path)} bytes")  # Debug print

    # try:
    #     # Try loading the audio with soundfile to verify integrity
    #     audio_data, sample_rate = sf.read(temp_path)
    #     print(f"Audio loaded: shape={audio_data.shape}, sample_rate={sample_rate}") # Debug print
    #     if audio_data.size == 0:
    #         print("WARNING: Audio file is empty or contains no samples.") # Debug print

    #     result = model.transcribe(temp_path, language="vi")
    #     print(f"Whisper transcription result: '{result['text']}'") # Debug print
    #     return jsonify({"text": result['text']})
    # except Exception as e:
    #     print(f"Error during transcription: {e}") # Debug print
    #     return jsonify({"error": str(e)}), 500
    # finally:
    #     if os.path.exists(temp_path):
    #         os.remove(temp_path)

# @app.route('/ai-chat', methods=['POST'])
# @swag_from({
#     'tags': ['Gemini AI Chat'],
#     'parameters': [
#         {
#             'name': 'body',
#             'in': 'body',
#             'required': True,
#             'schema': {
#                 'type': 'object',
#                 'properties': {
#                     'text': {
#                         'type': 'string',
#                         'example': 'Xin chào AI!'
#                     }
#                 },
#                 'required': ['text']
#             }
#         }
#     ],
#     'consumes': [
#         'application/json'
#     ],
#     'produces': [
#         'application/json'
#     ],
#     'responses': {
#         200: {
#             'description': 'Phản hồi từ AI',
#             'examples': {
#                 'application/json': {
#                     'response': 'Chào bạn! Tôi có thể giúp gì?'
#                 }
#             }
#         },
#         400: {
#             'description': 'Thiếu dữ liệu text'
#         },
#         500: {
#             'description': 'Lỗi nội bộ server'
#         }
#     }
# })
# def chat_with_ai():
    # try:
    #     data = request.get_json()
    #     user_text = data.get("text", "")

    #     if not user_text:
    #         return jsonify({"error": "Missing text field"}), 400

    #     # Gửi đến Gemini
    #     response = ai_model.generate_content(user_text)

    #     return jsonify({"response": response.text})

    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

# Cấu hình Supabase
# SUPABASE_URL = "https://ipakxqhbzarqwmixblyj.supabase.co"
# SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlwYWt4cWhiemFycXdtaXhibHlqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAwNzY0MzgsImV4cCI6MjA2NTY1MjQzOH0.MBDQI1uEamCOsPdrtvpPwuvIcXYuakgzMRrgLcHP7e0"  # Có thể dùng service role key
# SUPABASE_TABLE = "figures"

# def get_figures_from_supabase():
    # headers = {
    #     "apikey": SUPABASE_API_KEY,
    #     "Authorization": f"Bearer {SUPABASE_API_KEY}",
    #     "Content-Type": "application/json",
    # }
    # url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    # response = requests.get(url, headers=headers)

    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     return []

# @app.route('/ask', methods=['POST'])
# @swag_from({
#     'tags': ['Gemini AI Chat'],
#     'parameters': [
#         {
#             'name': 'body',
#             'in': 'body',
#             'required': True,
#             'schema': {
#                 'type': 'object',
#                 'properties': {
#                     'text': {
#                         'type': 'string',
#                         'example': 'Xin chào AI!'
#                     }
#                 },
#                 'required': ['text']
#             }
#         }
#     ],
#     'consumes': ['application/json'],
#     'produces': ['application/json'],
#     'responses': {
#         200: {
#             'description': 'Phản hồi từ AI',
#             'examples': {
#                 'application/json': {
#                     'answer': 'Chào bạn! Tôi có thể giúp gì?'
#                 }
#             }
#         },
#         400: {
#             'description': 'Thiếu dữ liệu text'
#         },
#         500: {
#             'description': 'Lỗi nội bộ server'
#         }
#     }
# })
# def ask():
    # data = request.get_json()
    # question = data.get("text", "")

    # if not question:
    #     return jsonify({"error": "Missing 'text' field"}), 400

    # figures = get_figures_from_supabase()

    # if not figures:
    #     return jsonify({"error": "Not fould"}), 404

    # prompt = f"""
    #     Dưới đây là danh sách các figure đang có:

    #     {figures}

    #     Câu hỏi: {question}

    #     Hãy trả lời dựa trên dữ liệu figure ở trên. Nếu không tìm thấy thông tin phù hợp, hãy nói rõ.
    #     """

    # try:
    #     response = ai_model.generate_content(prompt)
    #     return jsonify({
    #         "answer": response.text.strip()
    #     })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

# Lấy nội dung từ file văn bản để làm prompt cơ sở
def get_base_prompt_text():
    try:
        with open('tu_lieu.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "Không tìm thấy tài liệu gốc. Vui lòng cung cấp nội dung."

base_prompt = get_base_prompt_text()

# --- Gemini Chat Endpoint ---
@app.route('/ask_gemini', methods=['POST'])
@swag_from({
    'tags': ['Gemini AI Chat'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'question': {
                        'type': 'string',
                        'example': 'Sứ mệnh của giai cấp công nhân thể hiện ở những khía cạnh nào?'
                    }
                },
                'required': ['question']
            }
        }
    ],
    'consumes': ['application/json'],
    'produces': ['application/json'],
    'responses': {
        200: {
            'description': 'Phản hồi từ AI',
            'examples': {
                'application/json': {
                    'answer': 'Chào bạn! Tôi có thể giúp gì?'
                }
            }
        },
        400: {
            'description': 'Thiếu dữ liệu question'
        },
        500: {
            'description': 'Lỗi nội bộ server'
        }
    }
})
def ask_gemini():
    data = request.get_json(silent=True)
    if not data or 'question' not in data:
        return jsonify({"error": "Vui lòng cung cấp 'question' trong request body."}), 400

    user_question = data.get('question')
    full_prompt = f"Dựa trên các tài liệu sau về sứ mệnh lịch sử của giai cấp công nhân Việt Nam:\n\n---\n{base_prompt}\n---\n\n{user_question}"

    try:
        response = ai_model.generate_content(full_prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)