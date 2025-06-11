# ai_summary.py
from transformers import pipeline

# Load model summarization FLAN-T5
pipe = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_ai_summary(input_data, prediction, prob_success):
    label = "BERHASIL" if prediction == 1 else "GAGAL"

    prompt = f"""
Data startup:
- Total Pendanaan: {input_data['funding_total_usd_cleaned'][0]:,.0f} USD
- Putaran Pendanaan: {input_data['funding_rounds'][0]}
- Negara: {input_data['country_code'][0]}
- Kategori Utama: {input_data['primary_category'][0]}
- Usia Startup: {input_data['startup_age_years'][0]} tahun

Prediksi model: {label} dengan tingkat keyakinan {prob_success:.2f}%

Buat ringkasan analisis mengapa model memprediksi seperti itu, berdasarkan data yang tersedia.
"""

    try:
        result = pipe(prompt, max_new_tokens=150)[0]['generated_text']
        return result.strip()
    except Exception as e:
        return f"❗ Gagal membuat ringkasan AI: {e}"
