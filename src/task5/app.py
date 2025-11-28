import streamlit as st
from transformers import pipeline

# Load model once (cached so it doesn't reload every time)
# llm-finetuned folder have the model, which is 290 MB in my case. so I didn't included it on github repo
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="./llm-finetuned",
        device_map="auto"   # GPU if available, otherwise CPU
    )

generator = load_model()

# UI
st.title("🧠 GenAI Text Generator")
st.write("Enter a prompt and generate text using your fine-tuned LLM.")

# Input
prompt = st.text_area("Prompt", "Roadmap to become genai engineer starts with...?")

max_length = st.slider(
    "Max Length",
    min_value=20,
    max_value=300,
    value=80,
    step=10
)

temperature = st.slider(
    "Temperature (creativity)",
    min_value=0.1,
    max_value=1.5,
    value=0.7,
    step=0.1
)

top_p = st.slider(
    "Top-p (nucleus sampling)",
    min_value=0.1,
    max_value=1.0,
    value=0.9,
    step=0.05
)

# Generate button
if st.button("Generate"):
    with st.spinner("Generating…"):
        output = generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        result = output[0]["generated_text"]
        st.subheader("Generated Text")
        st.write(result)
