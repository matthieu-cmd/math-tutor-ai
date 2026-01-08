import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ⚠️ REPLACE THIS WITH YOUR ACTUAL USERNAME ⚠️
YOUR_HF_USERNAME = "matt0293"  # CHANGE THIS!
MODEL_NAME = f"{YOUR_HF_USERNAME}/math-tutor-llama"

st.set_page_config(page_title="AI Algebra Tutor", page_icon="🧮")

st.title("🧮 AI Algebra Tutor")
st.markdown(f"Using model: `{MODEL_NAME}`")

# Load from Hugging Face
@st.cache_resource
def load_tutor():
    try:
        with st.spinner("Loading AI model from Hugging Face..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Check if it's a PEFT model
            try:
                model = PeftModel.from_pretrained(model, MODEL_NAME)
            except:
                pass  # Not a PEFT model
            
            st.success("✅ Model loaded!")
            return model, tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load: {e}")
        return None, None

model, tokenizer = load_tutor()

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask algebra question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Solving..."):
            try:
                if model and tokenizer:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    outputs = model.generate(**inputs, max_new_tokens=300)
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    response = f"Teaching how to solve: {prompt}\n\n1. Identify variables\n2. Isolate x\n3. Solve step-by-step"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("📚 Examples")
    examples = ["2x + 5 = 13", "3x - 7 = 14", "2(x + 3) = 10"]
    for ex in examples:
        if st.button(f"Solve: {ex}"):
            st.session_state.messages.append({"role": "user", "content": f"Solve {ex}"})
            st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
