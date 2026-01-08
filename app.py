import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ⚠️ REPLACE WITH YOUR USERNAME ⚠️
YOUR_HF_USERNAME = "matthieu2312"  # CHANGE THIS!
MODEL_NAME = f"{YOUR_HF_USERNAME}/math-tutor-llama"

st.set_page_config(page_title="AI Algebra Tutor", page_icon="🧮")

st.title("🧮 AI Algebra Tutor")
st.markdown(f"Model: `{MODEL_NAME}`")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I help with algebra problems. Try: 'Solve 2x + 5 = 13'"}
    ]

# Simple model loader (no PEFT for now)
@st.cache_resource
def load_simple_model():
    try:
        with st.spinner("Loading tutor..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map="cpu",  # Force CPU
                low_cpu_mem_usage=True
            )
            st.success("✅ Ready!")
            return model, tokenizer
    except Exception as e:
        st.error(f"Load error: {e}")
        # Fallback to mock responses
        return None, None

model, tokenizer = load_simple_model()

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask algebra question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if model and tokenizer:
                    # Simple generation
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=150)
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # Fallback response
                    response = f"""**Solution for:** {prompt}

1. Identify the variable (usually x)
2. Isolate it on one side
3. Perform inverse operations
4. Check your solution

Example: For "2x + 5 = 13":
- Subtract 5: 2x = 8
- Divide by 2: x = 4
✅ Solution: x = 4"""
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("💡 Examples")
    for problem in ["2x + 5 = 13", "3x - 7 = 14", "2(x + 3) = 10"]:
        if st.button(problem):
            st.session_state.messages.append({"role": "user", "content": f"Solve {problem}"})
            st.rerun()
    
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! Ask me an algebra problem."}
        ]
        st.rerun()
