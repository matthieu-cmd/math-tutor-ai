import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# YOUR MODEL - CORRECT USERNAME
MODEL_NAME = "matt0293/math-tutor-llama"

st.set_page_config(page_title="AI Algebra Tutor", page_icon="🧮")

st.title("🧮 AI Algebra Tutor")
st.caption(f"Model: {MODEL_NAME}")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your algebra tutor. Ask me to solve equations like '2x + 5 = 13'"}
    ]

# Load model
@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading AI tutor..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,  # CPU compatible
                device_map="auto"
            )
            return model, tokenizer
    except Exception as e:
        st.error(f"⚠️ Couldn't load model: {e}")
        st.info("Using demo mode for now. Make sure your Hugging Face model is public.")
        return None, None

model, tokenizer = load_model()

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type algebra problem..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Solving..."):
            try:
                if model and tokenizer:
                    # Prepare input
                    full_prompt = f"Solve this algebra problem: {prompt}\n\nStep-by-step solution:"
                    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=256)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=250,
                            temperature=0.7,
                            do_sample=True
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean response
                    if "Step-by-step solution:" in response:
                        response = response.split("Step-by-step solution:")[-1].strip()
                    
                else:
                    # Demo response
                    response = f"""**Problem:** {prompt}

**Solution Steps:**
1. Identify the variable (x)
2. Isolate x on one side
3. Perform inverse operations
4. Simplify
5. Check solution

*Example:* For "2x + 5 = 13":
- Step 1: Subtract 5 from both sides: 2x = 8
- Step 2: Divide by 2: x = 4
✅ **Answer:** x = 4"""
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar
with st.sidebar:
    st.header("📚 Try These:")
    
    examples = [
        "Solve 2x + 5 = 13",
        "What is 3x - 7 = 14?",
        "Help: 2(x + 3) = 10",
        "Find x: 5x + 2 = 3x + 12"
    ]
    
    for ex in examples:
        if st.button(ex, use_container_width=True, type="secondary"):
            st.session_state.messages.append({"role": "user", "content": ex})
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! Ask me an algebra problem."}
        ]
        st.rerun()
    
    st.divider()
    st.markdown("**Need help?**")
    st.markdown("1. Ensure model is public on HF")
    st.markdown("2. Check [model page](https://huggingface.co/matt0293/math-tutor-llama)")
