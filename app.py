import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Your model
MODEL_NAME = "matt0293/math-tutor-llama"

st.set_page_config(page_title="AI Algebra Tutor", page_icon="🧮")

st.title("🧮 AI Algebra Tutor")
st.caption(f"Model: {MODEL_NAME}")

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I help solve algebra problems. Try: 'Solve 2x + 5 = 13'"}
    ]

# SIMPLE model loader - NO BITSANDBYTES
@st.cache_resource
def load_simple_model():
    try:
        with st.spinner("Setting up tutor..."):
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Load model WITHOUT any quantization (no bitsandbytes)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,  # Use float32 (CPU safe)
                low_cpu_mem_usage=True,     # Help with memory
                device_map="cpu"            # Force CPU
            )
            
            st.success("✅ Tutor ready!")
            return model, tokenizer
            
    except Exception as e:
        st.warning(f"⚠️ Using demo mode: {str(e)[:100]}...")
        return None, None

model, tokenizer = load_simple_model()

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Response function
def get_ai_response(prompt):
    if model is None or tokenizer is None:
        return f"""**Demo solution for:** {prompt}

**Steps to solve:**
1. Identify the variable (x)
2. Move numbers to the other side
3. Isolate x
4. Solve

*Example for "2x + 5 = 13":*
- Subtract 5: 2x = 8
- Divide by 2: x = 4
✅ **Answer:** x = 4"""
    
    try:
        # Simple prompt
        input_text = f"Algebra problem: {prompt}\n\nStep-by-step solution:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=200)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up
        if "Step-by-step solution:" in response:
            response = response.split("Step-by-step solution:")[-1].strip()
        
        return response
        
    except Exception as e:
        return f"AI error: {str(e)}"

# Chat input
if prompt := st.chat_input("Type algebra problem..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Calculating..."):
            response = get_ai_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.header("💡 Examples")
    
    examples = [
        "2x + 5 = 13",
        "3x - 7 = 14", 
        "2(x + 3) = 10",
        "5x + 2 = 3x + 12"
    ]
    
    for ex in examples:
        if st.button(f"Solve: {ex}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": f"Solve {ex}"})
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! Ask me anything."}
        ]
        st.rerun()
    
    st.divider()
    st.markdown("**Note:** First response may take 20-30 seconds to load model.")

# Footer
st.markdown("---")
st.caption("Fine-tuned LLaMA 3.1 for algebra | Specialized in linear equations")
