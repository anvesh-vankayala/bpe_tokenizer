import streamlit as st
import encoder_parallel_telugu as encode_parallel
from consecutive_tokens import get_consecutive_tokens, search_consecutive_tokens
import tokenizer

def encode(text):
    if text == "":
        return "Enter text to encode..."
    encoded_tokens = [token.encode('utf-8') for token in text]
    consective_tokens = get_consecutive_tokens(encoded_tokens,window_size=4)
    # Reading vocabulary from file
    formatted_vocab = tokenizer.read_vocab_from_file()
    # Invert vocabulary
    inverted_vocab = {v: k for k, v in formatted_vocab.items()}
    # Expand vocabulary
    decoder_map = tokenizer.expand_vocab(inverted_vocab)
    # Invert back again after expansion
    re_inverted_vocab = {k: v for v, k in decoder_map.items()}
    
    # encoded_tokens = [re_inverted_vocab.get(token) for token in consective_tokens]
    encoded_tokens = search_consecutive_tokens(consective_tokens, re_inverted_vocab)
    print(encoded_tokens)    
    return f"Encoded: {encoded_tokens}"

def decode(text):
    # Placeholder for decoding logic
    toks_li = [token for token in text.split(',')]
    # Reading vocabulary from file
    formatted_vocab = tokenizer.read_vocab_from_file()
    # Invert vocabulary
    inverted_vocab = {v: k for k, v in formatted_vocab.items()}
    # Expand vocabulary
    decoder_map = tokenizer.expand_vocab(inverted_vocab)
    decoded_tokens = [decoder_map.get(int(token)) for token in toks_li]
    decoded_tokens = [token[0] for token in decoded_tokens]
    tokens = [token.decode('utf-8') for token in decoded_tokens]
    decoded_tokens = b''.join(decoded_tokens)
    decoded_tokens = decoded_tokens.decode('utf-8')
    return f"->Decoded: {decoded_tokens} : ->Tokens: {tokens}"

st.set_page_config(page_title="Telugu BPE Tokenizer", layout="centered", initial_sidebar_state="expanded")
st.markdown("<h1 style='color: #2ECC40; text-align: center;'>Telugu BPE Tokenizer</h1>", unsafe_allow_html=True)

# Add custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        color: #FFFFFF;
        background-color: #2C3E50;
        font-family: "Arial", sans-serif;
        font-size: 2.5em;
        padding: 20px;
        text-align: center;
    }
    .subheader {
        color: #2980B9;
        font-size: 1.5em;
    }
    .text-area {
        background-color: #ECF0F1;
        border: 1px solid #BDC3C7;
        border-radius: 5px;
    }
    .orange-button {
        background-color: #FFA500; /* Bright orange color */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True
)

# Create two columns for encoder and decoder
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='subheader' style='color: #FFA500;'>Encoder</div>", unsafe_allow_html=True)
    encoder_input = st.text_area("Input Text for Encoding", placeholder="Enter text to encode...", key="encoder_input", height=100)
    if st.button("Encode", key="encode_button"):
        encoder_output = encode(encoder_input)
        st.text_area("Encoded Output", value=encoder_output, height=100, disabled=True, key="encoder_output")

with col2:
    st.markdown("<div class='subheader' style='color: #FFA500;'>Decoder</div>", unsafe_allow_html=True)
    decoder_input = st.text_area("Input Text for Decoding", placeholder="51,32,63,94,15", key="decoder_input", height=100)
    if st.button("Decode", key="decode_button"):
        decoder_output = decode(decoder_input)
        st.text_area("Decoded Output", value=decoder_output, height=100, disabled=True, key="decoder_output")
        
st.markdown("<hr style='border: 1px solid #BDC3C7;'>", unsafe_allow_html=True)  # Add a horizontal line above the section in grey
# Add sample texts at the end of the page
st.markdown("<div class='subheader'>Sample Texts</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 10px;'> <span style='font-weight: bold;'>రెండు&nbsp;&nbsp;విధాలా&nbsp;&nbsp;ఆలోచిస్తా.</span></div>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 10px;'> <span style='font-weight: bold;'>మోదీ&nbsp;&nbsp;మార్కు&nbsp;&nbsp;రాజకీయం.</span></div>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 10px;'> <span style='font-weight: bold;'>తెలుగు&nbsp;&nbsp;భాష&nbsp;&nbsp;ఒక&nbsp;&nbsp;ద్రావిడ&nbsp;&nbsp;భాష.</span></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.write("Streamlit app is running...")
    st.write("To view this page in your browser, run the command: `streamlit run app.py` and open the provided local URL.")
