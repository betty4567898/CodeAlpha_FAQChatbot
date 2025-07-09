
import spacy
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import random

# FAQ data
faq_data = [
    # Greetings
    {"question": "Hi", "answer": "Hello! How can I help you today?", "category": "greetings"},
    {"question": "Hello", "answer": "Hi there! Ask me anything.", "category": "greetings"},
    {"question": "Hey", "answer": "Hey! What can I do for you?", "category": "greetings"},
    {"question": "How are you?", "answer": "I'm doing great, thank you! How can I assist you?", "category": "greetings"},
    {"question": "Good morning", "answer": "Good morning! How can I help you today?", "category": "greetings"},
    {"question": "Good night", "answer": "Good night! Feel free to ask anything before you go.", "category": "greetings"},

    # Account
    {"question": "How do I create an account?", "answer": "Click on 'Sign Up' and fill out the required details.", "category": "account"},
    {"question": "How do I log in?", "answer": "Click 'Login' and enter your email and password.", "category": "account"},
    {"question": "I forgot my password", "answer": "Click 'Forgot Password' and follow the reset instructions.", "category": "account"},
    {"question": "Can I change my email address?", "answer": "Yes, go to Account Settings to update your email.", "category": "account"},
    {"question": "How do I delete my account?", "answer": "Please contact support to delete your account permanently.", "category": "account"},

    # Orders & Shipping
    {"question": "Where is my order?", "answer": "Track your order via the tracking link sent to your email.", "category": "orders"},
    {"question": "How do I track my order?", "answer": "Use the tracking number sent to you after shipping.", "category": "orders"},
    {"question": "Can I cancel my order?", "answer": "You can cancel orders within 1 hour of purchase.", "category": "orders"},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship to over 50 countries.", "category": "shipping"},
    {"question": "How long does delivery take?", "answer": "Standard shipping takes 3–5 business days.", "category": "shipping"},
    {"question": "Do you offer express delivery?", "answer": "Yes, we offer same-day and next-day delivery.", "category": "shipping"},
    {"question": "What happens if I'm not home?", "answer": "The courier will try again or leave a pickup note.", "category": "shipping"},

    # Returns & Refunds
    {"question": "What is your return policy?", "answer": "You can return any item within 30 days.", "category": "returns"},
    {"question": "How do I return an item?", "answer": "Login, go to 'Orders', and click 'Return Item'.", "category": "returns"},
    {"question": "How long does a refund take?", "answer": "Refunds are processed within 5–7 business days.", "category": "refunds"},
    {"question": "Can I exchange a product?", "answer": "Yes, exchanges are allowed within 14 days.", "category": "returns"},
    {"question": "What if my item is damaged?", "answer": "Contact support with a photo of the item.", "category": "support"},

    # Payments
    {"question": "What payment methods do you accept?", "answer": "We accept Visa, MasterCard, PayPal, and more.", "category": "payments"},
    {"question": "Is my payment secure?", "answer": "Yes, all payments are encrypted and secure.", "category": "payments"},
    {"question": "Can I pay on delivery?", "answer": "Cash on Delivery is available in selected regions.", "category": "payments"},
    {"question": "Why was my card declined?", "answer": "Make sure your card has sufficient balance and is valid.", "category": "payments"},
    {"question": "Can I use a promo code?", "answer": "Yes, apply it during checkout.", "category": "payments"},

    # Products
    {"question": "Are your products eco-friendly?", "answer": "Yes, many of our items are sustainably made.", "category": "products"},
    {"question": "Do you have size guides?", "answer": "Yes, each product page includes a size chart.", "category": "products"},
    {"question": "Is this item in stock?", "answer": "Check the product page for current availability.", "category": "products"},
    {"question": "Can I pre-order a product?", "answer": "Yes, if it's listed as pre-order.", "category": "products"},
    {"question": "Do you have warranties?", "answer": "Most products come with a 1-year warranty.", "category": "products"},

    # Support
    {"question": "How do I contact customer service?", "answer": "Email us at support@example.com or use live chat.", "category": "support"},
    {"question": "What are your support hours?", "answer": "We're available from 9 AM to 6 PM, Mon to Sat.", "category": "support"},
    {"question": "Do you have live chat?", "answer": "Yes, use the chat icon on the bottom right corner.", "category": "support"},
    {"question": "Can I speak to a human?", "answer": "Sure! Our agents are happy to help during working hours.", "category": "support"},

    # Random / Fun
    {"question": "Who built you?", "answer": "I was created by Ibtissam during her CodeAlpha internship!", "category": "about"},
    {"question": "What can you do?", "answer": "I can help you with FAQs, orders, returns, and more.", "category": "about"},
    {"question": "Do you like me?", "answer": "Of course! I'm here just for you 😄", "category": "about"},
    {"question": "Are you a human?", "answer": "Nope, I'm a friendly chatbot!", "category": "about"}
]

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Preprocess
def preprocess(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Prepare vectors
questions = [faq["question"] for faq in faq_data]
processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

# Chat history storage
chat_history = []

# Enhanced matching with confidence score
def get_best_answer(user_input, history):
    global chat_history
    
    if not user_input.strip():
        return history, ""
    
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    scores = cosine_similarity(input_vector, question_vectors)
    max_score = scores.max()
    index = scores.argmax()
    
    # Get the best answer
    best_faq = faq_data[index]
    answer = best_faq["answer"]
    category = best_faq.get("category", "general")
    
    # Add confidence and category info
    confidence = max_score * 100
    
    if confidence < 30:
        answer = "I'm not sure about that. Could you please rephrase your question or contact our support team for assistance?"
        category = "unknown"
    
    # Format response with metadata
    timestamp = datetime.datetime.now().strftime("%H:%M")
    
    # Add to chat history
    chat_history.append({
        "user": user_input,
        "bot": answer,
        "category": category,
        "confidence": confidence,
        "timestamp": timestamp
    })
    
    # Update history display
    history.append([user_input, f"**[{category.upper()}]** {answer}\n\n*Confidence: {confidence:.1f}% | Time: {timestamp}*"])
    
    return history, ""

# Clear chat function
def clear_chat():
    global chat_history
    chat_history = []
    return [], ""

# Get random suggestion
def get_suggestion():
    suggestions = [
        "How do I create an account?",
        "What is your return policy?",
        "How long does delivery take?",
        "What payment methods do you accept?",
        "How do I track my order?",
        "Do you offer international shipping?",
        "How do I contact customer service?"
    ]
    return random.choice(suggestions)

# Advanced CSS with animations and modern design
advanced_css = """
/* Global Styles */
* {
    box-sizing: border-box;
}

body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;
}

/* Main Container */
.gradio-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    margin: 20px;
    padding: 30px;
}

/* Header Styles */
h1 {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 3s ease infinite;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.description {
    color: rgba(255, 255, 255, 0.9);
    text-align: center;
    font-size: 1.2rem;
    margin-bottom: 30px;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

/* Chat Interface */
.chatbot {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    border: none;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    overflow: hidden;
}

.chatbot .message {
    padding: 15px;
    margin: 10px;
    border-radius: 12px;
    animation: messageSlide 0.3s ease-out;
}

@keyframes messageSlide {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.chatbot .message.user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    margin-left: 20%;
}

.chatbot .message.bot {
    background: linear-gradient(135deg, #f093fb, #f5576c);
    color: white;
    margin-right: 20%;
}

/* Input Styles */
.textbox textarea {
    background: rgba(255, 255, 255, 0.9);
    border: 2px solid transparent;
    border-radius: 15px;
    padding: 15px 20px;
    font-size: 16px;
    color: #333;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.textbox textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    transform: translateY(-2px);
}

/* Button Styles */
.btn {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none;
    border-radius: 12px;
    color: white;
    font-size: 16px;
    font-weight: 600;
    padding: 12px 24px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    background: linear-gradient(135deg, #764ba2, #667eea);
}

.btn:active {
    transform: translateY(-1px);
}

/* Secondary Button */
.btn-secondary {
    background: linear-gradient(135deg, #f093fb, #f5576c);
}

.btn-secondary:hover {
    background: linear-gradient(135deg, #f5576c, #f093fb);
}

/* Clear Button */
.btn-clear {
    background: linear-gradient(135deg, #ff6b6b, #ee5a52);
}

.btn-clear:hover {
    background: linear-gradient(135deg, #ee5a52, #ff6b6b);
}

/* Suggestion Button */
.btn-suggestion {
    background: linear-gradient(135deg, #4ecdc4, #44a08d);
    margin: 5px;
    font-size: 14px;
    padding: 8px 16px;
}

.btn-suggestion:hover {
    background: linear-gradient(135deg, #44a08d, #4ecdc4);
}

/* Loading Animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    border-top-color: #fff;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Responsive Design */
@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    
    .gradio-container {
        margin: 10px;
        padding: 20px;
    }
    
    .chatbot .message.user {
        margin-left: 10%;
    }
    
    .chatbot .message.bot {
        margin-right: 10%;
    }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2, #667eea);
}

/* Floating Elements */
.floating {
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* Pulse Effect */
.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
"""

# Create the advanced interface
with gr.Blocks(css=advanced_css, title="Advanced FAQ Chatbot", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 class="floating">🚀 Advanced FAQ Chatbot</h1>
            <p class="description">Powered by AI • Built with ❤️ by Ibtissam • CodeAlpha Internship</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="💬 Chat History",
                height=400,
                show_label=True,
                container=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Type your message",
                    placeholder="Ask me anything about our services...",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    container=False,
                    scale=4
                )
                submit_btn = gr.Button("Send 📤", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                suggestion_btn = gr.Button("Get Suggestion 💡", variant="secondary")
        
        with gr.Column(scale=1):
            gr.HTML("""
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                    <h3 style="color: white; margin-top: 0;">📊 Quick Stats</h3>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0;">🎯 Categories: 8</p>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0;">❓ Total FAQs: 45+</p>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0;">🤖 AI Powered</p>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0;">⚡ Real-time</p>
                </div>
            """)
            
            gr.HTML("""
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;">
                    <h3 style="color: white; margin-top: 0;">🏷️ Categories</h3>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <span style="background: #667eea; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">Account</span>
                        <span style="background: #764ba2; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">Orders</span>
                        <span style="background: #f093fb; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">Shipping</span>
                        <span style="background: #f5576c; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">Returns</span>
                        <span style="background: #4ecdc4; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">Payments</span>
                        <span style="background: #45b7d1; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">Products</span>
                        <span style="background: #96ceb4; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">Support</span>
                        <span style="background: #feca57; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">About</span>
                    </div>
                </div>
            """)
    
    # Event handlers
    def submit_message(message, history):
        return get_best_answer(message, history)
    
    def suggest_question():
        return get_suggestion()
    
    # Bind events
    submit_btn.click(submit_message, [msg, chatbot], [chatbot, msg])
    msg.submit(submit_message, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])
    suggestion_btn.click(suggest_question, outputs=[msg])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )



