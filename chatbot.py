import tkinter as tk
from tkinter import scrolledtext
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NLP SETUP ---
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Internship Database
faq_data = {
    # --- SECTION 1: TIMELINES & ADMISSIONS ---
    "what is the process": "The process is: 1. Receive Selection Letter. 2. Complete tasks from your domain list. 3. Upload code to GitHub. 4. Record a demo video. 5. Submit via Google Form.",
    "what is the duration": "The CodeAlpha internship typically lasts for 4 weeks (1 month).",
    "start and end date": "Please refer to your Offer Letter for your specific batch dates. Most internships start at the beginning of the month.",
    "is it paid": "This is a virtual, unpaid internship designed to provide students with hands-on experience and portfolio-worthy projects.",
    "selection letter": "Selection letters are sent via email. If you haven't received it, check your spam folder or contact services@codealpha.tech.",

    # --- SECTION 2: THE PROCESS (MANDATORY) ---
    "how to start": "1. Check your task list. 2. Set up your development environment. 3. Build your projects. 4. Record and submit.",
    "how to submit tasks": "Push your code to GitHub, record a Loom video, share it on LinkedIn tagging @CodeAlpha, and fill out the submission form.",
    "deadline for submission": "All tasks must be submitted by the end date mentioned in your offer letter, usually by the last day of the month.",
    "task criteria": "You must complete at least 2 or 3 tasks from the list provided in your domain.",

    # --- SECTION 3: RECOGNITION & CERTIFICATES ---
    "will i get a certificate": "Yes, upon successful completion and verification of your tasks, you will receive an Internship Completion Certificate.",
    "letter of recommendation": "Top performers who go above and beyond the required tasks may be eligible for a Letter of Recommendation (LOR).",
    "when will i receive the certificate": "Certificates are usually processed and emailed within 15-20 days after the final submission deadline.",

    # --- SECTION 4: GENERAL AI & LOGIC ---
    "who created you": "I was created by a CodeAlpha intern using Python, NLTK, and Scikit-Learn to assist fellow interns.",
    "how do you work": "I use Natural Language Processing (NLP) and Cosine Similarity to match your questions with my knowledge base.",
    "hi": "Hi there! I'm AlphaIntel. I can help you with start dates, submission links, and certificate info. What's on your mind?",
    "hello": "Hi there! I'm AlphaIntel. I can help you with start dates, submission links, and certificate info. What's on your mind?",
    "bye": "Good luck with your submission! Don't forget to push your code to GitHub!"
}

def nlp_preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text) # Tokenization
    cleaned = [re.sub(r'[^\w\s]', '', t) for t in tokens if t.strip()] # Cleaning
    return " ".join(cleaned)

class AlphaIntelBot:
    def __init__(self, root):
        self.root = root
        self.root.title("AlphaIntel | Internship Navigator")
        self.root.geometry("500x650")
        self.root.configure(bg="#1e1e1e")

        # UI Header
        tk.Label(root, text="AlphaIntel AI Assistant", font=("Helvetica", 16, "bold"), bg="#1e1e1e", fg="#0078d4").pack(pady=10)
        tk.Label(root, text="Ask anything related to Internship", font=("Arial", 9), bg="#1e1e1e", fg="#aaaaaa").pack()

        # Chat History
        self.chat_display = scrolledtext.ScrolledText(root, state='disabled', bg="#2d2d2d", fg="white", font=("Arial", 11), relief="flat")
        self.chat_display.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        # Guidance
        tk.Label(root, text="TYPE YOUR QUESTION BELOW:", font=("Arial", 10, "bold"), bg="#1e1e1e", fg="#0078d4").pack(anchor="w", padx=15)

        input_frame = tk.Frame(root, bg="#1e1e1e")
        input_frame.pack(fill=tk.X, padx=15, pady=(5, 20))

        self.user_input = tk.Entry(input_frame, font=("Arial", 12), bg="#3d3d3d", fg="white", insertbackground="white", relief="flat")
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10)
        self.user_input.bind("<Return>", lambda e: self.get_response())

        self.send_btn = tk.Button(input_frame, text="ASK", command=self.get_response, bg="#0078d4", fg="white", font=("Arial", 10, "bold"), width=10, relief="flat")
        self.send_btn.pack(side=tk.RIGHT, padx=(10, 0))

        self.log("AlphaIntel", "System Ready. How can I guide you today?")

    def log(self, sender, msg):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {msg}\n\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def get_response(self):
        user_text = self.user_input.get().strip()
        if not user_text: return
        
        self.log("You", user_text)
        self.user_input.delete(0, tk.END)

        # 1. Prepare data for vectorization
        questions = list(faq_data.keys())
        processed_qs = [nlp_preprocess(q) for q in questions]
        processed_user = nlp_preprocess(user_text)

        # 2. Convert text to numerical vectors
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(processed_qs + [processed_user])
        
        # 3. Calculate similarity score
        scores = cosine_similarity(tfidf[-1], tfidf[:-1])
        idx = scores.argmax()
        best_score = scores[0][idx]
        
        # --- THE FIX: Higher threshold for fallback ---
        if best_score > 0.4: # Increased from 0.3 to 0.4 to filter out random questions
            response = faq_data[questions[idx]]
        else:
            response = "I'm sorry, I specialize in CodeAlpha internship details. I don't have information on that specific topic like weather or general news."
        
        self.log("AlphaIntel", response)

if __name__ == "__main__":
    root = tk.Tk()
    app = AlphaIntelBot(root)
    root.mainloop()