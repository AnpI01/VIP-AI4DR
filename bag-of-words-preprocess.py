import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer

# --- 1. THE NUCLEAR RESET ---
# This command forces VS Code to close every single open image window
plt.close('all') 

# --- 2. LOAD DATA ---
print("Step 1: Loading Data...")
try:
    df = pd.read_csv('qprop_data/proppy_1.0.train.tsv', sep='\t', header=None)
    df.rename(columns={0: 'article_text', 14: 'propaganda_label'}, inplace=True)
except FileNotFoundError:
    print("❌ File not found.")
    exit()

# --- 3. CLEANING ---
print("Step 2: Cleaning Text...")
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

def simple_clean(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return " ".join([w for w in text.split() if w not in stop_words])

df['clean_text'] = df['article_text'].apply(simple_clean)

# --- 4. VECTORIZING ---
print("Step 3: Building Vectors...")
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text'])
vocab = vectorizer.get_feature_names_out()

# --- 5. GENERATING GRAPHS (With "Memory Wipe") ---
sns.set_style("whitegrid")

def draw_clean_graphs():
    # --- GRAPH 1: Class Balance ---
    print("   🎨 Drawing Graph 1...")
    plt.close('all') # Double check: Wipe memory again
    fig1 = plt.figure(figsize=(10, 6)) # Make it wide
    
    # Create temp data just for this graph
    plot_df = df.copy()
    plot_df['Label'] = plot_df['propaganda_label'].map({-1: 'News', 1: 'Propaganda'})
    
    ax = sns.countplot(x='Label', data=plot_df, palette='coolwarm', hue='Label', legend=False)
    plt.title('Class Distribution (News vs Propaganda)', fontsize=15)
    plt.ylabel('Number of Articles')
    plt.xlabel('')
    
    # Add numbers
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=12, padding=3)
        
    plt.tight_layout()
    plt.savefig('qprop_graph_1_balance.png')
    print("     ✅ Saved Graph 1")
    plt.clf() # Clear Figure
    plt.close(fig1) # Close File

    # --- GRAPH 2: Top Words ---
    print("   🎨 Drawing Graph 2...")
    plt.close('all') # Wipe memory again
    fig2 = plt.figure(figsize=(12, 8)) # Make it VERY wide to fit words
    
    # Calculate word counts
    counts = pd.DataFrame(X.toarray(), columns=vocab).sum()
    top20 = counts.sort_values(ascending=False).head(20)
    
    sns.barplot(x=top20.values, y=top20.index, palette='magma', hue=top20.index, legend=False)
    plt.title('Top 20 Words in Dataset', fontsize=15)
    plt.xlabel('Count')
    
    plt.tight_layout()
    plt.savefig('qprop_graph_2_words.png')
    print("     ✅ Saved Graph 2")
    plt.clf()
    plt.close(fig2)

    # --- GRAPH 3: Article Fingerprint ---
    print("   🎨 Drawing Graph 3...")
    plt.close('all') # Wipe memory again
    fig3 = plt.figure(figsize=(12, 8))
    
    # Get first article vector
    vector_array = X[0].toarray().flatten()
    article_data = pd.DataFrame({'word': vocab, 'count': vector_array})
    active_words = article_data[article_data['count'] > 0].sort_values(by='count', ascending=False).head(20)
    
    sns.barplot(x='count', y='word', data=active_words, palette='viridis', hue='word', legend=False)
    plt.title('Fingerprint of First Article', fontsize=15)
    
    plt.tight_layout()
    plt.savefig('qprop_graph_3_fingerprint.png')
    print("     ✅ Saved Graph 3")
    plt.clf()
    plt.close(fig3)

# Run the function
draw_clean_graphs()
print("\n🚀 DONE! Check your folder for the 3 new images.")
try:
    df = pd.read_csv('qprop_data/proppy_1.0.train.tsv', sep='\t', header=None)
    # We only need the text (Col 0) and the label (Col 14)
    df = df[[0, 14]].copy()
    df.rename(columns={0: 'text', 14: 'label'}, inplace=True)
    
    # Map labels to text for readability
    df['Label Type'] = df['label'].map({-1: 'Real News', 1: 'Propaganda'})
    
    print(f"✅ Loaded {len(df)} articles.")
except FileNotFoundError:
    print("❌ File not found.")
    exit()

# --- 3. CALCULATE WORD COUNTS ---
print("⏳ Counting words in every article (this is fast)...")

# This counts the number of spaces + 1 in every row
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

# --- 4. PRINT THE STATISTICS (The Numbers) ---
print("\n" + "="*40)
print("📊 AVERAGE ARTICLE LENGTH RESULTS")
print("="*40)

# Calculate averages
stats = df.groupby('Label Type')['word_count'].agg(['mean', 'median', 'min', 'max'])
print(stats)
print("\n" + "="*40)

# --- 5. DRAW THE HISTOGRAM ---
print("🎨 Drawing Distribution Graph...")
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Histogram with a limit (X-axis cut off at 2000 words to make it readable)
sns.histplot(data=df, x='word_count', hue='Label Type', kde=True, bins=50, palette={'Real News': 'blue', 'Propaganda': 'red'}, alpha=0.5)

plt.xlim(0, 2000) # Most articles are under 2000 words
plt.title('Distribution of Article Lengths: News vs. Propaganda', fontsize=14)
plt.xlabel('Number of Words in Article')
plt.ylabel('Number of Articles')

plt.tight_layout()
plt.savefig('qprop_length_distribution.png', dpi=300)
print("✅ Saved Graph: qprop_length_distribution.png")