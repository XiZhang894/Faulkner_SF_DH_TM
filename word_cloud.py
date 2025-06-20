import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set the font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Read the text file
with open('topic_keywords.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Parse the file content
topics = {}
current_topic = None
for line in content.split('\n'):
    if line.startswith('Topic'):
        current_topic = line.strip()
        topics[current_topic] = {}
    elif current_topic and line.strip():
        match = re.match(r'(\w+)\s+([\d.]+)', line)
        if match:
            word = match.group(1)
            weight = float(match.group(2))
            # Filter out words with very low weights
            if weight > 0.001:
                topics[current_topic][word] = weight

# Generate word clouds for each topic and save to files
for topic, keywords in topics.items():
    if keywords:  # Check if there are words for the word cloud
        # Create a word cloud object, specifying the font path for Times New Roman
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='C:/Windows/Fonts/times.ttf')
        # Generate the word cloud
        wordcloud.generate_from_frequencies(keywords)
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for {topic}')
        plt.axis('off')
        # Save the word cloud to a file
        plt.savefig(f'word_cloud_{topic}.png', bbox_inches='tight')
        plt.close()
    else:
        print(f'Skipping empty topic: {topic}')
