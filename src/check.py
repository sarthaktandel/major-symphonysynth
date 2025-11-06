import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.remi_m_tokenizer import REMIMTokenizer

tokenizer = REMIMTokenizer()
tokens = tokenizer.midi_to_tokens("/Users/sarthak/Documents/melodygen/outputs/generated.mid")

# Count token types
from collections import Counter
counts = Counter(t for t in tokens if isinstance(t, str))
print(counts.most_common(20))