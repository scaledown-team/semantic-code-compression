from collections import Counter
import random
import re

import nltk
from nltk.util import ngrams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Download necessary NLTK resources
nltk.download('punkt')  # Only this is needed

def ngram_analysis(code_snippets, n=3):
    """
    Perform n-gram analysis on a list of code snippets.
    
    Args:
        code_snippets (list of str): List of code snippets as strings.
        n (int): The 'n' in n-grams.
        
    Returns:
        Counter: A counter object with n-gram frequencies.
    """
    # Combine all code snippets into a single string for easier tokenization
    all_code_text = "\n".join(code_snippets)

    # Tokenize the combined text into words or tokens.
    # Using a simple regex to split by non-alphanumeric characters and whitespace
    # This will also handle punctuation and newlines as token separators.
    tokens = re.findall(r'\w+', all_code_text.lower())

    # Generate n-grams
    all_ngrams = list(ngrams(tokens, n))

    # Count the frequency of each unique n-gram
    ngram_frequencies = Counter(all_ngrams)

    # Store the n-gram frequencies (already in the Counter object)
    # The Counter object `ngram_frequencies` serves as the storage.

    print(f"Generated {len(all_ngrams)} {n}-grams.")
    print(f"Found {len(ngram_frequencies)} unique {n}-grams.")
    print("\nMost common n-grams:")
    for ngram, freq in ngram_frequencies.most_common(20):
        print(f"{ngram}: {freq}")
    
    return ngram_frequencies

def generate_fake_candidates(real_token):
    """Generates a fake candidate token by adding a plausible suffix to the real token."""
    plausible_suffixes = ['_new', '_data', '_handler', '_v2', 'Service', 'Manager', 'Helper', 'Utils', 'Config']
    # Randomly select one or more suffixes (for simplicity, let's pick just one for now)
    selected_suffix = random.choice(plausible_suffixes)
    fake_candidate = real_token + selected_suffix
    return fake_candidate     

def score_candidate(masked_snippet, mask_token, candidate_label, ngram_frequencies, n):
    """
    Replace a [MASK_#] in a snippet with a candidate and score it using n-gram frequencies.

    Args:
        masked_snippet (str): Code containing [MASK_#] tokens.
        mask_token (str): The specific mask token to replace (e.g., "[MASK_1]").
        candidate_label (str): Candidate token to insert in place of the mask.
        ngram_frequencies (Counter): Precomputed n-gram frequencies.
        n (int): Size of the n-grams.

    Returns:
        int: Total n-gram score for this candidate.
    """
    # Replace the mask token with the candidate label
    # Replace the mask token with the candidate
    candidate_snippet = masked_snippet.replace(mask_token, candidate_label)

    # Tokenize the snippet
    tokens = re.findall(r'\w+', candidate_snippet.lower())

    # Generate n-grams
    snippet_ngrams = list(ngrams(tokens, n))

    # Tokenize candidate for matching within n-grams
    candidate_tokens = re.findall(r'\w+', candidate_label.lower())

    # Sum frequencies of n-grams that contain any token from the candidate
    total_score = sum(ngram_frequencies.get(gram, 0)
                      for gram in snippet_ngrams
                      if any(c in gram for c in candidate_tokens))
    return total_score

def rank_candidates(masked_snippet, mask_token, real_token, ngram_frequencies, n=3):
    """
    Rank candidates for a masked token in a code snippet using n-gram frequencies.

    Args:
        masked_snippet (str): Code containing [MASK_#] tokens.
        mask_token (str): The specific mask token to replace (e.g., "[MASK_1]").
        real_token (str): The actual token that should replace the mask.
        ngram_frequencies (Counter): Precomputed n-gram frequencies.
        n (int): Size of the n-grams.

    Returns:
        tuple of list of tuples: Ranked list of both the real and fake candidates with their scores.
    """
    # Generate fake candidates
    fake_candidates = [generate_fake_candidates(real_token) for _ in range(5)]
    
    # Include the real token as a candidate
    real_candidates = [real_token]

    # Score each candidate
    real_scores = []
    fake_scores = []
    candidate_scores = []
    for candidate in real_candidates:
        score = score_candidate(masked_snippet, mask_token, candidate, ngram_frequencies, n)
        real_scores.append((candidate, score))
    
    for candidate in fake_candidates:
        score = score_candidate(masked_snippet, mask_token, candidate, ngram_frequencies, n)
        fake_scores.append((candidate, score))

    # Rank both real and fake candidates by score (higher is better)
    real_candidates = sorted(real_scores, key=lambda x: x[1], reverse=True)
    fake_candidates = sorted(fake_scores, key=lambda x: x[1], reverse=True)

    # Print the ranked real candidates
    print("\nRanked Candidates:")
    for candidate, score in real_candidates:
        print(f"{candidate}: {score}")
    
    # Print the ranked fake candidates
    for candidate, score in fake_candidates:
        print(f"{candidate}: {score}")

    return real_candidates, fake_candidates

def compare_rankings(real_candidates, fake_candidates):
    """
    Compare the rankings of real and fake candidates.

    Args:
        real_candidates (list of tuples): Ranked list of real candidates with their scores.
        fake_candidates (list of tuples): Ranked list of fake candidates with their scores.

    Returns:
        Percentage of real candidates ranked higher than fake candidates.
    """
    
    # Extract scores for comparison
    real_scores = [score for _, score in real_candidates]
    fake_scores = [score for _, score in fake_candidates]

    # Calculate the total number of scored pairs
    total_scored_pairs = min(len(real_scores), len(fake_scores))
    real_scores_higher_count = 0

    # Ensure we only compare up to the minimum length of the two lists
    for i in range(total_scored_pairs):
        if real_scores[i] > fake_scores[i]:
            real_scores_higher_count += 1

    # Compute the percentage of cases where the real score is higher
    percentage_real_higher = (real_scores_higher_count / total_scored_pairs) * 100 if total_scored_pairs > 0 else 0

    # Print the calculated percentage
    print(f"\nPercentage of cases where the real label scored higher than the fake label: {percentage_real_higher:.2f}%")
    return percentage_real_higher

def plot_rankings(real_candidates, fake_candidates):
    """
    Plot the rankings of real and fake candidates.

    Args:
        real_candidates (list of tuples): Ranked list of real candidates with their scores.
        fake_candidates (list of tuples): Ranked list of fake candidates with their scores.
    """
    # Extract scores for plotting
    real_scores = [score for _, score in real_candidates]
    fake_scores = [score for _, score in fake_candidates]

    # Create a DataFrame for easier plotting
    df_real = pd.DataFrame(real_scores, columns=['Score'])
    df_real['Type'] = 'Real'
    df_fake = pd.DataFrame(fake_scores, columns=['Score'])
    df_fake['Type'] = 'Fake'
    df = pd.concat([df_real, df_fake])

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Type', y='Score', data=df)
    plt.title('Candidate Score Distribution')
    plt.ylabel('Score')
    plt.xlabel('Candidate Type')
    plt.show()

    # Create a histogram for real scores
    plt.figure(figsize=(10, 6))
    sns.histplot(real_scores, kde=True, color='skyblue')
    plt.title('Distribution of Real Label Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()

    # Create a histogram for fake scores
    plt.figure(figsize=(10, 6))
    sns.histplot(fake_scores, kde=True, color='salmon')
    plt.title('Distribution of Fake Label Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()

    # Optional: Create a combined histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(real_scores, kde=True, color='skyblue', label='Real Scores', alpha=0.6)
    sns.histplot(fake_scores, kde=True, color='salmon', label='Fake Scores', alpha=0.6)
    plt.title('Distribution of Real vs. Fake Label Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()