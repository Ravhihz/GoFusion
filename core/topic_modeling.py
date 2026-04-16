import numpy as np
from collections import Counter, defaultdict
from .models import PreprocessedTweet, Label, Topic


class SimpleLDA:
    """
    Latent Dirichlet Allocation (LDA) for topic modeling
    Implementation based on Gibbs Sampling
    """
    
    def __init__(self, n_topics=3, alpha=0.1, beta=0.01, n_iterations=1000):
        """
        Args:
            n_topics: Number of topics to extract
            alpha: Document-topic prior (Dirichlet parameter)
            beta: Topic-word prior (Dirichlet parameter)
            n_iterations: Number of Gibbs sampling iterations
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iterations = n_iterations
        
        # Model parameters
        self.topic_word_dist = None  # φ (phi): P(word|topic)
        self.doc_topic_dist = None   # θ (theta): P(topic|doc)
        self.vocabulary = None
        self.doc_topic_counts = None
        self.topic_word_counts = None
        self.topic_counts = None
    
    def fit(self, documents):
        """
        Fit LDA model using Gibbs Sampling
        
        Args:
            documents: list of list of words
                Example: [['good', 'book'], ['bad', 'quality'], ...]
        
        Returns:
            self
        """
        print(f"\n[LDA] Starting LDA topic modeling...")
        print(f"[LDA] Documents: {len(documents)}")
        print(f"[LDA] Topics: {self.n_topics}")
        print(f"[LDA] Iterations: {self.n_iterations}")
        
        # Build vocabulary
        all_words = [word for doc in documents for word in doc]
        word_counts = Counter(all_words)
        
        # Filter words with min frequency
        min_freq = 2
        self.vocabulary = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
        
        n_docs = len(documents)
        n_words = len(self.vocabulary)
        
        print(f"[LDA] Vocabulary size: {n_words} (after filtering)")
        
        # Initialize count matrices
        self.doc_topic_counts = np.zeros((n_docs, self.n_topics))
        self.topic_word_counts = np.zeros((self.n_topics, n_words))
        self.topic_counts = np.zeros(self.n_topics)
        
        # Initialize topic assignments randomly
        doc_topics = []
        for doc_idx, doc in enumerate(documents):
            topics = []
            for word in doc:
                if word not in self.vocabulary:
                    continue
                
                word_idx = self.vocabulary[word]
                topic = np.random.randint(0, self.n_topics)
                topics.append(topic)
                
                # Update counts
                self.doc_topic_counts[doc_idx, topic] += 1
                self.topic_word_counts[topic, word_idx] += 1
                self.topic_counts[topic] += 1
            
            doc_topics.append(topics)
        
        # Gibbs Sampling
        print(f"[LDA] Running Gibbs sampling...")
        
        for iteration in range(self.n_iterations):
            for doc_idx, doc in enumerate(documents):
                word_idx_in_doc = 0
                
                for word in doc:
                    if word not in self.vocabulary:
                        continue
                    
                    word_idx = self.vocabulary[word]
                    old_topic = doc_topics[doc_idx][word_idx_in_doc]
                    
                    # Remove current topic assignment
                    self.doc_topic_counts[doc_idx, old_topic] -= 1
                    self.topic_word_counts[old_topic, word_idx] -= 1
                    self.topic_counts[old_topic] -= 1
                    
                    # Calculate topic probabilities
                    # P(z|w,d) ∝ P(w|z) * P(z|d)
                    doc_topic_prob = (
                        (self.doc_topic_counts[doc_idx] + self.alpha) / 
                        (np.sum(self.doc_topic_counts[doc_idx]) + self.n_topics * self.alpha)
                    )
                    
                    topic_word_prob = (
                        (self.topic_word_counts[:, word_idx] + self.beta) /
                        (self.topic_counts + n_words * self.beta)
                    )
                    
                    topic_prob = doc_topic_prob * topic_word_prob
                    topic_prob /= np.sum(topic_prob)
                    
                    # Sample new topic
                    new_topic = np.random.choice(self.n_topics, p=topic_prob)
                    doc_topics[doc_idx][word_idx_in_doc] = new_topic
                    
                    # Update counts
                    self.doc_topic_counts[doc_idx, new_topic] += 1
                    self.topic_word_counts[new_topic, word_idx] += 1
                    self.topic_counts[new_topic] += 1
                    
                    word_idx_in_doc += 1
            
            if (iteration + 1) % 100 == 0:
                perplexity = self.calculate_perplexity(documents)
                print(f"[LDA] Iteration {iteration + 1}/{self.n_iterations}, Perplexity: {perplexity:.2f}")
        
        # Calculate final distributions
        self._calculate_distributions(n_words)
        
        print(f"[LDA] Topic modeling complete!\n")
        return self
    
    def _calculate_distributions(self, n_words):
        """Calculate final topic-word and doc-topic distributions"""
        # Topic-word distribution: P(word|topic)
        self.topic_word_dist = (self.topic_word_counts + self.beta) / ( # type: ignore
            self.topic_counts[:, np.newaxis] + n_words * self.beta # type: ignore
        )
        
        # Document-topic distribution: P(topic|doc)
        self.doc_topic_dist = (self.doc_topic_counts + self.alpha) / ( # type: ignore
            np.sum(self.doc_topic_counts, axis=1)[:, np.newaxis] + self.n_topics * self.alpha # type: ignore
        )
    
    def get_top_words(self, n_words=10):
        """
        Get top words for each topic
        
        Args:
            n_words: Number of top words to return per topic
        
        Returns:
            list: List of lists containing top words per topic
        """
        top_words = []
        vocab_list = list(self.vocabulary.keys()) # type: ignore
        
        for topic_idx in range(self.n_topics):
            word_probs = self.topic_word_dist[topic_idx] # type: ignore
            top_indices = word_probs.argsort()[-n_words:][::-1]
            top_words.append([vocab_list[idx] for idx in top_indices])
        
        return top_words
    
    def get_topic_word_weights(self, n_words=10):
        """
        Get top words with their weights for each topic
        
        Args:
            n_words: Number of top words per topic
        
        Returns:
            dict: {topic_idx: {word: weight}}
        """
        topic_weights = {}
        vocab_list = list(self.vocabulary.keys()) # type: ignore
        
        for topic_idx in range(self.n_topics):
            word_probs = self.topic_word_dist[topic_idx] # type: ignore
            top_indices = word_probs.argsort()[-n_words:][::-1]
            
            topic_weights[topic_idx] = {
                vocab_list[idx]: float(word_probs[idx])
                for idx in top_indices
            }
        
        return topic_weights
    
    def calculate_perplexity(self, documents):
        """
        Calculate perplexity of the model
        Lower perplexity indicates better model fit
        
        Args:
            documents: list of list of words
        
        Returns:
            float: Perplexity value
        """
        log_likelihood = 0
        total_words = 0
        
        for doc_idx, doc in enumerate(documents):
            for word in doc:
                if word not in self.vocabulary:
                    continue
                
                word_idx = self.vocabulary[word] # type: ignore
                
                # P(w|d) = Σ_z P(w|z) * P(z|d)
                word_prob = 0
                for topic_idx in range(self.n_topics):
                    word_prob += (
                        self.topic_word_dist[topic_idx, word_idx] * # type: ignore
                        self.doc_topic_dist[doc_idx, topic_idx] # type: ignore
                    )
                
                log_likelihood += np.log(word_prob + 1e-10)
                total_words += 1
        
        perplexity = np.exp(-log_likelihood / total_words)
        return perplexity


def extract_sentiment_topics(dataset_id, sentiment_type='positive', n_topics=3):
    """
    Extract topics from tweets with specific sentiment using LDA
    
    Args:
        dataset_id: Dataset ID
        sentiment_type: 'positive', 'negative', or 'neutral'
        n_topics: Number of topics to extract
    
    Returns:
        tuple: (topics, perplexity, topic_info)
    """
    print(f"\n[TOPIC EXTRACTION] Starting for {sentiment_type} sentiment...")
    
    # Get labeled tweets with specific sentiment
    labeled_tweets = Label.objects.filter(
        tweet__dataset_id=dataset_id,
        sentiment=sentiment_type
    )
    
    if not labeled_tweets.exists():
        return None, None, {"error": f"No {sentiment_type} tweets found"}
    
    # Get preprocessed tokens
    documents = []
    tweet_ids = []
    
    for label in labeled_tweets:
        preprocessed = PreprocessedTweet.objects.filter(tweet=label.tweet).first()
        if preprocessed and preprocessed.tokens and len(preprocessed.tokens) > 0:
            documents.append(preprocessed.tokens)
            tweet_ids.append(label.tweet.id) # type: ignore
    
    if len(documents) < 10:
        return None, None, {
            "error": f"Not enough {sentiment_type} documents (found {len(documents)}, need at least 10)"
        }
    
    print(f"[TOPIC EXTRACTION] Processing {len(documents)} {sentiment_type} documents")
    
    # Apply LDA
    lda = SimpleLDA(n_topics=n_topics, alpha=0.1, beta=0.01, n_iterations=500)
    lda.fit(documents)
    
    # Get results
    top_words = lda.get_top_words(n_words=10)
    topic_weights = lda.get_topic_word_weights(n_words=10)
    perplexity = lda.calculate_perplexity(documents)
    
    # Save to database
    saved_topics = []
    for topic_idx, words in enumerate(top_words):
        topic = Topic.objects.create(
            name=f"Topic {topic_idx + 1} ({sentiment_type})",
            sentiment_type=sentiment_type,
            top_words=words,
            word_weights=topic_weights[topic_idx],
            document_distribution=lda.doc_topic_dist[:, topic_idx].tolist() # type: ignore
        )
        saved_topics.append(topic)
        
        print(f"[TOPIC {topic_idx + 1}] Top words: {', '.join(words[:5])}")
    
    topic_info = {
        'sentiment_type': sentiment_type,
        'n_topics': n_topics,
        'n_documents': len(documents),
        'perplexity': float(perplexity),
        'topics': [
            {
                'topic_id': topic_idx + 1,
                'top_words': words,
                'weights': topic_weights[topic_idx]
            }
            for topic_idx, words in enumerate(top_words)
        ]
    }
    
    print(f"[TOPIC EXTRACTION] Complete! Perplexity: {perplexity:.2f}\n")
    
    return saved_topics, perplexity, topic_info


def extract_all_sentiment_topics(dataset_id, n_topics=3):
    """
    Extract topics for all sentiment types (positive, negative, neutral)
    
    Args:
        dataset_id: Dataset ID
        n_topics: Number of topics per sentiment
    
    Returns:
        dict: Results for each sentiment type
    """
    results = {}
    
    for sentiment_type in ['positive', 'negative', 'neutral']:
        topics, perplexity, info = extract_sentiment_topics(
            dataset_id=dataset_id,
            sentiment_type=sentiment_type,
            n_topics=n_topics
        )
        
        results[sentiment_type] = {
            'topics': topics,
            'perplexity': perplexity,
            'info': info
        }
    
    return results