import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt

# ---- MODEL PARAMETERS ---- #
# Architecture
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1

# Training
BATCH_SIZE = 32
EPOCHS = 100
WARMUP_STEPS = 4000

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using CPU mode" if device.type == 'cpu' else "Using CUDA mode")

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# ============================
# DATASET: LOGICAL STATEMENTS
# ============================

# Simple statements for initial training
simple_statements = [
    ("If X is A, then X is B.", "Thing1 is A.", "Thing1 is B."),
    ("If X is even, it is divisible by 2.", "6 is even.", "6 is divisible by 2."),
    ("When something is cold, it has low temperature.", "Ice is cold.", "Ice has low temperature."),
]

# Medium complexity statements
medium_statements = [
    ("If a number is greater than 10, it is greater than 5.", "15 is greater than 10.", "15 is greater than 5."),
    ("If a number is greater than 20, it is greater than 15.", "25 is greater than 20.", "25 is greater than 15."),
    ("When metal is heated, it expands.", "The ring is heated.", "The ring expands."),
]

# Complex statements with multi-step reasoning
complex_statements = [
    ("If A implies B, and B implies C, then A implies C.", 
     "Rain implies wet ground, and wet ground implies mud.", 
     "Rain implies mud."),
    ("If X leads to Y, and Y leads to Z, then X leads to Z.",
     "Heat leads to expansion, and expansion leads to pressure.",
     "Heat leads to pressure."),
]

def log_results(message, filename="reasoning_results.txt"):
    """Log results to file with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{message}\n")

def pad_sequence(seq, max_len=64):
    return seq + [vocab["<PAD>"]] * (max_len - len(seq))

class LogicalDataset(Dataset):
    def __init__(self, statements):
        self.data = [
            (pad_sequence(encode(p1)), pad_sequence(encode(p2)), pad_sequence(encode(out)))
            for p1, p2, out in statements
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        p1, p2, out = self.data[idx]
        return (
            torch.tensor(p1, dtype=torch.long),
            torch.tensor(p2, dtype=torch.long),
            torch.tensor(out, dtype=torch.long)
        )

def create_vocab(statements):
    """Create vocabulary from all statements and test cases"""
    vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<EOS>": 3}
    
    # Add test cases to ensure their vocabulary is included
    test_statements = [
        # Novel number tests
        ("If a number is even, it is divisible by 2.", "14 is even.", "14 is divisible by 2."),
        ("If a number is greater than 20, it is greater than 15.", "25 is greater than 20.", "25 is greater than 15."),
        ("When something is cold, it has low temperature.", "The ice is cold.", "The ice has low temperature."),
        ("X always results in Y.", "The metal is heated.", "The metal expands."),
        
        # Multi-step logic
        ("If A is more than B, and B is more than C, then A is more than C.", 
         "15 is more than 10, and 10 is more than 5.", 
         "15 is more than 5."),
        ("If it rains, the ground gets wet. If the ground is wet, it becomes slippery.", 
         "It is raining.", 
         "The ground becomes slippery."),
        
        # Minimal context
        ("If X is A, then X is B.", "Thing1 is A.", "Thing1 is B."),
        ("When X is not Y, X cannot be Z.", "Thing2 is not Y.", "Thing2 cannot be Z."),
        
        # Additional test cases
        ("If X is not Y, and all Z are Y, then X is not Z.", 
         "A circle is not a polygon, and all squares are polygons.", 
         "A circle is not a square."),
        ("Either a number is positive or negative, but not both.", 
         "3 is positive.", 
         "3 is not negative."),
    ]
    
    # Add training data that matches test case patterns
    training_statements = [
        # Number comparisons
        ("If a number is greater than 10, it is greater than 5.", "12 is greater than 10.", "12 is greater than 5."),
        ("If a number is greater than 30, it is greater than 25.", "35 is greater than 30.", "35 is greater than 25."),
        ("If a number is even, it is divisible by 2.", "8 is even.", "8 is divisible by 2."),
        ("If a number is even, it is divisible by 2.", "6 is even.", "6 is divisible by 2."),
        
        # Temperature relationships
        ("When something is hot, it has high temperature.", "The sun is hot.", "The sun has high temperature."),
        ("When something is cold, it has low temperature.", "Snow is cold.", "Snow has low temperature."),
        ("When something is warm, it has medium temperature.", "Tea is warm.", "Tea has medium temperature."),
        
        # Physical state changes
        ("When metal is heated, it expands.", "The ring is heated.", "The ring expands."),
        ("When water is frozen, it becomes ice.", "The water is frozen.", "The water becomes ice."),
        ("When ice is heated, it melts.", "The ice is heated.", "The ice melts."),
        
        # Multi-step reasoning
        ("If A implies B, and B implies C, then A implies C.", 
         "Rain implies wet ground, and wet ground implies mud.", 
         "Rain implies mud."),
        ("If X leads to Y, and Y leads to Z, then X leads to Z.",
         "Heat leads to expansion, and expansion leads to pressure.",
         "Heat leads to pressure."),
         
        # Abstract reasoning
        ("If X is A, then X is B.", "Object1 is A.", "Object1 is B."),
        ("If X is A, then X is B.", "Item1 is A.", "Item1 is B."),
        ("When X is Y, X cannot be Z.", "Thing3 is Y.", "Thing3 cannot be Z."),
        ("When X is Y, X cannot be Z.", "Object2 is Y.", "Object2 cannot be Z."),
        
        # Shape relationships
        ("All squares are rectangles.", "This is a square.", "This is a rectangle."),
        ("No circles are polygons.", "This is a circle.", "This is not a polygon."),
        
        # Positive/Negative numbers
        ("A number cannot be both positive and negative.", "5 is positive.", "5 is not negative."),
        ("A number cannot be both positive and negative.", "7 is positive.", "7 is not negative."),
    ]
    
    all_statements = statements + test_statements + training_statements
    
    # First collect all unique words
    words = set()
    for rule, fact, conclusion in all_statements:
        # Split on spaces and punctuation
        for text in [rule.lower(), fact.lower(), conclusion.lower()]:
            # Handle punctuation carefully
            text = text.replace(".", " . ")
            text = text.replace(",", " , ")
            text = text.replace("(", " ( ")
            text = text.replace(")", " ) ")
            words.update(text.split())
    
    # Add numbers 0-100 to handle numerical reasoning
    for i in range(101):
        words.add(str(i))
    
    # Add common logical words and their variations
    logical_words = {
        # Basic logical operators
        "if", "then", "when", "implies", "leads", "to", "and", "or", "not",
        "all", "some", "none", "is", "are", "be", "being", "has", "have",
        "cannot", "must", "can", "may", "will", "would",
        
        # Comparisons
        "greater", "less", "equal", "than", "more", "most", "least",
        "above", "below", "under", "over", "between",
        
        # Mathematical concepts
        "divisible", "by", "even", "odd", "multiple", "factor",
        "number", "numbers", "numeric", "numerical",
        
        # Relationships
        "part", "of", "in", "within", "contains", "includes",
        "belongs", "member", "subset", "element",
        
        # States and changes
        "becomes", "gets", "turns", "changes", "transforms",
        "remains", "stays", "keeps", "maintains",
        
        # Properties
        "true", "false", "correct", "incorrect", "valid", "invalid",
        "possible", "impossible", "necessary", "sufficient",
        
        # Physical properties
        "temperature", "hot", "cold", "warm", "cool",
        "wet", "dry", "solid", "liquid", "gas",
        "high", "low", "medium",
        
        # Shapes and geometry
        "circle", "square", "triangle", "rectangle", "polygon",
        "shape", "line", "point", "angle", "side",
        
        # Common nouns
        "thing", "thing1", "thing2", "thing3", "object", "object1", "object2",
        "item", "item1", "item2", "metal", "ice", "water", "ground", "surface",
        
        # States
        "positive", "negative", "neutral", "zero",
        "heated", "cooled", "frozen", "melted",
        "slippery", "rough", "smooth",
        
        # Conjunctions and prepositions
        "but", "however", "although", "unless", "except",
        "both", "either", "neither", "nor",
        
        # Articles and pronouns
        "a", "an", "the", "this", "that", "these", "those",
        "it", "its", "they", "their", "them"
    }
    words.update(logical_words)
    
    # Sort words for consistent indexing
    for word in sorted(words):
        if word and word not in vocab:  # Ensure word is not empty
            vocab[word] = len(vocab)
    
    print("\nVocabulary Preview:")
    print(f"Total vocab size: {len(vocab)}")
    print("Special tokens:", {k: v for k, v in vocab.items() if k.startswith('<')})
    print("Sample logical tokens:", {k: v for k, v in vocab.items() if k in ["if", "then", "and", "or", "not"]})
    
    # Create reverse mapping globally
    global id2word
    id2word = {v: k for k, v in vocab.items()}
    
    return vocab

def encode(text):
    """Encode text to token indices"""
    text = text.lower().strip()
    # Handle punctuation carefully
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    
    # Split into words and map to indices
    words = text.split()
    indices = []
    for word in words:
        if word in vocab:
            indices.append(vocab[word])
        else:
            indices.append(vocab["<UNK>"])
    
    # Add start and end tokens
    return [vocab["<START>"]] + indices + [vocab["<EOS>"]]

def decode(indices):
    """Decode token indices back to text"""
    if not indices:
        return ""
        
    # Skip start token if present
    if indices[0] == vocab["<START>"]:
        indices = indices[1:]
    # Skip end token if present    
    if indices and indices[-1] == vocab["<EOS>"]:
        indices = indices[:-1]
    
    # Convert indices to words using global id2word mapping
    words = []
    for idx in indices:
        if idx in id2word:
            token = id2word[idx]
            # Skip special tokens
            if not (token.startswith("<") and token.endswith(">")):
                words.append(token)
        else:
            print(f"Warning: Unknown index {idx} encountered during decoding")
    
    # Join words and fix punctuation spacing
    text = " ".join(words)
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    
    return text

def print_vocab_stats():
    """Print vocabulary statistics and sample entries"""
    print("\nVocabulary Statistics:")
    print(f"Total vocabulary size: {len(vocab)}")
    print("\nSpecial tokens:", {k: v for k, v in vocab.items() if k.startswith('<')})
    print("\nSample entries:")
    sample_words = ["if", "then", "is", "not", "greater", "than", "number", "temperature"]
    for word in sample_words:
        if word in vocab:
            print(f"'{word}': {vocab[word]}")
        else:
            print(f"'{word}': Not in vocabulary!")
    
    # Test encoding/decoding
    test_texts = [
        "If a number is even, it is divisible by 2.",
        "The ice has low temperature.",
        "15 is greater than 10."
    ]
    
    print("\nEncoding/Decoding Tests:")
    for text in test_texts:
        encoded = encode(text)
        decoded = decode(encoded)
        print(f"\nOriginal: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")

# ---- MODEL ARCHITECTURE ---- #
class LogicalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer (no LogSoftmax, just linear projection)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_mask(self, src, tgt):
        """Create masks for transformer
        
        Args:
            src: source sequence [batch_size, src_len]
            tgt: target sequence [batch_size, tgt_len]
        Returns:
            src_padding_mask: mask for src padding tokens [batch_size, src_len]
            tgt_padding_mask: mask for tgt padding tokens [batch_size, tgt_len]
            tgt_mask: mask for future tokens [tgt_len, tgt_len]
        """
        # Create padding masks (True indicates position to mask)
        src_padding_mask = (src == vocab["<PAD>"]).to(src.device)
        tgt_padding_mask = (tgt == vocab["<PAD>"]).to(tgt.device)
        
        # Create causal mask for decoder (prevent attending to future tokens)
        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) * float('-inf'), diagonal=1)
        tgt_mask = tgt_mask.to(tgt.device)
        
        return src_padding_mask, tgt_padding_mask, tgt_mask
    
    def forward(self, src, tgt):
        """Forward pass
        
        Args:
            src: source sequence [batch_size, src_len]
            tgt: target sequence [batch_size, tgt_len]
        Returns:
            output: sequence of logits [batch_size, tgt_len, vocab_size]
        """
        # Create masks
        src_padding_mask, tgt_padding_mask, tgt_mask = self.create_mask(src, tgt)
        
        # Embed and add positional encoding
        src_embedded = self.pos_encoder(self.embedding(src))  # [batch_size, src_len, d_model]
        tgt_embedded = self.pos_encoder(self.embedding(tgt))  # [batch_size, tgt_len, d_model]
        
        # Transformer forward pass
        transformer_out = self.transformer(
            src_embedded,
            tgt_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask
        )
        
        # Output projection (return raw logits)
        return self.output_layer(transformer_out)  # [batch_size, tgt_len, vocab_size]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # Changed shape for batch_first
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]  # Changed indexing for batch_first
        return self.dropout(x)

class LogicalConsistencyLoss(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.base_criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
        
        # Initialize loss statistics for monitoring
        self.running_base_loss = 0.0
        self.running_reward = 0.0
        self.num_updates = 0
        self.reward_scale = 0.1  # Start with small scale, will be adjusted dynamically
        
    def forward(self, output, target, src, return_components=False):
        """Calculate loss with dynamic reward scaling
        
        Args:
            output: Model output logits [batch_size, seq_len, vocab_size]
            target: Target indices [batch_size, seq_len]
            src: Source indices [batch_size, src_len]
            return_components: Return loss components (default: False)
        Returns:
            total_loss: Total loss (base loss - reward)
            loss_components: Dictionary with base loss, reward, and reward scale (if return_components=True)
        """
        # Ensure target has correct shape
        if len(target.shape) == 3:  # If target includes one-hot encoding
            target = target.argmax(dim=-1)  # Convert to indices [batch_size, seq_len]
        
        # Reshape output for cross entropy
        B, T, V = output.shape  # Batch, Time, Vocab
        output_flat = output.reshape(-1, V)  # [batch_size * seq_len, vocab_size]
        target_flat = target.reshape(-1)     # [batch_size * seq_len]
        
        # Calculate base loss (cross entropy)
        base_loss = self.base_criterion(output_flat, target_flat)
        
        # Calculate logical consistency reward
        logical_reward = self._calculate_logical_reward(output, target, src)
        
        # Update running statistics
        self.num_updates += 1
        self.running_base_loss = 0.95 * self.running_base_loss + 0.05 * base_loss.item()
        self.running_reward = 0.95 * self.running_reward + 0.05 * logical_reward.item()
        
        # Dynamically adjust reward scale to maintain balance with base loss
        if self.num_updates > 100:  # Wait for some initial statistics
            target_ratio = 0.2  # Reward should be ~20% of base loss
            current_ratio = self.running_reward / (self.running_base_loss + 1e-8)
            if current_ratio > target_ratio * 1.2:  # Too high
                self.reward_scale *= 0.95
            elif current_ratio < target_ratio * 0.8:  # Too low
                self.reward_scale *= 1.05
            self.reward_scale = min(max(self.reward_scale, 0.01), 0.5)  # Keep in reasonable range
        
        # Combine losses with dynamic scaling
        total_loss = base_loss - self.reward_scale * logical_reward
        
        # Log statistics periodically
        if self.num_updates % 100 == 0:
            print(f"\nLoss Statistics (update {self.num_updates}):")
            print(f"Base Loss: {self.running_base_loss:.4f}")
            print(f"Logical Reward: {self.running_reward:.4f}")
            print(f"Reward Scale: {self.reward_scale:.4f}")
            print(f"Total Loss: {total_loss.item():.4f}")
            print(f"Output shape: {output.shape}, Target shape: {target.shape}")
            print(f"Flattened shapes - Output: {output_flat.shape}, Target: {target_flat.shape}")
        
        if return_components:
            return total_loss, {
                'total': total_loss.item(),
                'base': base_loss.item(),
                'reward': logical_reward.item(),
                'reward_scale': self.reward_scale
            }
        else:
            return total_loss
    
    def _calculate_logical_reward(self, output, target, src):
        """Calculate reward based on logical consistency with improved normalization
        
        Returns a normalized reward in roughly the same scale as the cross-entropy loss
        """
        reward = 0.0
        
        # Convert predictions to token indices
        pred_tokens = output.argmax(dim=-1)  # [batch_size, seq_len]
        
        # Get key logical tokens
        logical_tokens = {
            "if": self.vocab.get("if", -1),
            "then": self.vocab.get("then", -1),
            "implies": self.vocab.get("implies", -1),
            "and": self.vocab.get("and", -1),
            "or": self.vocab.get("or", -1),
            "not": self.vocab.get("not", -1)
        }
        
        batch_size = output.size(0)
        batch_rewards = []
        
        for b in range(batch_size):
            sequence_reward = 0.0
            
            # 1. Check if-then relationships (weight: 0.4)
            if logical_tokens["if"] != -1 and logical_tokens["then"] != -1:
                has_if = (pred_tokens[b] == logical_tokens["if"]).any()
                has_then = (pred_tokens[b] == logical_tokens["then"]).any()
                if has_if == has_then:  # Both present or both absent
                    sequence_reward += 0.4
            
            # 2. Check implication chains (weight: 0.3)
            if logical_tokens["implies"] != -1:
                pred_implies = (pred_tokens[b] == logical_tokens["implies"]).sum().float()
                target_implies = (target[b] == logical_tokens["implies"]).sum().float()
                implies_diff = -torch.abs(pred_implies - target_implies) / (target.size(1) + 1e-8)
                sequence_reward += 0.3 * (1 + implies_diff)  # Normalize to [0, 0.3]
            
            # 3. Check logical operators (and, or, not) usage (weight: 0.2)
            for op in ["and", "or", "not"]:
                if logical_tokens[op] != -1:
                    pred_op = (pred_tokens[b] == logical_tokens[op]).sum().float()
                    target_op = (target[b] == logical_tokens[op]).sum().float()
                    if pred_op > 0 and target_op > 0:  # Both use the operator
                        sequence_reward += 0.2 / 3  # Split 0.2 among three operators
            
            # 4. Length similarity (weight: 0.1)
            pred_len = (pred_tokens[b] != self.vocab["<PAD>"]).sum().float()
            target_len = (target[b] != self.vocab["<PAD>"]).sum().float()
            len_similarity = 1 - torch.abs(pred_len - target_len) / (target_len + 1e-8)
            sequence_reward += 0.1 * len_similarity
            
            batch_rewards.append(sequence_reward)
        
        # Average rewards across batch and normalize to similar scale as cross-entropy
        reward = torch.tensor(batch_rewards).mean().to(output.device)
        return reward

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        return self.d_model ** (-0.5) * min(self.current_step ** (-0.5), 
                                          self.current_step * self.warmup_steps ** (-1.5))

def get_curriculum_data(epoch):
    """Get training data based on curriculum learning"""
    # Start with simple patterns and gradually add complexity
    statements = []
    
    # Level 1: Direct implications (epochs 0-10)
    if epoch >= 0:
        statements.extend([
            ("If X is A, then X is B.", "Thing1 is A.", "Thing1 is B."),
            ("If X is even, it is divisible by 2.", "6 is even.", "6 is divisible by 2."),
            ("When something is cold, it has low temperature.", "Ice is cold.", "Ice has low temperature."),
        ])
    
    # Level 2: Number comparisons (epochs 10-20)
    if epoch >= 10:
        statements.extend([
            ("If a number is greater than 10, it is greater than 5.", "15 is greater than 10.", "15 is greater than 5."),
            ("If a number is greater than 20, it is greater than 15.", "25 is greater than 20.", "25 is greater than 15."),
        ])
    
    # Level 3: Physical state changes (epochs 20-30)
    if epoch >= 20:
        statements.extend([
            ("When metal is heated, it expands.", "The ring is heated.", "The ring expands."),
            ("When ice is heated, it melts.", "The ice is heated.", "The ice melts."),
        ])
    
    # Level 4: Multi-step reasoning (epochs 30-40)
    if epoch >= 30:
        statements.extend([
            ("If A implies B, and B implies C, then A implies C.", 
             "Rain implies wet ground, and wet ground implies mud.", 
             "Rain implies mud."),
            ("If X leads to Y, and Y leads to Z, then X leads to Z.",
             "Heat leads to expansion, and expansion leads to pressure.",
             "Heat leads to pressure."),
        ])
    
    # Level 5: Negation and exclusion (epochs 40+)
    if epoch >= 40:
        statements.extend([
            ("A number cannot be both positive and negative.", "5 is positive.", "5 is not negative."),
            ("When X is not Y, X cannot be Z.", "Thing2 is not Y.", "Thing2 cannot be Z."),
            ("No circles are polygons.", "This is a circle.", "This is not a polygon."),
        ])
    
    # Add variations of each pattern to help generalization
    variations = []
    for rule, fact, conclusion in statements:
        # Create variations by substituting numbers and objects
        if "number" in rule.lower():
            variations.extend([
                (rule, f"{n} is {pred}", f"{n} is {conc}")
                for n, pred, conc in [
                    ("4", "even", "divisible by 2"),
                    ("10", "greater than 8", "greater than 5"),
                    ("20", "greater than 15", "greater than 10")
                ]
            ])
        
        # Create variations for temperature relations
        if "temperature" in rule.lower():
            variations.extend([
                ("When something is cold, it has low temperature.", 
                 obj + " is cold.", 
                 obj + " has low temperature.")
                for obj in ["The water", "The metal", "The ice"]
            ])
        
        # Create variations for abstract reasoning
        if "X is A" in rule:
            variations.extend([
                (rule, obj + " is A.", obj + " is B.")
                for obj in ["Thing1", "Thing2", "Thing3"]
            ])
    
    statements.extend(variations)
    return statements

def train():
    global model  # Declare model as global
    start_time = time.time()
    num_epochs = EPOCHS
    VOCAB_SIZE = len(vocab)
    model = LogicalTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    criterion = LogicalConsistencyLoss(vocab).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0)  # Start with 0 lr, will be set by scheduler
    scheduler = WarmupScheduler(optimizer, D_MODEL, WARMUP_STEPS)
    
    # Initialize tracking metrics
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    min_epochs = 30  # Ensure we train for at least this many epochs
    
    # Track loss components and gradients
    loss_history = {
        'total_loss': [],
        'base_loss': [],
        'reward': [],
        'reward_scale': [],
        'grad_norm': [],
        'param_norm': []
    }
    
    # Log training start
    log_results("Starting new training session with curriculum learning")
    log_results("\nModel Architecture:")
    log_results(f"Vocab Size: {VOCAB_SIZE}")
    log_results(f"Model Dimension: {D_MODEL}")
    log_results(f"Number of Heads: {NHEAD}")
    log_results(f"Encoder Layers: {NUM_ENCODER_LAYERS}")
    log_results(f"Decoder Layers: {NUM_DECODER_LAYERS}")
    log_results(f"Feedforward Dimension: {DIM_FEEDFORWARD}")
    log_results(f"Dropout: {DROPOUT}")
    
    print("Starting training with logical consistency loss...")
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_tokens = 0
            epoch_loss_components = {
                'total': [], 'base': [], 'reward': [], 
                'grad_norm': [], 'param_norm': []
            }
            
            # Get curriculum data for this epoch
            current_statements = get_curriculum_data(epoch)
            train_dataset = LogicalDataset(current_statements)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            for batch_idx, (premise1, premise2, target) in enumerate(train_loader):
                scheduler.step()  # Update learning rate
                
                premise1 = premise1.to(device)
                premise2 = premise2.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                
                # Concatenate premises for input
                src = torch.cat([premise1, premise2], dim=1)
                
                # Teacher forcing: use target tokens shifted right
                tgt_input = target[:, :-1]
                tgt_output = target[:, 1:]
                
                # Forward pass
                output = model(src, tgt_input)
                
                # Calculate loss and components
                loss, loss_components = criterion(output, tgt_output, src, return_components=True)
                
                # Backward pass
                loss.backward()
                
                # Calculate gradient norms before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Calculate parameter norms
                param_norm = torch.norm(torch.stack([p.norm() for p in model.parameters()]))
                
                optimizer.step()
                
                # Update statistics
                total_loss += loss.item()
                total_tokens += (tgt_output != vocab["<PAD>"]).sum().item()
                
                # Track components
                epoch_loss_components['total'].append(loss_components['total'])
                epoch_loss_components['base'].append(loss_components['base'])
                epoch_loss_components['reward'].append(loss_components['reward'])
                epoch_loss_components['grad_norm'].append(grad_norm.item())
                epoch_loss_components['param_norm'].append(param_norm.item())
                
                # Log detailed statistics every 100 batches
                if batch_idx % 100 == 0:
                    print(f"\nBatch {batch_idx}, Epoch {epoch+1}")
                    print(f"Total Loss: {loss.item():.4f}")
                    print(f"Base Loss: {loss_components['base']:.4f}")
                    print(f"Reward: {loss_components['reward']:.4f}")
                    print(f"Gradient Norm: {grad_norm.item():.4f}")
                    print(f"Parameter Norm: {param_norm.item():.4f}")
                    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Calculate epoch averages
            avg_loss = total_loss / total_tokens if total_tokens > 0 else total_loss
            avg_components = {k: sum(v)/len(v) for k, v in epoch_loss_components.items()}
            
            # Update loss history
            loss_history['total_loss'].append(avg_components['total'])
            loss_history['base_loss'].append(avg_components['base'])
            loss_history['reward'].append(avg_components['reward'])
            loss_history['grad_norm'].append(avg_components['grad_norm'])
            loss_history['param_norm'].append(avg_components['param_norm'])
            loss_history['reward_scale'].append(criterion.reward_scale)
            
            # Log epoch statistics
            if (epoch + 1) % 10 == 0:
                epoch_results = []
                epoch_results.append(f"\nEpoch {epoch+1}/{num_epochs}")
                epoch_results.append(f"Loss Components:")
                epoch_results.append(f"  Total Loss: {avg_components['total']:.4f}")
                epoch_results.append(f"  Base Loss: {avg_components['base']:.4f}")
                epoch_results.append(f"  Reward: {avg_components['reward']:.4f}")
                epoch_results.append(f"  Gradient Norm: {avg_components['grad_norm']:.4f}")
                epoch_results.append(f"  Parameter Norm: {avg_components['param_norm']:.4f}")
                epoch_results.append(f"  Reward Scale: {criterion.reward_scale:.4f}")
                epoch_results.append(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
                epoch_results.append(f"Curriculum level: {min(epoch // 10, 5)}")
                log_results("\n".join(epoch_results))
                
                # Plot loss components
                if epoch >= 20:
                    plot_training_dynamics(loss_history, epoch)
                    evaluate()
                    evaluate_logical_reasoning()
            
            # Early stopping check (only after min_epochs)
            if epoch >= min_epochs:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
    except Exception as e:
        log_results(f"Error during training: {str(e)}")
        raise e
    finally:
        training_time = (time.time() - start_time) / 60  # Convert to minutes
        log_results(f"Total training time: {training_time:.2f} minutes")

def plot_training_dynamics(history, epoch):
    """Plot training dynamics to visualize loss components and gradient behavior"""
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss components
        epochs = range(len(history['total_loss']))
        ax1.plot(epochs, history['total_loss'], label='Total Loss')
        ax1.plot(epochs, history['base_loss'], label='Base Loss')
        ax1.plot(epochs, history['reward'], label='Reward')
        ax1.set_title('Loss Components')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot reward scaling
        ax2.plot(epochs, history['reward_scale'], label='Reward Scale')
        ax2.set_title('Reward Scale Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Scale')
        
        # Plot gradient norms
        ax3.plot(epochs, history['grad_norm'], label='Gradient Norm')
        ax3.set_title('Gradient Norm Evolution')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Norm')
        
        # Plot parameter norms
        ax4.plot(epochs, history['param_norm'], label='Parameter Norm')
        ax4.set_title('Parameter Norm Evolution')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Norm')
        
        plt.tight_layout()
        plt.savefig(f'training_dynamics_epoch_{epoch}.png')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for plotting")

def evaluate():
    """Evaluate the model on test cases"""
    model.eval()
    results = []
    
    test_cases = [
        # Basic numerical relationships
        ("If a number is even, it is divisible by 2.", "8 is even.", "8 is divisible by 2."),
        ("If a number is greater than 10, it is greater than 5.", "15 is greater than 10.", "15 is greater than 5."),
        
        # Physical properties
        ("When something is cold, it has low temperature.", "Ice is cold.", "Ice has low temperature."),
        ("When metal is heated, it expands.", "The ring is heated.", "The ring expands."),
        
        # Abstract reasoning
        ("If A implies B, and B implies C, then A implies C.",
         "Rain implies wet ground, and wet ground implies mud.",
         "Rain implies mud."),
    ]
    
    with torch.no_grad():
        for premise1, premise2, expected in test_cases:
            # Encode inputs
            src = torch.tensor([encode(premise1 + " " + premise2)], dtype=torch.long).to(device)
            
            # Initialize target with start token
            tgt = torch.tensor([[vocab["<START>"]]], dtype=torch.long).to(device)
            
            # Generate output token by token
            for _ in range(64):
                # Get model prediction
                output = model(src, tgt)
                next_token = output[:, -1].argmax(dim=-1)
                
                # Break if EOS token
                if next_token.item() == vocab["<EOS>"]:
                    break
                
                # Add predicted token to target
                tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            
            # Decode output
            output_text = decode(tgt[0].tolist())
            
            # Calculate accuracy
            correct = output_text.strip() == expected.lower().strip()
            
            # Log results
            results.append(f"\nTest Case:")
            results.append(f"Premise 1: {premise1}")
            results.append(f"Premise 2: {premise2}")
            results.append(f"Expected: {expected}")
            results.append(f"Got: {output_text}")
            results.append(f"Correct: {correct}")
    
    # Log evaluation results
    log_results("\n".join(results))
    print("\nEvaluation complete. Results written to reasoning_results.txt")

def evaluate_logical_reasoning():
    """Test the model's logical reasoning capabilities"""
    model.eval()
    correct = 0
    total = 0
    results = []
    
    # Get test cases from curriculum data
    test_statements = get_curriculum_data(50)  # Get all levels of complexity
    
    with torch.no_grad():
        for premise1, premise2, expected in test_statements:
            # Encode inputs
            src_tokens1 = encode(premise1)
            src_tokens2 = encode(premise2)
            src = torch.tensor([src_tokens1 + src_tokens2], device=device)
            
            # Initialize target with START token
            tgt = torch.tensor([[vocab["<START>"]]], device=device)
            
            # Track token statistics
            generated_tokens = []
            token_probs = []
            prev_token = None
            consecutive_unk = 0
            consecutive_same = 0
            max_length = 50  # Maximum sequence length
            
            # Generate output token by token
            for _ in range(max_length):
                # Get model predictions
                with torch.no_grad():
                    output = model(src, tgt)
                    probs = torch.softmax(output[:, -1], dim=-1)
                    
                    # Get top K tokens and their probabilities
                    top_k = 5
                    top_probs, top_tokens = torch.topk(probs, top_k)
                    
                    # Default to most probable token
                    next_token = top_tokens[0][0]
                    
                    # Handle UNK tokens
                    if next_token.item() == vocab["<UNK>"]:
                        consecutive_unk += 1
                        if consecutive_unk >= 2:  # If two UNKs in a row
                            # Try next most probable token that isn't UNK
                            for t in top_tokens[0][1:]:
                                if t.item() != vocab["<UNK>"]:
                                    next_token = t
                                    consecutive_unk = 0
                                    break
                    else:
                        consecutive_unk = 0
                    
                    # Handle repetitive tokens
                    if prev_token is not None and next_token.item() == prev_token:
                        consecutive_same += 1
                        if consecutive_same >= 3:  # If same token three times
                            # Try next most probable different token
                            for t in top_tokens[0][1:]:
                                if t.item() != prev_token:
                                    next_token = t
                                    consecutive_same = 0
                                    break
                    else:
                        consecutive_same = 0
                    
                    # Store token and its probability
                    generated_tokens.append(next_token.item())
                    token_probs.append(probs[0][next_token].item())
                    
                    # Early stopping conditions
                    if next_token.item() == vocab["<EOS>"]:
                        break
                    
                    # Force stop if:
                    if consecutive_unk >= 5 or consecutive_same >= 5 or len(generated_tokens) >= max_length:
                        break
                    
                    # Update target sequence and previous token
                    next_token = next_token.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
                    tgt = torch.cat([tgt, next_token], dim=1)
                    prev_token = next_token.item()
            
            # Decode output, removing any trailing UNKs
            while generated_tokens and generated_tokens[-1] == vocab["<UNK>"]:
                generated_tokens.pop()
                token_probs.pop()
            
            output_text = decode(generated_tokens)
            avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0
            
            # Check correctness
            total += 1
            if output_text.strip() == expected.lower().strip():
                correct += 1
            
            # Log results
            results.append("\nTest Case:")
            results.append(f"Premise 1: {premise1}")
            results.append(f"Premise 2: {premise2}")
            results.append(f"Expected: {expected}")
            results.append(f"Got: {output_text}")
            results.append(f"Average token probability: {avg_prob:.4f}")
            results.append(f"Correct: {output_text.strip() == expected.lower().strip()}")
    
    # Log overall results
    accuracy = (correct / total) * 100 if total > 0 else 0
    results.append(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    # Write results to file
    log_results("\n".join(results))
    
    return accuracy

def main():
    # Initialize vocabulary with all training data
    all_statements = simple_statements + medium_statements + complex_statements
    global vocab
    vocab = create_vocab(all_statements)
    
    # Initialize idx_to_word for decoding
    global id2word
    id2word = {v: k for k, v in vocab.items()}
    
    # Print vocabulary stats at startup
    print_vocab_stats()
    
    # Train Model
    train()
    evaluate()

if __name__ == "__main__":
    main()