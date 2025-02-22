import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
from datetime import datetime
import wandb
import bitsandbytes as bnb
from accelerate import Accelerator
import logging
import sys
import traceback
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# ---- SETUP LOGGING ---- #
def setup_logging(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ---- ENVIRONMENT VALIDATION ---- #
def validate_environment():
    """Validate all required components before starting"""
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for training.")
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        if gpu_memory < 16:
            raise RuntimeError(f"Insufficient GPU memory: {gpu_memory:.1f}GB (need minimum 16GB)")
        
        # Validate paths and permissions
        if not os.access("/wynton/scratch", os.W_OK):
            raise RuntimeError("No write access to /wynton/scratch")
            
        return True
    except Exception as e:
        logging.error(f"Environment validation failed: {str(e)}")
        raise

# ---- SETUP ---- #
try:
    # Basic setup
    MODEL_NAME = "tiiuae/falcon-7b"
    DATASET_NAME = "roneneldan/TinyStories"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR = Path(f"/wynton/scratch/$USER/falcon_training/{timestamp}")
    CHECKPOINT_DIR = LOGS_DIR / "checkpoints"
    BACKUP_DIR = LOGS_DIR / "backups"
    CSV_DIR = LOGS_DIR / "metrics"

    # Create directories
    for dir_path in [LOGS_DIR, CHECKPOINT_DIR, BACKUP_DIR, CSV_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize CSV files
    metrics_file = CSV_DIR / "training_metrics.csv"
    attention_file = CSV_DIR / "attention_metrics.csv"
    singular_values_file = CSV_DIR / "singular_values.csv"
    
    # Create CSV headers
    pd.DataFrame(columns=[
        'step', 'epoch', 'loss', 'learning_rate', 'attention_entropy',
        'mutual_information', 'gpu_memory'
    ]).to_csv(metrics_file, index=False)
    
    pd.DataFrame(columns=[
        'step', 'epoch', 'layer', 'attention_entropy', 'attention_sparsity',
        'attention_max_value'
    ]).to_csv(attention_file, index=False)
    
    pd.DataFrame(columns=[
        'step', 'epoch', 'singular_value_number', 'magnitude'
    ]).to_csv(singular_values_file, index=False)

    # Setup logging
    logger = setup_logging(LOGS_DIR)
    logger.info("Starting training setup...")

    # Validate environment
    validate_environment()
    
    # Initialize wandb with auto-retry
    for attempt in range(3):
        try:
            wandb.init(
                project="falcon-training",
                name=f"run_{timestamp}",
                dir=str(LOGS_DIR),
                resume="allow"
            )
            break
        except Exception as e:
            if attempt == 2:
                logger.error("Failed to initialize wandb after 3 attempts")
                raise
            logger.warning(f"Wandb initialization attempt {attempt + 1} failed, retrying...")

    # ---- LOAD MODEL & TOKENIZER ---- #
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # ---- LOAD DATASET ---- #
    logger.info("Loading and processing dataset...")
    
    # Setup HuggingFace cache directory
    CACHE_DIR = Path(f"/wynton/scratch/$USER/hf_cache/{timestamp}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(CACHE_DIR)
    
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=str(CACHE_DIR))
    
    def tokenize_function(examples):
        try:
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Tokenization failed: {str(e)}")
            raise

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    # ---- ENTROPY FUNCTION ---- #
    def compute_entropy(logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
        return entropy.mean().item()

    # ---- ANALYSIS FUNCTIONS ---- #
    def compute_mutual_information(joint_probs):
        """Compute mutual information from joint probability distribution"""
        marginal_x = np.sum(joint_probs, axis=1)
        marginal_y = np.sum(joint_probs, axis=0)
        mutual_info = 0
        for i in range(joint_probs.shape[0]):
            for j in range(joint_probs.shape[1]):
                if joint_probs[i,j] > 0:
                    mutual_info += joint_probs[i,j] * np.log2(
                        joint_probs[i,j] / (marginal_x[i] * marginal_y[j] + 1e-10)
                    )
        return mutual_info

    def compute_relative_mutual_information(I_XY, H_X):
        """Compute relative mutual information"""
        return I_XY / (H_X + 1e-10)

    def compute_attention_entropy(attention_weights):
        """Compute entropy of attention matrix"""
        flat_weights = attention_weights.flatten()
        return -np.sum(flat_weights * np.log2(flat_weights + 1e-10))

    def analyze_layer(layer_output, attention_weights):
        """Analyze a transformer layer's outputs and attention patterns"""
        # SVD analysis
        singular_values = torch.linalg.svd(layer_output, compute_uv=False)
        
        # Attention entropy
        attn_entropy = compute_attention_entropy(attention_weights.detach().cpu().numpy())
        
        # Get probabilities for MI calculation
        probs = F.softmax(layer_output, dim=-1).detach().cpu().numpy()
        
        return {
            'singular_values': singular_values.cpu().numpy(),
            'attention_entropy': attn_entropy,
            'output_probs': probs
        }

    # ---- CUSTOM TRAINER WITH ANALYSIS ---- #
    class FalconAnalysisTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.accelerator = Accelerator()
            self.best_loss = float('inf')
            self.step = 0
            
            # Analysis history
            self.attention_entropy_history = []
            self.mutual_info_history = []
            self.singular_values_history = []
            self.embedding_history = []
            
            # CSV files
            self.metrics_file = metrics_file
            self.attention_file = attention_file
            self.singular_values_file = singular_values_file
            
        def _save_to_csv(self, file_path, data_dict):
            """Safely append data to CSV file"""
            try:
                df = pd.DataFrame([data_dict])
                df.to_csv(file_path, mode='a', header=False, index=False)
            except Exception as e:
                logger.error(f"Error saving to CSV {file_path}: {str(e)}")
                
        def training_step(self, model, batch):
            try:
                with self.accelerator.autocast():
                    # Forward pass with attention outputs
                    outputs = model(**batch, output_attentions=True)
                    loss = outputs.loss
                    
                    # Get attention weights and hidden states
                    attention_weights = outputs.attentions  # All layer attentions
                    hidden_states = outputs.hidden_states   # All layer hidden states
                    
                    # Increment step counter
                    self.step += 1
                    
                    # Store basic metrics
                    metrics_dict = {
                        'step': self.step,
                        'epoch': self.state.epoch,
                        'loss': loss.item(),
                        'learning_rate': self.lr_scheduler.get_last_lr()[0],
                        'gpu_memory': torch.cuda.memory_allocated() / 1e9
                    }
                    
                    # Analyze each layer
                    for layer_idx, (layer_attn, layer_hidden) in enumerate(zip(attention_weights, hidden_states)):
                        # Layer analysis
                        analysis = analyze_layer(layer_hidden, layer_attn)
                        
                        # Store attention metrics
                        attention_dict = {
                            'step': self.step,
                            'epoch': self.state.epoch,
                            'layer': layer_idx,
                            'attention_entropy': analysis['attention_entropy'],
                            'attention_sparsity': (layer_attn < 0.01).float().mean().item(),
                            'attention_max_value': layer_attn.max().item()
                        }
                        self._save_to_csv(self.attention_file, attention_dict)
                        
                        # Store singular values
                        for sv_idx, sv_value in enumerate(analysis['singular_values']):
                            sv_dict = {
                                'step': self.step,
                                'epoch': self.state.epoch,
                                'singular_value_number': sv_idx + 1,
                                'magnitude': sv_value
                            }
                            self._save_to_csv(self.singular_values_file, sv_dict)
                    
                    # Compute mutual information if possible
                    if hasattr(outputs, 'logits'):
                        probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                        joint_probs = np.outer(
                            probs.mean(axis=0), 
                            analysis['output_probs'].mean(axis=0)
                        )
                        I_XY = compute_mutual_information(joint_probs)
                        H_X = -np.sum(probs.mean(axis=0) * np.log2(probs.mean(axis=0) + 1e-10))
                        metrics_dict['mutual_information'] = compute_relative_mutual_information(I_XY, H_X)
                    
                    # Save metrics to CSV
                    self._save_to_csv(self.metrics_file, metrics_dict)
                    
                    # Log to wandb
                    self.log(metrics_dict)
                    
                    return loss
                    
            except Exception as e:
                logger.error(f"Error in training step: {str(e)}")
                logger.error(traceback.format_exc())
                raise

        def _save_checkpoint(self, name):
            """Save checkpoint with analysis data and CSV paths"""
            checkpoint_path = CHECKPOINT_DIR / f"{name}.pt"
            backup_path = BACKUP_DIR / f"{name}.pt"
            
            # Save to temporary file first
            temp_path = checkpoint_path.with_suffix('.tmp')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.best_loss,
                'epoch': self.state.epoch,
                'step': self.step,
                'metrics_file': str(self.metrics_file),
                'attention_file': str(self.attention_file),
                'singular_values_file': str(self.singular_values_file)
            }, temp_path)
            
            # Atomic rename and backup
            temp_path.rename(checkpoint_path)
            shutil.copy2(checkpoint_path, backup_path)

        def _save_analysis_plots(self):
            """Save analysis visualization plots"""
            # Create plots directory
            plots_dir = LOGS_DIR / "analysis_plots"
            plots_dir.mkdir(exist_ok=True)
            
            # Plot attention entropy
            plt.figure(figsize=(10, 6))
            plt.plot(self.attention_entropy_history)
            plt.title("Attention Entropy Over Training")
            plt.xlabel("Step")
            plt.ylabel("Entropy")
            plt.grid(True)
            plt.savefig(plots_dir / "attention_entropy.png")
            plt.close()
            
            # Plot mutual information
            if self.mutual_info_history:
                plt.figure(figsize=(10, 6))
                plt.plot(self.mutual_info_history)
                plt.title("Relative Mutual Information Over Training")
                plt.xlabel("Step")
                plt.ylabel("RMI")
                plt.grid(True)
                plt.savefig(plots_dir / "mutual_information.png")
                plt.close()
            
            # Plot singular value evolution
            plt.figure(figsize=(10, 6))
            singular_values = np.array(self.singular_values_history)
            for i in range(min(5, singular_values.shape[1])):  # Plot top 5 singular values
                plt.plot(singular_values[:, i], label=f'SV {i+1}')
            plt.title("Top Singular Values Over Training")
            plt.xlabel("Step")
            plt.ylabel("Magnitude")
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / "singular_values.png")
            plt.close()
            
            logger.info(f"Analysis plots saved to {plots_dir}")

    # ---- TRAINING ARGS ---- #
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=100,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=4,
        report_to="wandb",
        max_grad_norm=1.0,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        # Add early stopping
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # ---- RUN TRAINING ---- #
    trainer = FalconAnalysisTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=None,  # DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting training with analysis...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer._save_checkpoint("final_model")
    
    # Save final analysis plots
    trainer._save_analysis_plots()
    
    logger.info("Training completed successfully!")

except Exception as e:
    logger.error("Fatal error in training script")
    logger.error(traceback.format_exc())
    # Ensure wandb syncs before exit
    if wandb.run is not None:
        wandb.finish(exit_code=1)
    sys.exit(1)

finally:
    # Clean up and sync
    if wandb.run is not None:
        wandb.finish()
    logger.info("Script finished executing")
