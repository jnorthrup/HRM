"""
HRM Pijul-Git Model
Learns version control semantics and developer intent from dual representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Block,
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1Carry
)
from models.layers import CastedEmbedding, CastedLinear


@dataclass 
class PijulGitCarry:
    """Carries both Git and Pijul understanding"""
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    # Version control specific state
    git_understanding: torch.Tensor  # Git DAG representation
    pijul_understanding: torch.Tensor  # Patch theory representation
    semantic_understanding: torch.Tensor  # What the developer meant
    
    # Adaptive computation
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_PijulGitV1(nn.Module):
    """
    HRM that learns from parallel Git/Pijul representations
    to understand version control semantics and developer intent
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = 256  # Byte-level tokens
        
        # Dual input embeddings for Git and Pijul streams
        self.git_embedding = CastedEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size // 2,
            cast_to=torch.bfloat16
        )
        
        self.pijul_embedding = CastedEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size // 2,
            cast_to=torch.bfloat16
        )
        
        # Semantic understanding layers
        self.intent_classifier = nn.Linear(self.hidden_size, 32)  # Developer intents
        self.language_emergence = nn.Linear(self.hidden_size, self.vocab_size)  # Learn to generate text
        
        # Version control specific modules
        self.patch_algebra = PatchAlgebraModule(self.hidden_size)
        self.dag_processor = DAGProcessorModule(self.hidden_size)
        self.conflict_resolver = ConflictResolutionModule(self.hidden_size)
        
        # Hierarchical reasoning modules
        self.H_module = self._build_h_module(config)  # High-level: understand intent
        self.L_module = self._build_l_module(config)  # Low-level: process changes
        
        # Cross-representation attention
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=config.num_heads,
            batch_first=True
        )
        
        # Output heads
        self.to_git = nn.Linear(self.hidden_size, self.vocab_size)
        self.to_pijul = nn.Linear(self.hidden_size, self.vocab_size)
        self.to_natural_language = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Q-learning for adaptive computation
        self.q_halt = nn.Linear(self.hidden_size, 1)
        self.q_continue = nn.Linear(self.hidden_size, 1)
        
    def forward(
        self,
        git_stream: torch.Tensor,
        pijul_stream: torch.Tensor,
        target_format: str = "both"
    ) -> Dict[str, torch.Tensor]:
        """
        Process parallel representations and learn their equivalence
        
        Args:
            git_stream: Git representation as byte stream
            pijul_stream: Pijul representation as byte stream  
            target_format: Output format ("git", "pijul", "both", "intent")
        """
        
        batch_size, seq_len = git_stream.shape
        
        # Embed both representations
        git_emb = self.git_embedding(git_stream)
        pijul_emb = self.pijul_embedding(pijul_stream)
        
        # Combine representations
        combined = torch.cat([git_emb, pijul_emb], dim=-1)
        
        # Initialize carry
        h_state = torch.zeros(batch_size, seq_len, self.hidden_size)
        l_state = torch.zeros(batch_size, seq_len, self.hidden_size)
        
        outputs = []
        halting_probs = []
        
        # Adaptive computation loop
        for step in range(self.config.halt_max_steps):
            # High-level reasoning: What is the developer trying to do?
            h_state = self.H_module(h_state, combined)
            
            # Cross-representation attention: Learn equivalence
            attended, _ = self.cross_attention(h_state, combined, combined)
            
            # Low-level processing: How to represent this change?
            l_state = self.L_module(l_state, attended)
            
            # Process through specialized modules
            if target_format in ["git", "both"]:
                git_understanding = self.dag_processor(l_state)
                
            if target_format in ["pijul", "both"]:
                pijul_understanding = self.patch_algebra(l_state)
                
            # Check if we should halt
            q_halt = self.q_halt(l_state).squeeze(-1)
            q_continue = self.q_continue(l_state).squeeze(-1)
            
            halt_prob = torch.sigmoid(q_halt - q_continue)
            halting_probs.append(halt_prob)
            
            # Store output
            outputs.append(l_state)
            
            # Check halting condition
            if halt_prob.mean() > 0.5:
                break
        
        # Stack outputs
        final_output = torch.stack(outputs, dim=1).mean(dim=1)  # Average over steps
        
        # Generate outputs based on target format
        results = {}
        
        if target_format in ["git", "both"]:
            results["git_output"] = self.to_git(final_output)
            
        if target_format in ["pijul", "both"]:
            results["pijul_output"] = self.to_pijul(final_output)
            
        if target_format == "intent":
            results["intent"] = self.intent_classifier(final_output)
            results["natural_language"] = self.to_natural_language(final_output)
            
        # Learn to generate commit messages / understand language
        results["emergent_language"] = self.language_emergence(final_output)
        
        # Conflict resolution understanding
        results["conflict_resolution"] = self.conflict_resolver(final_output)
        
        results["halting_probs"] = torch.stack(halting_probs)
        
        return results
    
    def _build_h_module(self, config):
        """Build high-level reasoning module"""
        layers = []
        for _ in range(config.H_layers):
            layers.append(HierarchicalReasoningModel_ACTV1Block(config))
        return nn.ModuleList(layers)
    
    def _build_l_module(self, config):
        """Build low-level processing module"""
        layers = []
        for _ in range(config.L_layers):
            layers.append(HierarchicalReasoningModel_ACTV1Block(config))
        return nn.ModuleList(layers)


class PatchAlgebraModule(nn.Module):
    """Understands Pijul's patch theory and commutation"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.commutation_check = nn.Linear(hidden_size * 2, 1)
        self.dependency_extractor = nn.Linear(hidden_size, hidden_size)
        self.patch_composer = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x):
        # Learn patch algebra operations
        deps = self.dependency_extractor(x)
        return deps


class DAGProcessorModule(nn.Module):
    """Understands Git's DAG structure"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.parent_finder = nn.Linear(hidden_size, hidden_size)
        self.merge_processor = nn.Linear(hidden_size * 2, hidden_size)
        self.history_encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        # Process Git DAG structure
        parents = self.parent_finder(x)
        return parents


class ConflictResolutionModule(nn.Module):
    """Learns how developers resolve conflicts"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.conflict_detector = nn.Linear(hidden_size, 2)  # Binary: conflict or not
        self.resolution_strategy = nn.Linear(hidden_size, 4)  # Ours/theirs/manual/auto
        self.semantic_merger = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x):
        conflict = self.conflict_detector(x)
        strategy = self.resolution_strategy(x)
        return torch.cat([conflict, strategy], dim=-1)