"""
Pijul-Git Gateway Dataset Builder
Teaches HRM version control semantics through dual representations
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import subprocess
import json
import numpy as np
from pathlib import Path

@dataclass
class VersionControlExample:
    """Dual representation of the same change"""
    # Git representation
    git_commit: str
    git_diff: str
    git_tree: Dict
    git_history: List[str]
    
    # Pijul representation  
    pijul_patch: str
    pijul_dependencies: List[str]
    pijul_context: str
    pijul_theory: Dict  # Patch algebra representation
    
    # Common semantic meaning
    author_intent: str  # What the author was trying to do
    code_before: str
    code_after: str
    conflict_resolution: Optional[str]
    
    # Meta-learning signals
    is_bugfix: bool
    is_refactor: bool
    breaks_tests: bool
    merge_conflict: bool
    

class PijulGitCodec:
    """Translates between Git and Pijul representations"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.git_repo = self.repo_path / ".git"
        self.pijul_repo = self.repo_path / ".pijul"
        
    def extract_parallel_history(self) -> List[VersionControlExample]:
        """Extract the same repository history in both formats"""
        examples = []
        
        # Get Git history
        git_log = subprocess.run(
            ["git", "log", "--pretty=format:%H|%s|%an|%at", "--reverse"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout.split('\n')
        
        # Get Pijul patches
        pijul_log = subprocess.run(
            ["pijul", "log", "--output-format=json"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout
        
        # Build parallel examples
        for git_commit in git_log:
            if not git_commit:
                continue
                
            commit_hash, message, author, timestamp = git_commit.split('|')
            
            # Get Git details
            git_diff = self._get_git_diff(commit_hash)
            git_tree = self._get_git_tree(commit_hash)
            
            # Get corresponding Pijul patch
            pijul_patch = self._find_pijul_patch(timestamp, message)
            pijul_deps = self._get_pijul_dependencies(pijul_patch)
            
            # Extract semantic meaning
            intent = self._infer_author_intent(message, git_diff)
            
            examples.append(VersionControlExample(
                git_commit=commit_hash,
                git_diff=git_diff,
                git_tree=git_tree,
                git_history=self._get_git_parents(commit_hash),
                pijul_patch=pijul_patch,
                pijul_dependencies=pijul_deps,
                pijul_context=self._get_pijul_context(pijul_patch),
                pijul_theory=self._extract_patch_algebra(pijul_patch),
                author_intent=intent,
                code_before=self._get_code_before(commit_hash),
                code_after=self._get_code_after(commit_hash),
                conflict_resolution=self._get_conflict_resolution(commit_hash),
                is_bugfix='fix' in message.lower(),
                is_refactor='refactor' in message.lower(),
                breaks_tests=self._check_test_status(commit_hash),
                merge_conflict=self._has_merge_conflict(commit_hash)
            ))
            
        return examples
    
    def _infer_author_intent(self, message: str, diff: str) -> str:
        """Infer what the author was trying to accomplish"""
        # This is where HRM will learn to understand developer intent
        # from commit messages and code changes
        
        intents = []
        
        # Analyze commit message
        if 'fix' in message.lower():
            intents.append('bug_fix')
        if 'add' in message.lower() or '+' in diff[:100]:
            intents.append('feature_addition')
        if 'refactor' in message.lower():
            intents.append('code_improvement')
        if 'test' in message.lower():
            intents.append('test_addition')
        if 'merge' in message.lower():
            intents.append('integration')
            
        # Analyze diff patterns
        if 'TODO' in diff or 'FIXME' in diff:
            intents.append('technical_debt')
        if 'import' in diff or 'require' in diff:
            intents.append('dependency_change')
            
        return '|'.join(intents) if intents else 'unknown'
    
    def _extract_patch_algebra(self, patch_id: str) -> Dict:
        """Extract the mathematical patch theory representation"""
        # Pijul's patch algebra allows patches to commute
        # This teaches HRM about change independence
        
        return {
            'commutes_with': self._get_commutable_patches(patch_id),
            'conflicts_with': self._get_conflicting_patches(patch_id),
            'inverse': self._get_inverse_patch(patch_id),
            'composition': self._get_patch_composition(patch_id)
        }
    
    def tokenize_for_hrm(self, example: VersionControlExample) -> Dict[str, np.ndarray]:
        """Convert to HRM's byte-stream format"""
        
        # Encode Git representation as bytes
        git_bytes = []
        git_bytes.extend(example.git_commit.encode('utf-8'))
        git_bytes.append(0xFF)  # Separator
        git_bytes.extend(example.git_diff.encode('utf-8'))
        
        # Encode Pijul representation as bytes
        pijul_bytes = []
        pijul_bytes.extend(example.pijul_patch.encode('utf-8'))
        pijul_bytes.append(0xFF)  # Separator
        pijul_bytes.extend(example.pijul_context.encode('utf-8'))
        
        # Encode semantic understanding
        intent_bytes = example.author_intent.encode('utf-8')
        
        return {
            'git_stream': np.array(git_bytes, dtype=np.uint8),
            'pijul_stream': np.array(pijul_bytes, dtype=np.uint8),
            'intent_stream': np.array(list(intent_bytes), dtype=np.uint8),
            'is_equivalent': True  # Both represent the same change
        }
    
    def _get_git_diff(self, commit: str) -> str:
        return subprocess.run(
            ["git", "diff", f"{commit}^", commit],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout
    
    def _get_git_tree(self, commit: str) -> Dict:
        tree_output = subprocess.run(
            ["git", "ls-tree", "-r", commit],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout
        
        tree = {}
        for line in tree_output.split('\n'):
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    tree[parts[1]] = parts[0].split(' ')[2]  # filename: hash
        return tree
    
    def _find_pijul_patch(self, timestamp: str, message: str) -> str:
        """Find corresponding Pijul patch by timestamp/message"""
        # Implementation depends on Pijul CLI output format
        return f"patch_{timestamp}_{hash(message)}"
    
    def _get_pijul_dependencies(self, patch: str) -> List[str]:
        """Get patches this one depends on"""
        deps_output = subprocess.run(
            ["pijul", "dependencies", patch],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout
        return deps_output.strip().split('\n') if deps_output else []
    
    def _get_commutable_patches(self, patch: str) -> List[str]:
        """Find patches that can be reordered with this one"""
        # This is where Pijul shines - patch independence
        return []  # Implement based on Pijul theory
    
    def _get_code_before(self, commit: str) -> str:
        """Get file contents before change"""
        return subprocess.run(
            ["git", "show", f"{commit}^:README.md"],  # Example file
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout
    
    def _get_code_after(self, commit: str) -> str:
        """Get file contents after change"""
        return subprocess.run(
            ["git", "show", f"{commit}:README.md"],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        ).stdout
    
    # Additional helper methods...
    def _get_git_parents(self, commit: str) -> List[str]:
        return []
    
    def _get_pijul_context(self, patch: str) -> str:
        return ""
    
    def _get_conflict_resolution(self, commit: str) -> Optional[str]:
        return None
    
    def _check_test_status(self, commit: str) -> bool:
        return False
    
    def _has_merge_conflict(self, commit: str) -> bool:
        return False
    
    def _get_conflicting_patches(self, patch: str) -> List[str]:
        return []
    
    def _get_inverse_patch(self, patch: str) -> str:
        return ""
    
    def _get_patch_composition(self, patch: str) -> List[str]:
        return []


def build_dataset(repos: List[str], output_dir: str):
    """Build training dataset from multiple repositories"""
    
    all_examples = []
    
    for repo_path in repos:
        print(f"Processing {repo_path}...")
        
        # Clone/update both Git and Pijul versions
        codec = PijulGitCodec(repo_path)
        
        # Extract parallel history
        examples = codec.extract_parallel_history()
        
        # Tokenize for HRM
        for example in examples:
            tokenized = codec.tokenize_for_hrm(example)
            all_examples.append(tokenized)
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "git_streams.npy", 
            np.array([e['git_stream'] for e in all_examples]))
    np.save(output_path / "pijul_streams.npy",
            np.array([e['pijul_stream'] for e in all_examples]))
    np.save(output_path / "intent_streams.npy",
            np.array([e['intent_stream'] for e in all_examples]))
    
    print(f"Dataset saved to {output_dir}")
    print(f"Total examples: {len(all_examples)}")


if __name__ == "__main__":
    # Process popular open source repos
    repos = [
        "/path/to/linux",
        "/path/to/rust",
        "/path/to/pytorch",
        # Add more repos with rich history
    ]
    
    build_dataset(repos, "data/pijulgit-corpus")