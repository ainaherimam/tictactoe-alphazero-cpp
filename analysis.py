import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy.stats import entropy
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class GameDatasetAnalyzer:
    """Comprehensive analyzer for AlphaZero game datasets"""

    def __init__(self, dataset_path: str):
        """Load dataset from saved tensors"""
        print(f"Loading dataset from {dataset_path}...")

        # Load tensors - handling TorchScript modules from C++
        boards_data = torch.load(f"{dataset_path}_boards.pt", weights_only=False)
        pi_data = torch.load(f"{dataset_path}_pi.pt", weights_only=False)
        z_data = torch.load(f"{dataset_path}_z.pt", weights_only=False)
        mask_data = torch.load(f"{dataset_path}_mask.pt", weights_only=False)

        # Extract tensors from TorchScript state_dict
        def extract_from_state_dict(module):
            if isinstance(module, torch.Tensor):
                return [module[i] for i in range(module.shape[0])]
            elif isinstance(module, list):
                return module
            else:
                # TorchScript module - extract from state_dict
                state = module.state_dict()
                # Sort by numeric keys
                sorted_keys = sorted(state.keys(), key=lambda x: int(x))
                return [state[key] for key in sorted_keys]

        self.boards = extract_from_state_dict(boards_data)
        self.pi_targets = extract_from_state_dict(pi_data)
        self.z_targets = extract_from_state_dict(z_data)
        self.legal_mask = extract_from_state_dict(mask_data)

        self.num_positions = len(self.boards)
        print(f"Dataset loaded: {self.num_positions} positions")
        print(f"Board tensor shape: {self.boards[0].shape}")
        print(f"Pi tensor shape: {self.pi_targets[0].shape}")
        print(f"Z tensor shape: {self.z_targets[0].shape if hasattr(self.z_targets[0], 'shape') else 'scalar'}")
        print(f"Mask tensor shape: {self.legal_mask[0].shape}\n")

        self.games = self._split_into_games()

    def _is_empty_board(self, board_tensor: torch.Tensor) -> bool:
        """Check if a board state is empty (game start)"""
        # Check first 2 planes (current position for both players)
        return torch.sum(board_tensor[0:2]).item() == 0

    def _count_pieces(self, board_tensor: torch.Tensor) -> int:
        """Count total pieces on the board"""
        return torch.sum(board_tensor[0:2]).item()

    def _split_into_games(self) -> List[List[int]]:
        """Split dataset into individual games by detecting game resets"""
        games = []
        current_game = []
        prev_pieces = -1

        for idx in range(self.num_positions):
            current_pieces = self._count_pieces(self.boards[idx])

            # Game boundary detection:
            # 1. Empty board (game start)
            # 2. Piece count decreases (new game started)
            # 3. Piece count drops significantly (likely new game)
            is_game_start = (
                self._is_empty_board(self.boards[idx]) or
                (prev_pieces > 0 and current_pieces < prev_pieces) or
                (prev_pieces > 0 and current_pieces == 0)
            )

            if is_game_start and current_game:
                games.append(current_game)
                current_game = [idx]
            else:
                current_game.append(idx)

            prev_pieces = current_pieces

        if current_game:
            games.append(current_game)

        # Additional pass: split very long "games" that are likely multiple games
        refined_games = []
        for game in games:
            if len(game) > 20:  # 4x4 board max is 16 moves
                # Look for resets within this "game"
                subgames = []
                current_sub = [game[0]]

                for i in range(1, len(game)):
                    idx = game[i]
                    prev_idx = game[i-1]

                    current_pieces = self._count_pieces(self.boards[idx])
                    prev_pieces = self._count_pieces(self.boards[prev_idx])

                    # If pieces decrease, it's likely a new game
                    if current_pieces < prev_pieces - 1 or (prev_pieces > 2 and current_pieces <= 1):
                        subgames.append(current_sub)
                        current_sub = [idx]
                    else:
                        current_sub.append(idx)

                if current_sub:
                    subgames.append(current_sub)

                refined_games.extend(subgames)
            else:
                refined_games.append(game)

        return refined_games

    def visualize_board_state(self, board_tensor: torch.Tensor, ax=None):
        """Visualize a 4x4 board state"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        # Extract current position (planes 0 and 1)
        x_plane = board_tensor[0].numpy()
        o_plane = board_tensor[1].numpy()

        # Create board visualization
        board_display = np.zeros((4, 4))
        board_display[x_plane == 1] = 1  # X
        board_display[o_plane == 1] = -1  # O

        # Plot
        cmap = sns.color_palette("RdBu_r", as_cmap=True)
        sns.heatmap(board_display, annot=True, fmt='.0f', cmap=cmap,
                    center=0, vmin=-1, vmax=1, cbar=False, ax=ax,
                    linewidths=2, linecolor='black',
                    xticklabels=['A', 'B', 'C', 'D'],
                    yticklabels=['1', '2', '3', '4'])
        ax.set_title('Board State (1=X, -1=O, 0=Empty)')

    def visualize_policy(self, pi_tensor: torch.Tensor, ax=None):
        """Visualize policy distribution as heatmap"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        # Reshape policy to 4x4 board
        policy_board = pi_tensor.reshape(4, 4).numpy()

        sns.heatmap(policy_board, annot=True, fmt='.3f', cmap='YlOrRd',
                    ax=ax, linewidths=1, linecolor='gray',
                    xticklabels=['A', 'B', 'C', 'D'],
                    yticklabels=['1', '2', '3', '4'])
        ax.set_title('Policy Distribution')

    def show_game(self, game_idx: int, start_move: int = 0, num_moves: int = None):
        """Display all moves in a specific game"""
        if game_idx >= len(self.games):
            print(f"Game {game_idx} does not exist. Only {len(self.games)} games available.")
            return

        game_positions = self.games[game_idx]
        if num_moves is None:
            num_moves = len(game_positions)

        end_move = min(start_move + num_moves, len(game_positions))

        print(f"\n{'='*60}")
        print(f"GAME {game_idx} - Moves {start_move} to {end_move-1}")
        print(f"Total positions in game: {len(game_positions)}")
        print(f"{'='*60}\n")

        for i in range(start_move, end_move):
            idx = game_positions[i]

            print(f"\n--- Move {i} (Position {idx}) ---")

            # Create subplot for board, policy, and info
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Visualize board
            self.visualize_board_state(self.boards[idx], axes[0])

            # Visualize policy
            self.visualize_policy(self.pi_targets[idx], axes[1])

            # Print additional info
            z_value = self.z_targets[idx].item()
            current_player = "O" if self.boards[idx][10][0][0].item() > 0.5 else "X"
            legal_moves = torch.sum(self.legal_mask[idx]).item()
            policy_entropy = entropy(self.pi_targets[idx].numpy() + 1e-10)

            fig.suptitle(f'Move {i} - Player: {current_player} | Value: {z_value:.3f} | '
                        f'Legal: {int(legal_moves)} | Entropy: {policy_entropy:.3f}')
            plt.tight_layout()
            plt.show()

            print(f"Value Target (z): {z_value:.3f}")
            print(f"Current Player: {current_player}")
            print(f"Number of Legal Moves: {int(legal_moves)}")
            print(f"Policy Entropy: {policy_entropy:.3f}")

    def compute_statistics(self) -> Dict:
        """Compute comprehensive dataset statistics"""
        stats = {}

        # Basic statistics
        stats['total_positions'] = self.num_positions
        stats['total_games'] = len(self.games)
        stats['avg_game_length'] = np.mean([len(g) for g in self.games])
        stats['min_game_length'] = min([len(g) for g in self.games])
        stats['max_game_length'] = max([len(g) for g in self.games])

        # Value distribution
        z_values = torch.stack(self.z_targets).numpy()
        stats['z_mean'] = np.mean(z_values)
        stats['z_std'] = np.std(z_values)
        stats['z_wins'] = np.sum(z_values > 0.5)
        stats['z_losses'] = np.sum(z_values < -0.5)
        stats['z_draws'] = np.sum(np.abs(z_values) < 0.5)

        # Policy statistics
        policy_entropies = []
        for pi in self.pi_targets:
            pi_np = pi.numpy() + 1e-10
            policy_entropies.append(entropy(pi_np))

        stats['avg_policy_entropy'] = np.mean(policy_entropies)
        stats['min_policy_entropy'] = np.min(policy_entropies)
        stats['max_policy_entropy'] = np.max(policy_entropies)

        # Legal moves statistics
        legal_counts = [torch.sum(mask).item() for mask in self.legal_mask]
        stats['avg_legal_moves'] = np.mean(legal_counts)
        stats['min_legal_moves'] = np.min(legal_counts)
        stats['max_legal_moves'] = np.max(legal_counts)

        # Position uniqueness
        unique_positions = self._count_unique_positions()
        stats['unique_positions'] = unique_positions
        stats['uniqueness_ratio'] = unique_positions / self.num_positions
        stats['avg_position_frequency'] = self.num_positions / unique_positions

        # Move diversity (per position)
        move_diversity = self._compute_move_diversity()
        stats['avg_move_diversity'] = move_diversity

        return stats

    def _count_unique_positions(self) -> int:
        """Count unique board positions"""
        seen = set()
        for board in self.boards:
            # Hash based on current position planes
            board_hash = tuple(board[0:2].flatten().numpy())
            seen.add(board_hash)
        return len(seen)

    def _compute_move_diversity(self) -> float:
        """Compute average number of moves considered (non-zero policy)"""
        diversities = []
        for pi in self.pi_targets:
            non_zero = torch.sum(pi > 1e-6).item()
            diversities.append(non_zero)
        return np.mean(diversities)

    def find_duplicate_positions(self, position_idx: int, max_display: int = 10):
        """Find and display all duplicates of a given position"""
        if position_idx >= self.num_positions:
            print(f"Position {position_idx} does not exist. Max index: {self.num_positions - 1}")
            return

        # Get the target board state (first 2 planes = current position)
        target_board = self.boards[position_idx][0:2]

        # Find all positions with matching board state
        duplicates = []
        for idx in range(self.num_positions):
            if idx == position_idx:
                continue

            current_board = self.boards[idx][0:2]
            if torch.equal(target_board, current_board):
                duplicates.append(idx)

        print(f"\n{'='*70}")
        print(f"DUPLICATE SEARCH FOR POSITION {position_idx}")
        print(f"{'='*70}")
        print(f"Total duplicates found: {len(duplicates)}")

        if len(duplicates) == 0:
            print("No duplicates found for this position.")
            return

        # Display original position
        print(f"\n{'─'*70}")
        print(f"ORIGINAL POSITION {position_idx}")
        print(f"{'─'*70}")
        self._display_position_details(position_idx)

        # Display up to max_display duplicates
        num_to_show = min(max_display, len(duplicates))
        print(f"\n{'─'*70}")
        print(f"SHOWING {num_to_show} OF {len(duplicates)} DUPLICATES")
        print(f"{'─'*70}")

        for i, dup_idx in enumerate(duplicates[:num_to_show]):
            print(f"\n{'─'*70}")
            print(f"DUPLICATE {i+1}/{num_to_show} - Position {dup_idx}")
            print(f"{'─'*70}")
            self._display_position_details(dup_idx)

        # Compare policy targets
        if len(duplicates) > 0:
            self._compare_duplicate_policies(position_idx, duplicates[:num_to_show])

    def _display_position_details(self, idx: int):
        """Display detailed information about a position"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Visualize board
        self.visualize_board_state(self.boards[idx], axes[0])

        # Visualize policy
        self.visualize_policy(self.pi_targets[idx], axes[1])

        # Print info
        z_value = self.z_targets[idx].item() if hasattr(self.z_targets[idx], 'item') else self.z_targets[idx]
        current_player = "O" if self.boards[idx][10][0][0].item() > 0.5 else "X"
        legal_moves = torch.sum(self.legal_mask[idx]).item()
        policy_entropy = entropy(self.pi_targets[idx].numpy() + 1e-10)

        fig.suptitle(f'Position {idx} - Player: {current_player} | Value: {z_value:.3f} | '
                    f'Legal: {int(legal_moves)} | Entropy: {policy_entropy:.3f}', fontsize=12)
        plt.tight_layout()
        plt.show()

        print(f"  Value Target: {z_value:.3f}")
        print(f"  Current Player: {current_player}")
        print(f"  Legal Moves: {int(legal_moves)}")
        print(f"  Policy Entropy: {policy_entropy:.3f}")

        # Show top 3 policy moves
        pi_np = self.pi_targets[idx].numpy()
        top_indices = np.argsort(pi_np)[-3:][::-1]
        print(f"  Top 3 moves:")
        for rank, move_idx in enumerate(top_indices, 1):
            row = move_idx // 4
            col = move_idx % 4
            prob = pi_np[move_idx]
            print(f"    {rank}. ({row+1}{chr(65+col)}): {prob:.4f}")

    def _compare_duplicate_policies(self, original_idx: int, duplicate_indices: List[int]):
        """Compare policy targets across duplicates"""
        print(f"\n{'─'*70}")
        print("POLICY COMPARISON ACROSS DUPLICATES")
        print(f"{'─'*70}")

        original_pi = self.pi_targets[original_idx].numpy()

        # Calculate policy differences
        max_diff_list = []
        l2_diff_list = []
        kl_div_list = []

        for dup_idx in duplicate_indices:
            dup_pi = self.pi_targets[dup_idx].numpy()

            max_diff = np.max(np.abs(original_pi - dup_pi))
            l2_diff = np.sqrt(np.sum((original_pi - dup_pi)**2))

            # KL divergence (with smoothing)
            orig_smooth = original_pi + 1e-10
            dup_smooth = dup_pi + 1e-10
            orig_smooth /= orig_smooth.sum()
            dup_smooth /= dup_smooth.sum()
            kl = np.sum(orig_smooth * np.log(orig_smooth / dup_smooth))

            max_diff_list.append(max_diff)
            l2_diff_list.append(l2_diff)
            kl_div_list.append(kl)

        print(f"Policy Difference Statistics:")
        print(f"  Max Absolute Diff - Mean: {np.mean(max_diff_list):.4f}, Max: {np.max(max_diff_list):.4f}")
        print(f"  L2 Distance - Mean: {np.mean(l2_diff_list):.4f}, Max: {np.max(l2_diff_list):.4f}")
        print(f"  KL Divergence - Mean: {np.mean(kl_div_list):.4f}, Max: {np.max(kl_div_list):.4f}")

        if np.mean(max_diff_list) > 0.1:
            print(f"\n  ⚠️  WARNING: Significant policy disagreement detected!")
            print(f"     Same positions have different policy targets.")
            print(f"     This may indicate issues with MCTS consistency.")
        elif np.mean(max_diff_list) > 0.01:
            print(f"\n  💡 Moderate policy variation across duplicates (normal for MCTS exploration)")
        else:
            print(f"\n  ✅ Policies are very consistent across duplicates")

    def print_statistics(self):
        """Print formatted statistics report"""
        stats = self.compute_statistics()

        print("\n" + "="*70)
        print("ALPHAZERO DATASET STATISTICS")
        print("="*70)

        print("\n📊 DATASET OVERVIEW")
        print(f"  Total Positions: {stats['total_positions']:,}")
        print(f"  Total Games: {stats['total_games']:,}")
        print(f"  Unique Positions: {stats['unique_positions']:,}")
        print(f"  Uniqueness Ratio: {stats['uniqueness_ratio']:.2%}")
        print(f"  Avg Position Frequency: {stats['avg_position_frequency']:.2f}x")

        print("\n🎮 GAME LENGTH")
        print(f"  Average: {stats['avg_game_length']:.1f} moves")
        print(f"  Min: {stats['min_game_length']} moves")
        print(f"  Max: {stats['max_game_length']} moves")

        print("\n🎯 VALUE TARGETS (Z)")
        print(f"  Mean: {stats['z_mean']:.3f}")
        print(f"  Std Dev: {stats['z_std']:.3f}")
        print(f"  Wins (+1): {stats['z_wins']:,} ({stats['z_wins']/stats['total_positions']:.1%})")
        print(f"  Losses (-1): {stats['z_losses']:,} ({stats['z_losses']/stats['total_positions']:.1%})")
        print(f"  Draws (0): {stats['z_draws']:,} ({stats['z_draws']/stats['total_positions']:.1%})")

        print("\n📈 POLICY DIVERSITY")
        print(f"  Avg Policy Entropy: {stats['avg_policy_entropy']:.3f}")
        print(f"  Min Policy Entropy: {stats['min_policy_entropy']:.3f}")
        print(f"  Max Policy Entropy: {stats['max_policy_entropy']:.3f}")
        print(f"  Avg Moves Considered: {stats['avg_move_diversity']:.1f}")

        print("\n⚖️  LEGAL MOVES")
        print(f"  Average: {stats['avg_legal_moves']:.1f}")
        print(f"  Min: {int(stats['min_legal_moves'])}")
        print(f"  Max: {int(stats['max_legal_moves'])}")

        print("\n✅ TRAINING READINESS ASSESSMENT")
        self._assess_training_readiness(stats)

        print("="*70 + "\n")

    def _assess_training_readiness(self, stats: Dict):
        """Assess if dataset is ready for AlphaZero training"""
        issues = []
        warnings_list = []

        # Check minimum dataset size
        if stats['total_positions'] < 10000:
            issues.append(f"⚠️  Small dataset ({stats['total_positions']} positions). Recommend 10k+")
        elif stats['total_positions'] < 50000:
            warnings_list.append(f"💡 Moderate dataset size. 50k+ recommended for stable training")

        # Check uniqueness
        if stats['uniqueness_ratio'] < 0.5:
            warnings_list.append(f"💡 Low uniqueness ({stats['uniqueness_ratio']:.1%}). Consider more diverse self-play")

        # Check policy entropy
        if stats['avg_policy_entropy'] < 0.5:
            issues.append("⚠️  Very low policy entropy. Policies may be too deterministic")
        elif stats['avg_policy_entropy'] < 1.0:
            warnings_list.append("💡 Low policy entropy. Check MCTS exploration")

        # Check value balance
        win_rate = stats['z_wins'] / stats['total_positions']
        if win_rate < 0.3 or win_rate > 0.7:
            warnings_list.append(f"💡 Imbalanced outcomes ({win_rate:.1%} wins). May indicate evaluation issues")

        # Check game completion
        if stats['avg_game_length'] < 3:
            issues.append("⚠️  Very short games. Check game logic")

        # Print assessment
        if not issues and not warnings_list:
            print("  ✅ Dataset appears ready for training!")
        else:
            if issues:
                print("  ISSUES:")
                for issue in issues:
                    print(f"    {issue}")
            if warnings_list:
                print("  WARNINGS:")
                for warning in warnings_list:
                    print(f"    {warning}")

    def plot_distributions(self):
        """Plot key distribution visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Game lengths
        game_lengths = [len(g) for g in self.games]
        axes[0, 0].hist(game_lengths, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Game Length Distribution')
        axes[0, 0].set_xlabel('Number of Moves')
        axes[0, 0].set_ylabel('Frequency')

        # Value distribution
        z_values = torch.stack(self.z_targets).numpy()
        axes[0, 1].hist(z_values, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Value Target Distribution')
        axes[0, 1].set_xlabel('Value (z)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.5)

        # Policy entropy
        entropies = [entropy(pi.numpy() + 1e-10) for pi in self.pi_targets]
        axes[0, 2].hist(entropies, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_title('Policy Entropy Distribution')
        axes[0, 2].set_xlabel('Entropy')
        axes[0, 2].set_ylabel('Frequency')

        # Legal moves
        legal_counts = [torch.sum(mask).item() for mask in self.legal_mask]
        axes[1, 0].hist(legal_counts, bins=16, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Legal Moves Distribution')
        axes[1, 0].set_xlabel('Number of Legal Moves')
        axes[1, 0].set_ylabel('Frequency')

        # Move diversity
        move_diversity = [torch.sum(pi > 1e-6).item() for pi in self.pi_targets]
        axes[1, 1].hist(move_diversity, bins=16, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Move Diversity (Non-zero Policies)')
        axes[1, 1].set_xlabel('Number of Moves Considered')
        axes[1, 1].set_ylabel('Frequency')

        # Value over game progress
        avg_z_by_move = defaultdict(list)
        for game in self.games:
            for i, idx in enumerate(game):
                avg_z_by_move[i].append(self.z_targets[idx].item())

        move_numbers = sorted(avg_z_by_move.keys())
        avg_values = [np.mean(avg_z_by_move[m]) for m in move_numbers]
        axes[1, 2].plot(move_numbers, avg_values, marker='o', markersize=3)
        axes[1, 2].set_title('Average Value by Move Number')
        axes[1, 2].set_xlabel('Move Number')
        axes[1, 2].set_ylabel('Average Value')
        axes[1, 2].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = GameDatasetAnalyzer("dataset")

    # Print comprehensive statistics
    analyzer.print_statistics()

    # Plot distributions
    analyzer.plot_distributions()

    # View specific games
    print(f"\nTotal games in dataset: {len(analyzer.games)}")

    # Show first game
    analyzer.show_game(0)

    # Show specific moves from a game
    # analyzer.show_game(game_idx=5, start_move=0, num_moves=5)