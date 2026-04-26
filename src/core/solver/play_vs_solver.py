#!/usr/bin/env python3
"""
Interactive CLI: play 4x4 Misere Tic-Tac-Toe against the perfect solver.

Before each of your moves, the script prints the theoretical value of every
legal move (from your perspective after making it): +1 win, 0 draw, -1 loss.
The solver replies with an optimal move.

Run:
    python -m src.core.solver.play_vs_solver            # play as X (first)
    python -m src.core.solver.play_vs_solver --as o     # play as O (second)
"""

import argparse
from typing import List, Tuple

from misere_solver import MisereSolver, has_line


VALUE_LABEL = {1: "WIN", 0: "DRAW", -1: "LOSS"}


def board_str(bx: int, bo: int, highlight: int = -1) -> str:
    lines = ["    0   1   2   3 ", "  +---+---+---+---+"]
    for r in range(4):
        cells = []
        for c in range(4):
            idx = r * 4 + c
            bit = 1 << idx
            if bx & bit:
                ch = "X"
            elif bo & bit:
                ch = "O"
            else:
                ch = str(idx).rjust(2) if idx == highlight else " "
            cells.append(f" {ch if len(ch)==1 else ch} ".ljust(3)[:3])
        lines.append(f"{r} |" + "|".join(cells) + "|")
        lines.append("  +---+---+---+---+")
    return "\n".join(lines)


def print_action_values(avs: List[Tuple[int, int]]) -> None:
    if not avs:
        print("  (no legal moves)")
        return
    best = max(v for _, v in avs)
    print("  Move        Value")
    print("  ----------  -----")
    for cell, val in sorted(avs, key=lambda x: (-x[1], x[0])):
        marker = "*" if val == best else " "
        print(
            f"  {marker} cell {cell:2d}  (r{cell // 4} c{cell % 4})  "
            f"{val:+d}  [{VALUE_LABEL[val]}]"
        )
    print("  (* = optimal)")


def read_human_move(bx: int, bo: int) -> int:
    occupied = bx | bo
    while True:
        raw = input("Your move (cell 0-15, or 'row col'): ").strip().lower()
        if raw in ("q", "quit", "exit"):
            raise SystemExit("Exiting.")
        try:
            parts = raw.split()
            if len(parts) == 1:
                cell = int(parts[0])
            elif len(parts) == 2:
                cell = int(parts[0]) * 4 + int(parts[1])
            else:
                raise ValueError
            if not 0 <= cell <= 15:
                raise ValueError
        except ValueError:
            print("  Invalid input. Use 0-15 or 'row col' (e.g. '1 2'). 'q' to quit.")
            continue
        if occupied & (1 << cell):
            print("  Cell occupied. Try again.")
            continue
        return cell


def check_terminal(bx: int, bo: int) -> str:
    """Return a non-empty message if the position is terminal, else ''."""
    if has_line(bx):
        return "X formed a 3-in-a-row  ->  X LOSES  (O wins)"
    if has_line(bo):
        return "O formed a 3-in-a-row  ->  O LOSES  (X wins)"
    if (bx | bo) == 0xFFFF:
        return "Board full, no 3-in-a-row  ->  DRAW"
    return ""


def play(human_is_x: bool, pvp: bool = False) -> None:
    solver = MisereSolver()
    print("Solving 4x4 Misere Tic-Tac-Toe (one-time)...")
    root_val = solver.solve()
    print(
        f"Game-theoretic value for X (first player): "
        f"{root_val:+d} [{VALUE_LABEL[root_val]}]\n"
    )

    if pvp:
        print("Player vs Player mode. Solver stats shown for both sides.")
        print("X moves first. Type 'q' to quit.\n")
    else:
        human_side = "X" if human_is_x else "O"
        print(f"You play as {human_side}.  X moves first. Type 'q' to quit.\n")

    bx, bo = 0, 0
    is_x_turn = True  # X always starts

    while True:
        print(board_str(bx, bo))

        term = check_terminal(bx, bo)
        if term:
            print(f"\n>>> {term}")
            return

        side = "X" if is_x_turn else "O"
        pos_val = solver.get_position_value(bx, bo, is_x_turn)
        print(
            f"\n{side} to move.  "
            f"Position value for {side}: {pos_val:+d} [{VALUE_LABEL[pos_val]}]"
        )

        human_turn = pvp or (is_x_turn == human_is_x)
        if human_turn:
            avs = solver.get_action_values(bx, bo, is_x_turn)
            print("Your options:")
            print_action_values(avs)
            cell = read_human_move(bx, bo)
        else:
            cell = solver.get_best_move(bx, bo, is_x_turn)
            child_val = next(
                v for c, v in solver.get_action_values(bx, bo, is_x_turn) if c == cell
            )
            print(
                f"Solver plays cell {cell} (r{cell // 4} c{cell % 4})  "
                f"->  value for {side}: {child_val:+d} [{VALUE_LABEL[child_val]}]"
            )

        bit = 1 << cell
        if is_x_turn:
            bx |= bit
        else:
            bo |= bit
        is_x_turn = not is_x_turn
        print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Play 4x4 Misere Tic-Tac-Toe against the perfect solver."
    )
    p.add_argument(
        "--as",
        dest="side",
        choices=["x", "o"],
        default="x",
        help="Side to play vs solver (x=first, o=second). Default: x",
    )
    p.add_argument(
        "--mode",
        choices=["vs-solver", "pvp"],
        default="vs-solver",
        help="vs-solver: play against the solver (default); pvp: two human players with solver stats shown",
    )
    args = p.parse_args()
    try:
        play(human_is_x=(args.side == "x"), pvp=(args.mode == "pvp"))
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")


if __name__ == "__main__":
    main()
