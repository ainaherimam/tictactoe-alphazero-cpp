#include "core/solver/misere_solver.h"
#include <algorithm>
#include <random>
#include <bit>
#include <climits>
#include <cassert>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MisereSolver::MisereSolver() {
    std::mt19937_64 rng(0xDEADBEEF42ULL);
    for (int i = 0; i < 16; ++i) {
        zobrist_x_[i] = rng();
        zobrist_o_[i] = rng();
    }
}

// ---------------------------------------------------------------------------
// Line detection
// ---------------------------------------------------------------------------

bool MisereSolver::has_line(uint16_t bits) {
    for (int i = 0; i < NUM_LINES; ++i) {
        if ((bits & LINE_MASKS[i]) == LINE_MASKS[i])
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Zobrist hashing
// ---------------------------------------------------------------------------

uint64_t MisereSolver::compute_hash(uint16_t bx, uint16_t bo) const {
    uint64_t h = 0;
    uint16_t tmp = bx;
    while (tmp) {
        int bit = std::countr_zero(static_cast<unsigned>(tmp));
        h ^= zobrist_x_[bit];
        tmp &= tmp - 1;
    }
    tmp = bo;
    while (tmp) {
        int bit = std::countr_zero(static_cast<unsigned>(tmp));
        h ^= zobrist_o_[bit];
        tmp &= tmp - 1;
    }
    return h;
}

// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

uint16_t MisereSolver::transform_board(uint16_t bits, int sym) {
    uint16_t result = 0;
    uint16_t tmp = bits;
    while (tmp) {
        int cell = std::countr_zero(static_cast<unsigned>(tmp));
        result |= (1u << SYM_PERM[sym][cell]);
        tmp &= tmp - 1;
    }
    return result;
}

void MisereSolver::canonicalize(uint16_t bx, uint16_t bo,
                                uint16_t& canon_bx, uint16_t& canon_bo,
                                int& best_sym) {
    canon_bx = bx;
    canon_bo = bo;
    best_sym = 0;

    for (int s = 1; s < 8; ++s) {
        uint16_t tx = transform_board(bx, s);
        uint16_t to = transform_board(bo, s);
        if (tx < canon_bx || (tx == canon_bx && to < canon_bo)) {
            canon_bx = tx;
            canon_bo = to;
            best_sym = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Negamax — flat ±1/0 scores, tight alpha-beta window for fast pruning.
// ---------------------------------------------------------------------------

int MisereSolver::negamax(uint16_t bx, uint16_t bo, bool is_x_turn,
                           int alpha, int beta, int depth) {
    // Terminal: last mover completed a line → they lose → current player wins.
    if (depth > 0) {
        uint16_t last_bits = is_x_turn ? bo : bx;
        if (has_line(last_bits))
            return +1;
    }

    uint16_t occupied = bx | bo;
    if (occupied == 0xFFFF)
        return 0;

    // Canonicalize + TT probe
    uint16_t cbx, cbo; int sym;
    canonicalize(bx, bo, cbx, cbo, sym);
    uint64_t hash = compute_hash(cbx, cbo);

    auto it = tt_.find(hash);
    if (it != tt_.end()) {
        const TTEntry& entry = it->second;
        if (entry.hash == hash) {
            if (entry.flag == EXACT)       return entry.value;
            if (entry.flag == LOWER_BOUND) alpha = std::max(alpha, (int)entry.value);
            if (entry.flag == UPPER_BOUND) beta  = std::min(beta,  (int)entry.value);
            if (alpha >= beta)             return entry.value;
        }
    }

    uint16_t empty = static_cast<uint16_t>(~occupied & 0xFFFF);
    int best      = -2;
    int orig_alpha = alpha;

    for (int i = 0; i < 16; ++i) {
        int cell = MOVE_ORDER[i];
        if (!(empty & (1u << cell))) continue;

        uint16_t new_bx = bx, new_bo = bo;
        if (is_x_turn) new_bx |= (1u << cell);
        else           new_bo |= (1u << cell);

        // Placing here completes our own line → instant loss. Score -1.
        // Do NOT skip — record it so all moves are considered.
        uint16_t our_bits = is_x_turn ? new_bx : new_bo;
        if (has_line(our_bits)) {
            best = std::max(best, -1);
            continue; // -1 can never raise alpha above 0, so no pruning needed
        }

        int score = -negamax(new_bx, new_bo, !is_x_turn, -beta, -alpha, depth + 1);
        if (score > best)  best  = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    if (best == -2) best = -1; // all moves were immediate losses

    TTEntry entry;
    entry.hash  = hash;
    entry.cbx   = cbx;
    entry.cbo   = cbo;
    entry.value = static_cast<int8_t>(best);
    entry.depth = static_cast<uint8_t>(depth);
    if (best <= orig_alpha)  entry.flag = UPPER_BOUND;
    else if (best >= beta)   entry.flag = LOWER_BOUND;
    else                     entry.flag = EXACT;
    tt_[hash] = entry;

    return best;
}

// ---------------------------------------------------------------------------
// depth_to_end — returns moves until game ends under minimax-optimal play:
//   • Winning side minimises depth (ends game ASAP).
//   • Losing  side maximises depth (delays as long as possible).
//
// Correctness fixes vs. original:
//   1. Uses get_position_value() for child values — no raw TT lookups that
//      silently skip alpha-beta-pruned positions (the original bug: pruned
//      children were "continue"d, making self-loss moves look longest).
//   2. When losing, skips immediate self-loss moves if any non-self-loss
//      optimal move exists — prevents selecting obvious self-losses.
//   3. Memoises results in depth_cache_ (canonical hash) for speed.
//   4. Correct INT_MIN sentinel (not -1) so first valid depth always wins.
// ---------------------------------------------------------------------------

int MisereSolver::depth_to_end(uint16_t bx, uint16_t bo, bool is_x_turn, int depth) {
    // Terminal: last mover completed a line.
    if (depth > 0) {
        uint16_t last_bits = is_x_turn ? bo : bx;
        if (has_line(last_bits)) return depth;
    }
    uint16_t occupied = bx | bo;
    if (occupied == 0xFFFF) return depth; // full board = draw

    // Cache lookup (canonical).
    uint16_t cbx, cbo; int sym;
    canonicalize(bx, bo, cbx, cbo, sym);
    uint64_t hash = compute_hash(cbx, cbo);
    {
        auto it = depth_cache_.find(hash);
        if (it != depth_cache_.end()) return it->second;
    }

    int my_val = get_position_value(bx, bo, is_x_turn);
    bool winning = (my_val > 0);

    uint16_t empty_bits = static_cast<uint16_t>(~occupied & 0xFFFF);

    // For a losing player: check whether any non-self-loss move achieves the
    // optimal value.  If so, we must never select an immediate self-loss move
    // (which would end the game one move sooner than necessary).
    bool has_non_self_loss_opt = false;
    if (!winning) {
        for (int i = 0; i < 16; ++i) {
            int cell = MOVE_ORDER[i];
            if (!(empty_bits & (1u << cell))) continue;
            uint16_t new_bx = bx, new_bo = bo;
            if (is_x_turn) new_bx |= (1u << cell);
            else           new_bo |= (1u << cell);
            uint16_t our_bits = is_x_turn ? new_bx : new_bo;
            if (has_line(our_bits)) continue; // self-loss — skip for now
            uint16_t full = new_bx | new_bo;
            int cv = (full == 0xFFFF) ? 0 : -get_position_value(new_bx, new_bo, !is_x_turn);
            if (cv == my_val) { has_non_self_loss_opt = true; break; }
        }
    }

    int best = winning ? INT_MAX : INT_MIN;

    for (int i = 0; i < 16; ++i) {
        int cell = MOVE_ORDER[i];
        if (!(empty_bits & (1u << cell))) continue;

        uint16_t new_bx = bx, new_bo = bo;
        if (is_x_turn) new_bx |= (1u << cell);
        else           new_bo |= (1u << cell);

        uint16_t our_bits = is_x_turn ? new_bx : new_bo;

        // Skip immediate self-loss when better alternatives exist.
        if (!winning && has_non_self_loss_opt && has_line(our_bits)) continue;

        uint16_t full = new_bx | new_bo;
        int child_val;
        if      (has_line(our_bits)) child_val = -1;
        else if (full == 0xFFFF)     child_val =  0;
        else child_val = -get_position_value(new_bx, new_bo, !is_x_turn);

        if (child_val != my_val) continue;

        int d;
        if (has_line(our_bits) || full == 0xFFFF)
            d = depth + 1;
        else
            d = depth_to_end(new_bx, new_bo, !is_x_turn, depth + 1);

        if (winning) best = std::min(best, d);
        else         best = std::max(best, d);
    }

    int result = (best == INT_MAX || best == INT_MIN) ? depth : best;
    depth_cache_[hash] = result;
    return result;
}

// ---------------------------------------------------------------------------
// find_best_move — picks the move with the correct game-theoretic value that
// leads to the longest game (when losing) or shortest game (when winning).
//
// Correctness fixes vs. original:
//   1. Uses get_position_value() for child values — never skips pruned moves.
//   2. When losing, skips immediate self-loss if any non-self-loss optimal
//      move exists (same logic as depth_to_end).
//   3. INT_MIN sentinel (not -1) for losing best_depth.
// ---------------------------------------------------------------------------

int MisereSolver::find_best_move(uint16_t bx, uint16_t bo, bool is_x_turn,
                                  int position_value, int depth) {
    uint16_t occupied = bx | bo;
    uint16_t empty    = static_cast<uint16_t>(~occupied & 0xFFFF);

    bool winning = (position_value > 0);

    // Pre-scan: does any non-self-loss move achieve the optimal value?
    bool has_non_self_loss_opt = false;
    if (!winning) {
        for (int i = 0; i < 16; ++i) {
            int cell = MOVE_ORDER[i];
            if (!(empty & (1u << cell))) continue;
            uint16_t new_bx = bx, new_bo = bo;
            if (is_x_turn) new_bx |= (1u << cell);
            else           new_bo |= (1u << cell);
            uint16_t our_bits = is_x_turn ? new_bx : new_bo;
            if (has_line(our_bits)) continue;
            uint16_t full = new_bx | new_bo;
            int cv = (full == 0xFFFF) ? 0 : -get_position_value(new_bx, new_bo, !is_x_turn);
            if (cv == position_value) { has_non_self_loss_opt = true; break; }
        }
    }

    int best_move  = -1;
    int best_depth = winning ? INT_MAX : INT_MIN;

    for (int i = 0; i < 16; ++i) {
        int cell = MOVE_ORDER[i];
        if (!(empty & (1u << cell))) continue;

        uint16_t new_bx = bx, new_bo = bo;
        if (is_x_turn) new_bx |= (1u << cell);
        else           new_bo |= (1u << cell);

        uint16_t our_bits = is_x_turn ? new_bx : new_bo;

        // Skip immediate self-loss when better alternatives exist.
        if (!winning && has_non_self_loss_opt && has_line(our_bits)) continue;

        uint16_t full = new_bx | new_bo;
        int child_val;
        if      (has_line(our_bits)) child_val = -1;
        else if (full == 0xFFFF)     child_val =  0;
        else child_val = -get_position_value(new_bx, new_bo, !is_x_turn);

        if (child_val != position_value) continue;

        int d;
        if (has_line(our_bits) || full == 0xFFFF)
            d = depth + 1;
        else
            d = depth_to_end(new_bx, new_bo, !is_x_turn, depth + 1);

        if (winning ? (d < best_depth) : (d > best_depth)) {
            best_depth = d;
            best_move  = cell;
        }
    }

    return best_move;
}

// ---------------------------------------------------------------------------
// Public solve interface
// ---------------------------------------------------------------------------

MisereSolver::Result MisereSolver::solve() {
    return solve(0, 0, true);
}

MisereSolver::Result MisereSolver::solve(uint16_t board_x, uint16_t board_o,
                                          bool is_x_turn) {
    tt_.clear();
    depth_cache_.clear();
    tt_.reserve(1 << 20);

    int depth = std::popcount(static_cast<unsigned>(board_x))
              + std::popcount(static_cast<unsigned>(board_o));

    // Phase 1: fast ±1 negamax with tight alpha-beta — fills TT
    int value = negamax(board_x, board_o, is_x_turn, -1, 1, depth);

    // Phase 2: delay-aware best move selection using populated TT
    int best_move = find_best_move(board_x, board_o, is_x_turn, value, depth);

    // Fallback
    if (best_move == -1) {
        uint16_t occ = board_x | board_o;
        for (int i = 0; i < 16; ++i) {
            int cell = MOVE_ORDER[i];
            if (!(occ & (1u << cell))) { best_move = cell; break; }
        }
    }

    return {value, best_move};
}

// ---------------------------------------------------------------------------
// Optimal moves
// ---------------------------------------------------------------------------

std::vector<int> MisereSolver::get_optimal_moves(uint16_t board_x, uint16_t board_o,
                                                  bool is_x_turn) const {
    uint16_t occupied = board_x | board_o;
    uint16_t empty    = static_cast<uint16_t>(~occupied & 0xFFFF);
    std::vector<int> moves;

    uint16_t cbx, cbo; int sym;
    canonicalize(board_x, board_o, cbx, cbo, sym);
    uint64_t h = compute_hash(cbx, cbo);
    auto it = tt_.find(h);
    if (it == tt_.end()) return moves;
    int position_value = it->second.value;

    for (int cell = 0; cell < 16; ++cell) {
        if (!(empty & (1u << cell))) continue;

        uint16_t new_bx = board_x, new_bo = board_o;
        if (is_x_turn) new_bx |= (1u << cell);
        else           new_bo |= (1u << cell);

        uint16_t our_bits = is_x_turn ? new_bx : new_bo;
        if (has_line(our_bits)) continue;

        uint16_t ccbx, ccbo; int csym;
        canonicalize(new_bx, new_bo, ccbx, ccbo, csym);
        auto cit = tt_.find(compute_hash(ccbx, ccbo));
        if (cit == tt_.end()) continue;

        if (-cit->second.value == position_value)
            moves.push_back(cell);
    }

    return moves;
}

// ---------------------------------------------------------------------------
// get_position_value — exact value using TT; fallback to targeted negamax.
// Does NOT clear the TT. Call solve() first for fast O(1) TT hits.
// ---------------------------------------------------------------------------

int MisereSolver::get_position_value(uint16_t bx, uint16_t bo, bool is_x_turn) {
    // Terminal: last mover completed a line → current player wins.
    if (bx | bo) { // depth > 0
        uint16_t last_bits = is_x_turn ? bo : bx;
        if (has_line(last_bits)) return +1;
    }
    if ((bx | bo) == 0xFFFF) return 0;

    uint16_t cbx, cbo; int sym;
    canonicalize(bx, bo, cbx, cbo, sym);
    uint64_t hash = compute_hash(cbx, cbo);
    auto it = tt_.find(hash);
    if (it != tt_.end() && it->second.flag == EXACT)
        return it->second.value;

    // Not in TT with exact value — targeted negamax (TT not cleared, acts as cache).
    int depth = std::popcount(static_cast<unsigned>(bx))
              + std::popcount(static_cast<unsigned>(bo));
    return negamax(bx, bo, is_x_turn, -1, 1, depth);
}

// ---------------------------------------------------------------------------
// get_action_values — exact value for every legal action, in original cell indices.
// Canonical form is used internally; results are reported for the raw board.
// ---------------------------------------------------------------------------

std::vector<MisereSolver::ActionValue> MisereSolver::get_action_values(
        uint16_t bx, uint16_t bo, bool is_x_turn) {
    std::vector<ActionValue> results;
    uint16_t occupied = bx | bo;

    for (int cell = 0; cell < 16; ++cell) {
        if (occupied & (1u << cell)) continue;

        uint16_t new_bx = bx, new_bo = bo;
        if (is_x_turn) new_bx |= (1u << cell);
        else           new_bo |= (1u << cell);

        uint16_t our_bits = is_x_turn ? new_bx : new_bo;

        // Completing our own line → immediate loss for us.
        if (has_line(our_bits)) {
            results.push_back({cell, -1});
            continue;
        }

        // Board full after move → draw.
        if ((new_bx | new_bo) == 0xFFFF) {
            results.push_back({cell, 0});
            continue;
        }

        // Child value from child's perspective; negate for our perspective.
        int child_val = get_position_value(new_bx, new_bo, !is_x_turn);
        results.push_back({cell, -child_val});
    }

    return results;
}

// ---------------------------------------------------------------------------
// get_best_move — delay-aware best move using populated TT (no TT clear).
// ---------------------------------------------------------------------------

int MisereSolver::get_best_move(uint16_t bx, uint16_t bo, bool is_x_turn) {
    int pos_val = get_position_value(bx, bo, is_x_turn);
    int depth = std::popcount(static_cast<unsigned>(bx))
              + std::popcount(static_cast<unsigned>(bo));
    int move = find_best_move(bx, bo, is_x_turn, pos_val, depth);
    if (move == -1) {
        uint16_t occ = bx | bo;
        for (int i = 0; i < 16; ++i) {
            int cell = MOVE_ORDER[i];
            if (!(occ & (1u << cell))) { move = cell; break; }
        }
    }
    return move;
}

// ---------------------------------------------------------------------------
// Draw position retrieval
// ---------------------------------------------------------------------------

std::vector<MisereSolver::DrawPosition> MisereSolver::get_draw_positions() const {
    std::vector<DrawPosition> draws;
    for (const auto& [key, entry] : tt_) {
        if (entry.flag == EXACT && entry.value == 0) {
            int d = std::popcount(static_cast<unsigned>(entry.cbx))
                  + std::popcount(static_cast<unsigned>(entry.cbo));
            draws.push_back({entry.cbx, entry.cbo, d});
        }
    }
    std::sort(draws.begin(), draws.end(), [](const DrawPosition& a, const DrawPosition& b) {
        return a.depth < b.depth;
    });
    return draws;
}
