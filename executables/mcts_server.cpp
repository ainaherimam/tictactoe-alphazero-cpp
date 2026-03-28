/**
 * AlphaZero MCTS HTTP Server
 *
 * Accepts POST /mcts  with JSON body:
 *   { "board": [16 ints], "player": "X"|"O" }
 *   board values: 0 = empty, 1 = X, 2 = O
 *
 * Returns JSON:
 *   { "policy": [16 floats], "move": <int>, "value": <float> }
 *
 * NN inference is handled by the shared-memory inference server
 * (run_inference_server.sh), exactly as in selfplay.
 *
 * Usage:
 *   ./AlphaZero_MCTS_Server [shm_name] [port] [iterations]
 *   Defaults: /mcts_jax_inference  5556  400
 */

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <array>
#include <iostream>
#include <sstream>
#include <string>

#include "core/game/board.h"
#include "core/game/cell_state.h"
#include "core/game/constants.h"
#include "core/mcts/mcts_agent_selfplay.h"
#include "core/mcts/mcts_config.h"
#include "inference/shared_memory/inference_queue_shm.h"

namespace beast = boost::beast;
namespace http  = beast::http;
namespace net   = boost::asio;
using     tcp   = net::ip::tcp;

// ─── JSON helpers ─────────────────────────────────────────────────────────────

static bool parse_request(const std::string& body,
                           std::array<int, BOARD_CELLS>& cells,
                           std::string& player)
{
    // Extract "board": [...]
    auto pos   = body.find("\"board\"");
    if (pos == std::string::npos) return false;
    auto start = body.find('[', pos);
    auto end   = body.find(']', start);
    if (start == std::string::npos || end == std::string::npos) return false;

    std::string arr = body.substr(start + 1, end - start - 1);
    std::istringstream iss(arr);
    std::string tok;
    int idx = 0;
    while (std::getline(iss, tok, ',') && idx < BOARD_CELLS) {
        tok.erase(0, tok.find_first_not_of(" \t\n\r"));
        if (auto last = tok.find_last_not_of(" \t\n\r"); last != std::string::npos)
            tok.erase(last + 1);
        cells[idx++] = std::stoi(tok);
    }
    if (idx != BOARD_CELLS) return false;

    // Extract "player": "X" or "O"
    pos = body.find("\"player\"");
    if (pos == std::string::npos) return false;
    auto colon = body.find(':', pos + 8);
    auto vq1   = body.find('"', colon);
    auto vq2   = body.find('"', vq1 + 1);
    if (vq1 == std::string::npos || vq2 == std::string::npos) return false;
    player = body.substr(vq1 + 1, vq2 - vq1 - 1);
    return !player.empty();
}

static std::string build_response(const std::vector<float>& policy,
                                   int move_idx, float value)
{
    std::ostringstream oss;
    oss << "{\"policy\":[";
    for (int i = 0; i < (int)policy.size(); ++i) {
        if (i > 0) oss << ",";
        oss << policy[i];
    }
    oss << "],\"move\":" << move_idx
        << ",\"value\":" << value << "}";
    return oss.str();
}

// ─── Request handler ──────────────────────────────────────────────────────────

static void handle(std::shared_ptr<SharedMemoryInferenceQueue> queue,
                   int iterations,
                   const http::request<http::string_body>& req,
                   http::response<http::string_body>& res)
{
    res.set(http::field::access_control_allow_origin,  "*");
    res.set(http::field::access_control_allow_methods, "POST, OPTIONS");
    res.set(http::field::access_control_allow_headers, "Content-Type");
    res.set(http::field::content_type, "application/json");

    if (req.method() == http::verb::options) {
        res.result(http::status::ok);
        res.body() = "";
        return;
    }

    if (req.method() != http::verb::post || req.target() != "/mcts") {
        res.result(http::status::not_found);
        res.body() = "{\"error\":\"Use POST /mcts\"}";
        return;
    }

    std::array<int, BOARD_CELLS> raw{};
    std::string player_str;
    if (!parse_request(req.body(), raw, player_str)) {
        res.result(http::status::bad_request);
        res.body() = "{\"error\":\"Invalid request body\"}";
        return;
    }

    // Reconstruct board from flat int array (0=empty, 1=X, 2=O)
    std::array<Cell_state, BOARD_CELLS> cells{};
    for (int i = 0; i < BOARD_CELLS; ++i) {
        if      (raw[i] == 1) cells[i] = Cell_state::X;
        else if (raw[i] == 2) cells[i] = Cell_state::O;
        else                  cells[i] = Cell_state::Empty;
    }
    Board board;
    board.load_board(cells);

    Cell_state player = (player_str == "X") ? Cell_state::X : Cell_state::O;

    // Mirror selfplay config from main.cpp — no dirichlet noise for human-vs-AI
    Mcts_config cfg(
        /*exploration_factor=*/ 1.4,
        /*number_iteration=*/   iterations,
        /*log_level=*/          LogLevel::NONE,
        /*temperature=*/        0.1f,
        /*dirichlet_alpha=*/    0.3f,
        /*dirichlet_epsilon=*/  0.0f,   // no noise: we want best play, not exploration
        /*queue=*/              queue,
        /*max_depth=*/          10,
        /*tree_reuse=*/         false,
        /*model_id=*/           1
    );

    Mcts_agent_selfplay agent(cfg);
    auto [move, policy] = agent.choose_move(board, player);

    int   move_idx = move.x * BOARD_WIDTH + move.y;
    float value    = agent.get_root_value();

    res.result(http::status::ok);
    res.body() = build_response(policy, move_idx, value);
}

// ─── Session ──────────────────────────────────────────────────────────────────

static void run_session(tcp::socket socket,
                         std::shared_ptr<SharedMemoryInferenceQueue> queue,
                         int iterations)
{
    beast::flat_buffer buf;
    http::request<http::string_body> req;
    http::read(socket, buf, req);

    http::response<http::string_body> res{http::status::ok, req.version()};
    res.set(http::field::server, "AlphaZero-MCTS");
    res.keep_alive(false);

    handle(queue, iterations, req, res);
    res.prepare_payload();
    http::write(socket, res);
}

// ─── Entry point ──────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    std::string    shm_name   = "/mcts_jax_inference";
    unsigned short port       = 5556;
    int            iterations = 100;

    if (argc > 1) shm_name   = argv[1];
    if (argc > 2) port       = static_cast<unsigned short>(std::stoi(argv[2]));
    if (argc > 3) iterations = std::stoi(argv[3]);

    std::cout << "AlphaZero MCTS Server\n";
    std::cout << "MCTS iterations per move (current default: " << iterations << "): ";
    std::string line;
    if (std::getline(std::cin, line) && !line.empty()) {
        try { iterations = std::stoi(line); }
        catch (...) { std::cout << "Invalid input, using default.\n"; }
    }

    std::cout << "  Inference SHM : " << shm_name << "\n"
              << "  HTTP port     : " << port << "\n"
              << "  Iterations    : " << iterations << "\n";

    auto queue = std::make_shared<SharedMemoryInferenceQueue>(shm_name);

    std::cout << "Waiting for inference server (" << shm_name << ")...\n";
    if (!queue->wait_for_server(30000)) {
        std::cerr << "Fatal: inference server not ready within 30s. "
                     "Is run_inference_server.sh running?\n";
        return 1;
    }
    std::cout << "Inference server ready.\n";

    net::io_context ioc;
    tcp::acceptor   acceptor(ioc, {tcp::v4(), port});
    std::cout << "Listening on port " << port << "...\n";

    for (;;) {
        tcp::socket socket(ioc);
        acceptor.accept(socket);
        try {
            run_session(std::move(socket), queue, iterations);
        } catch (const std::exception& e) {
            std::cerr << "Session error: " << e.what() << "\n";
        }
    }
}
