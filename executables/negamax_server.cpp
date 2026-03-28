/**
 * Negamax Perfect-Play HTTP Server
 *
 * Pre-solves the entire 4x4 Misere Tic-Tac-Toe game tree at startup,
 * then answers position queries instantly from the transposition table.
 *
 * POST /negamax
 *   Request:  { "board": [16 ints], "player": "X"|"O" }
 *             board values: 0=empty, 1=X, 2=O
 *   Response: { "action_values": [16 ints or null],
 *               "state_value": int,
 *               "best_move": int }
 *             action_values[i]: +1 win, 0 draw, -1 loss (current player),
 *                               null if cell is occupied
 *             state_value: game-theoretic value from current player's perspective
 *             best_move:   cell index of optimal move, -1 if terminal
 *
 * Usage: ./NegamaxServer [port]    (default port: 5557)
 */

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "core/solver/misere_solver.h"

namespace beast = boost::beast;
namespace http  = beast::http;
namespace net   = boost::asio;
using     tcp   = net::ip::tcp;

// ─── JSON helpers ─────────────────────────────────────────────────────────────

static bool parse_request(const std::string& body,
                           std::array<int, 16>& cells,
                           std::string& player)
{
    auto pos = body.find("\"board\"");
    if (pos == std::string::npos) return false;
    auto start = body.find('[', pos);
    auto end   = body.find(']', start);
    if (start == std::string::npos || end == std::string::npos) return false;

    std::string arr = body.substr(start + 1, end - start - 1);
    std::istringstream iss(arr);
    std::string tok;
    int idx = 0;
    while (std::getline(iss, tok, ',') && idx < 16) {
        tok.erase(0, tok.find_first_not_of(" \t\n\r"));
        if (auto last = tok.find_last_not_of(" \t\n\r"); last != std::string::npos)
            tok.erase(last + 1);
        cells[idx++] = std::stoi(tok);
    }
    if (idx != 16) return false;

    pos = body.find("\"player\"");
    if (pos == std::string::npos) return false;
    auto colon = body.find(':', pos + 8);
    auto vq1   = body.find('"', colon);
    auto vq2   = body.find('"', vq1 + 1);
    if (vq1 == std::string::npos || vq2 == std::string::npos) return false;
    player = body.substr(vq1 + 1, vq2 - vq1 - 1);
    return !player.empty();
}

// ─── Request handler ──────────────────────────────────────────────────────────

static void handle(MisereSolver& solver,
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

    if (req.method() != http::verb::post || req.target() != "/negamax") {
        res.result(http::status::not_found);
        res.body() = "{\"error\":\"Use POST /negamax\"}";
        return;
    }

    std::array<int, 16> raw{};
    std::string player_str;
    if (!parse_request(req.body(), raw, player_str)) {
        res.result(http::status::bad_request);
        res.body() = "{\"error\":\"Invalid request body\"}";
        return;
    }

    // Convert flat int array to bitmasks (0=empty, 1=X, 2=O)
    uint16_t bx = 0, bo = 0;
    for (int i = 0; i < 16; ++i) {
        if      (raw[i] == 1) bx |= static_cast<uint16_t>(1u << i);
        else if (raw[i] == 2) bo |= static_cast<uint16_t>(1u << i);
    }
    bool is_x_turn = (player_str == "X");

    // Terminal: someone already completed a line — no valid moves
    if (MisereSolver::has_line(bx) || MisereSolver::has_line(bo)) {
        res.result(http::status::ok);
        res.body() = "{\"action_values\":"
                     "[null,null,null,null,null,null,null,null,"
                     "null,null,null,null,null,null,null,null],"
                     "\"state_value\":0,\"best_move\":-1}";
        return;
    }

    int state_value             = solver.get_position_value(bx, bo, is_x_turn);
    auto action_values          = solver.get_action_values(bx, bo, is_x_turn);
    int  best_move              = solver.get_best_move(bx, bo, is_x_turn);

    // Build 16-element action_values array (null for occupied cells)
    std::array<std::string, 16> av;
    for (int i = 0; i < 16; ++i) av[i] = "null";
    for (const auto& entry : action_values)
        av[entry.cell] = std::to_string(entry.value);

    std::ostringstream oss;
    oss << "{\"action_values\":[";
    for (int i = 0; i < 16; ++i) {
        if (i > 0) oss << ",";
        oss << av[i];
    }
    oss << "],\"state_value\":" << state_value
        << ",\"best_move\":"    << best_move << "}";

    res.result(http::status::ok);
    res.body() = oss.str();
}

// ─── Session ──────────────────────────────────────────────────────────────────

static void run_session(tcp::socket socket, MisereSolver& solver)
{
    beast::flat_buffer buf;
    http::request<http::string_body> req;
    http::read(socket, buf, req);

    http::response<http::string_body> res{http::status::ok, req.version()};
    res.set(http::field::server, "NegamaxSolver");
    res.keep_alive(false);

    handle(solver, req, res);
    res.prepare_payload();
    http::write(socket, res);
}

// ─── Entry point ──────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    unsigned short port = 5557;
    if (argc > 1) port = static_cast<unsigned short>(std::stoi(argv[1]));

    std::cout << "Negamax Perfect-Play Server\n";
    std::cout << "Pre-solving 4x4 Misere Tic-Tac-Toe...\n";

    MisereSolver solver;
    auto result = solver.solve();

    std::cout << "Solved. Root value (X perspective): " << result.value << "\n";
    std::cout << "Transposition table entries: " << solver.table_size() << "\n";
    std::cout << "Listening on port " << port << "...\n";

    net::io_context ioc;
    tcp::acceptor   acceptor(ioc, {tcp::v4(), port});

    for (;;) {
        tcp::socket socket(ioc);
        acceptor.accept(socket);
        try {
            run_session(std::move(socket), solver);
        } catch (const std::exception& e) {
            std::cerr << "Session error: " << e.what() << "\n";
        }
    }
}
