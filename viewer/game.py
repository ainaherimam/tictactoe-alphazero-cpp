#!/usr/bin/env python3
"""
All-in-one server for MISÈRE 4×4 Neural Tic-Tac-Toe.

  • Serves the game UI at  http://localhost:5555/
  • Proxies Triton Inference Server (CORS) at  http://localhost:5555/v2/...
  • Run:  python server.py
  • Then open:  http://localhost:5555
"""

import json
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

TRITON_URL = "http://localhost:8000"
PROXY_PORT = 5555

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED HTML  (game.html theme + viewer tab injected)
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MISÈRE 4×4 — Neural Tic-Tac-Toe</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3f;
    --accent-x: #ff4560;
    --accent-o: #00c9a7;
    --accent-warn: #ffd166;
    --text: #e8e8f0;
    --text-dim: #6b6b88;
    --glow-x: rgba(255,69,96,0.25);
    --glow-o: rgba(0,201,167,0.25);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(42,42,63,0.3) 1px, transparent 1px),
      linear-gradient(90deg, rgba(42,42,63,0.3) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 960px;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
  }
  header h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 3px;
    background: linear-gradient(135deg, var(--accent-x), var(--accent-o));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  header .subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    line-height: 1.6;
  }

  /* ── Top-level page tabs ── */
  .page-tabs {
    display: flex;
    gap: 6px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
  }
  .page-tab {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 10px 20px;
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    background: var(--surface2);
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.15s;
    position: relative;
    bottom: -1px;
  }
  .page-tab:hover { color: var(--text); background: var(--surface); }
  .page-tab.active {
    background: var(--surface);
    color: var(--accent-o);
    border-color: var(--border);
    border-bottom-color: var(--surface);  /* hide bottom border */
  }

  .page-content { display: none; }
  .page-content.active { display: block; }

  /* ── Game layout ── */
  .main-layout {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 20px;
  }
  @media (max-width: 700px) {
    .main-layout { grid-template-columns: 1fr; }
    header h1 { font-size: 2rem; }
  }

  /* Turn indicator */
  .turn-bar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .turn-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 2px;
  }
  .turn-player {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 2px;
    transition: color 0.3s;
  }
  .turn-player.X { color: var(--accent-x); text-shadow: 0 0 20px var(--glow-x); }
  .turn-player.O { color: var(--accent-o); text-shadow: 0 0 20px var(--glow-o); }
  .rule-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-warn);
    background: rgba(255,209,102,0.1);
    border: 1px solid rgba(255,209,102,0.3);
    padding: 4px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Board */
  .board-wrap { display: flex; flex-direction: column; gap: 14px; }
  .board {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    aspect-ratio: 1;
    width: 100%;
    max-width: 420px;
  }
  .cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.1s, box-shadow 0.2s;
    aspect-ratio: 1;
    flex-direction: column;
  }
  .cell:hover:not(.taken):not(.game-over) { border-color: var(--text-dim); transform: scale(1.03); }
  .cell.taken { cursor: default; }
  .cell .piece {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    line-height: 1;
    z-index: 2;
    transition: all 0.15s;
  }
  .cell .piece.X { color: var(--accent-x); text-shadow: 0 0 12px var(--glow-x); }
  .cell .piece.O { color: var(--accent-o); text-shadow: 0 0 12px var(--glow-o); }
  .cell .ai-bg {
    position: absolute; inset: 0; border-radius: 9px;
    opacity: 0; transition: opacity 0.4s; z-index: 0;
  }
  .cell .ai-prob {
    position: absolute; bottom: 4px; right: 5px;
    font-family: 'Space Mono', monospace; font-size: 0.58rem;
    color: rgba(255,255,255,0.7); z-index: 3; opacity: 0;
    transition: opacity 0.4s; pointer-events: none;
  }
  .cell.ai-shown .ai-bg { opacity: 1; }
  .cell.ai-shown .ai-prob { opacity: 1; }
  .cell.losing { animation: lose-pulse 0.6s ease-in-out 2; }
  @keyframes lose-pulse {
    0%,100% { box-shadow: none; }
    50% { box-shadow: 0 0 20px var(--accent-x), inset 0 0 10px rgba(255,69,96,0.2); }
  }

  /* Game over */
  .game-result { display: none; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; text-align: center; }
  .game-result.visible { display: block; }
  .game-result .winner-text { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; letter-spacing: 3px; }
  .game-result .winner-text.X { color: var(--accent-o); }
  .game-result .winner-text.O { color: var(--accent-x); }
  .game-result .winner-text.draw { color: var(--accent-warn); }
  .game-result .sub { font-size: 0.8rem; color: var(--text-dim); margin-top: 4px; font-family: 'Space Mono', monospace; }

  /* Controls */
  .controls { display: flex; gap: 10px; flex-wrap: wrap; }
  .btn {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 10px 18px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn:hover { border-color: var(--text-dim); background: var(--surface); }
  .btn-ai { border-color: var(--accent-o); color: var(--accent-o); background: rgba(0,201,167,0.08); flex: 1; }
  .btn-ai:hover { background: rgba(0,201,167,0.15); box-shadow: 0 0 12px var(--glow-o); }
  .btn-ai:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-ai.loading { animation: pulse-btn 0.8s ease-in-out infinite; }
  @keyframes pulse-btn { 0%,100% { opacity: 0.6; } 50% { opacity: 1; } }
  .btn-reset { color: var(--accent-warn); border-color: rgba(255,209,102,0.4); }
  .btn-reset:hover { background: rgba(255,209,102,0.08); }

  /* Sidebar */
  .sidebar { display: flex; flex-direction: column; gap: 16px; }
  .value-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: none; }
  .value-panel.visible { display: block; }
  .value-label { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
  .value-display { font-family: 'Bebas Neue', sans-serif; font-size: 3.5rem; line-height: 1; text-align: center; letter-spacing: 2px; transition: color 0.4s; }
  .value-good { color: var(--accent-o); text-shadow: 0 0 30px var(--glow-o); }
  .value-bad  { color: var(--accent-x); text-shadow: 0 0 30px var(--glow-x); }
  .value-neutral { color: var(--accent-warn); }
  .value-interp { font-size: 0.75rem; color: var(--text-dim); text-align: center; margin-top: 6px; font-style: italic; }
  .value-bar-wrap { margin-top: 12px; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
  .value-bar { height: 100%; border-radius: 3px; transition: width 0.5s ease, background 0.5s; }

  /* Config panel */
  .config-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .config-title { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; }
  .config-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .config-row label { font-size: 0.72rem; color: var(--text-dim); }
  .config-input { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-family: 'Space Mono', monospace; font-size: 0.7rem; padding: 5px 10px; width: 150px; outline: none; }
  .config-input:focus { border-color: var(--accent-o); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-dim); display: inline-block; margin-right: 6px; transition: background 0.3s; }
  .status-dot.ok      { background: var(--accent-o); box-shadow: 0 0 6px var(--glow-o); }
  .status-dot.err     { background: var(--accent-x); }
  .status-dot.loading { background: var(--accent-warn); animation: pulse-btn 0.8s infinite; }
  .status-text { font-size: 0.7rem; color: var(--text-dim); }

  /* History */
  .history-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; flex: 1; }
  .history-title { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px; }
  .history-list { display: flex; flex-wrap: wrap; gap: 5px; max-height: 200px; overflow-y: auto; }
  .history-item { font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 3px 8px; border-radius: 5px; border: 1px solid var(--border); }
  .history-item.X { color: var(--accent-x); border-color: rgba(255,69,96,0.3); background: rgba(255,69,96,0.07); }
  .history-item.O { color: var(--accent-o); border-color: rgba(0,201,167,0.3); background: rgba(0,201,167,0.07); }

  /* ══════════════════════════════════════════════════
     VIEWER TAB — restyled into game.html dark theme
     ══════════════════════════════════════════════════ */
  .viewer-layout {
    display: grid;
    grid-template-columns: 1fr 260px;
    gap: 20px;
  }
  @media (max-width: 700px) { .viewer-layout { grid-template-columns: 1fr; } }

  .viewer-left { display: flex; flex-direction: column; gap: 16px; }

  /* file loader */
  .load-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .load-panel .config-title { margin-bottom: 10px; }
  .file-btn-label {
    display: block; padding: 10px 18px;
    background: rgba(0,201,167,0.08);
    border: 1px solid var(--accent-o);
    color: var(--accent-o);
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 1.5px;
    border-radius: 8px; cursor: pointer; text-align: center;
    transition: all 0.15s;
  }
  .file-btn-label:hover { background: rgba(0,201,167,0.15); box-shadow: 0 0 12px var(--glow-o); }
  #folderInput { display: none; }

  /* monitor controls */
  .monitor-bar {
    display: none;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-top: 10px;
    padding: 10px 14px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-family: 'Space Mono', monospace; font-size: 0.62rem; color: var(--text-dim);
  }
  .monitor-bar.visible { display: flex; }
  .mon-indicator { width: 8px; height: 8px; border-radius: 50%; background: var(--text-dim); flex-shrink: 0; }
  .mon-indicator.active { background: var(--accent-o); animation: pulse-btn 1.5s infinite; }
  .mon-label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
  .mon-label input[type=checkbox] { accent-color: var(--accent-o); cursor: pointer; }
  .mon-num { width: 50px; padding: 3px 7px; background: var(--surface); border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-family: 'Space Mono', monospace; font-size: 0.62rem; }
  .new-badge { background: var(--accent-o); color: var(--bg); padding: 1px 7px; border-radius: 10px; font-size: 0.58rem; font-weight: 700; }

  /* game selector */
  .game-sel-row {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--text-dim);
  }
  .game-num-input {
    width: 90px; padding: 6px 10px;
    background: var(--surface2); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text); font-family: 'Space Mono', monospace; font-size: 0.7rem; outline: none;
  }
  .game-num-input:focus { border-color: var(--accent-o); }
  .game-num-input:disabled { opacity: 0.4; }

  /* viewer board */
  .v-board-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .v-board {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 7px;
    max-width: 380px;
    margin: 0 auto;
  }
  .v-cell {
    aspect-ratio: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Bebas Neue', sans-serif; font-size: 2rem;
    position: relative; transition: all 0.2s;
  }
  .v-cell.x { color: var(--accent-x); text-shadow: 0 0 12px var(--glow-x); }
  .v-cell.o { color: var(--accent-o); text-shadow: 0 0 12px var(--glow-o); }
  .v-cell.highlight {
    border-color: var(--accent-warn);
    box-shadow: 0 0 10px rgba(255,209,102,0.3);
    background: rgba(255,209,102,0.06);
  }
  .v-cell-label {
    position: absolute; top: 3px; left: 5px;
    font-family: 'Space Mono', monospace; font-size: 0.48rem; color: var(--text-dim);
    font-weight: normal;
  }

  /* nav */
  .v-nav {
    display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 14px;
    font-family: 'Space Mono', monospace;
  }
  .v-nav-btn {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    padding: 8px 14px; border-radius: 7px; cursor: pointer; font-size: 1rem;
    transition: all 0.15s;
  }
  .v-nav-btn:hover:not(:disabled) { border-color: var(--accent-o); color: var(--accent-o); }
  .v-nav-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .v-move-disp { font-size: 0.7rem; color: var(--text-dim); min-width: 90px; text-align: center; }

  /* viewer info panel */
  .v-info-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .v-info-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.72rem; }
  .v-info-row:last-child { margin-bottom: 0; }
  .v-info-label { color: var(--text-dim); }
  .v-info-value { color: var(--text); font-weight: 600; font-family: 'Space Mono', monospace; font-size: 0.65rem; }
  .v-result { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 0.62rem; font-weight: 700; }
  .v-result.win-x { background: rgba(255,69,96,0.12); color: var(--accent-x); border: 1px solid rgba(255,69,96,0.3); }
  .v-result.win-o { background: rgba(0,201,167,0.12); color: var(--accent-o); border: 1px solid rgba(0,201,167,0.3); }
  .v-result.draw  { background: rgba(255,209,102,0.12); color: var(--accent-warn); border: 1px solid rgba(255,209,102,0.3); }

  /* stats tab */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; padding-top: 4px; }
  .stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
  .stat-card h3 { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 14px; }
  .stat-value { font-family: 'Bebas Neue', sans-serif; font-size: 2.6rem; color: var(--accent-o); margin-bottom: 4px; }
  .stat-label { font-size: 0.7rem; color: var(--text-dim); }
  .bar-item { margin-bottom: 12px; }
  .bar-lbl { display: flex; justify-content: space-between; font-family: 'Space Mono', monospace; font-size: 0.6rem; color: var(--text-dim); margin-bottom: 5px; }
  .bar-bg { background: var(--surface2); border-radius: 4px; height: 20px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: 0.6rem; font-weight: 700; color: var(--bg); transition: width 0.5s ease; }
  .bar-fill.x-color { background: linear-gradient(90deg, var(--accent-x), #ff8a9a); }
  .bar-fill.o-color { background: linear-gradient(90deg, var(--accent-o), #00e8c0); }
  .bar-fill.draw-color { background: linear-gradient(90deg, var(--accent-warn), #ffe599); }
  .histogram { display: flex; align-items: flex-end; justify-content: space-around; height: 120px; margin-top: 14px; gap: 6px; }
  .hbar { flex: 1; background: linear-gradient(180deg, var(--accent-x) 0%, var(--accent-o) 100%); border-radius: 4px 4px 0 0; position: relative; min-height: 2px; transition: all 0.3s; }
  .hbar:hover { opacity: 0.8; transform: translateY(-2px); }
  .hbar-lbl { position: absolute; bottom: -18px; left: 0; right: 0; text-align: center; font-family: 'Space Mono', monospace; font-size: 0.48rem; color: var(--text-dim); }
  .hbar-val { position: absolute; top: -18px; left: 0; right: 0; text-align: center; font-family: 'Space Mono', monospace; font-size: 0.5rem; color: var(--accent-o); font-weight: 700; }
  .no-data { text-align: center; color: var(--text-dim); padding: 40px; font-family: 'Space Mono', monospace; font-size: 0.7rem; }

  /* Viewer sub-tabs */
  .sub-tabs { display: flex; gap: 4px; margin-bottom: 16px; }
  .sub-tab { font-family: 'Space Mono', monospace; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1px; padding: 7px 16px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface2); color: var(--text-dim); cursor: pointer; transition: all 0.15s; }
  .sub-tab:hover { color: var(--text); }
  .sub-tab.active { border-color: var(--accent-o); color: var(--accent-o); background: rgba(0,201,167,0.07); }
  .sub-content { display: none; }
  .sub-content.active { display: block; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--surface2); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>
<div class="container">

  <!-- ── Header ── -->
  <header>
    <h1>MISÈRE 4×4</h1>
    <div class="subtitle">Neural Tic-Tac-Toe<br>3-in-a-row LOSES</div>
  </header>

  <!-- ── Page-level tabs ── -->
  <div class="page-tabs">
    <button class="page-tab active" onclick="switchPage('game')">⚡ Play vs AI</button>
    <button class="page-tab"       onclick="switchPage('viewer')">📂 Game Viewer</button>
  </div>

  <!-- ══════════════════════════════════════════════════════
       PAGE: PLAY
       ══════════════════════════════════════════════════════ -->
  <div id="page-game" class="page-content active">
    <div class="main-layout">

      <!-- Left: board + controls -->
      <div class="board-wrap">
        <div class="turn-bar">
          <div>
            <div class="turn-label">Current Turn</div>
            <div class="turn-player X" id="turnPlayer">X</div>
          </div>
          <div class="rule-badge">3 in a row = LOSE</div>
        </div>

        <div class="board" id="board"></div>

        <div class="game-result" id="gameResult">
          <div class="winner-text" id="winnerText"></div>
          <div class="sub" id="winnerSub"></div>
        </div>

        <div class="controls">
          <button class="btn btn-ai" id="btnAI" onclick="requestAI()">⚡ Ask AI</button>
          <button class="btn btn-reset" onclick="resetGame()">↺ Reset</button>
        </div>
      </div>

      <!-- Right: sidebar -->
      <div class="sidebar">
        <div class="value-panel" id="valuePanel">
          <div class="value-label">🧠 Neural Value — Current Player</div>
          <div class="value-display" id="valueDisplay">—</div>
          <div class="value-interp" id="valueInterp"></div>
          <div class="value-bar-wrap">
            <div class="value-bar" id="valueBar" style="width:50%"></div>
          </div>
        </div>

        <div class="config-panel">
          <div class="config-title">⚙ Proxy → Triton</div>
          <div class="config-row">
            <label>Host:Port</label>
            <input class="config-input" id="tritonHost" value="http://localhost:5555" placeholder="http://localhost:5555">
          </div>
          <div style="margin-top:8px; display:flex; align-items:center;">
            <span class="status-dot" id="statusDot"></span>
            <span class="status-text" id="statusText">Not connected</span>
          </div>
        </div>

        <div class="history-panel">
          <div class="history-title">📜 Move History</div>
          <div class="history-list" id="historyList"></div>
        </div>
      </div>
    </div>
  </div><!-- /page-game -->


  <!-- ══════════════════════════════════════════════════════
       PAGE: VIEWER
       ══════════════════════════════════════════════════════ -->
  <div id="page-viewer" class="page-content">

    <!-- Sub-tabs -->
    <div class="sub-tabs">
      <button class="sub-tab active" onclick="switchSub('viewer-game')">🎮 Game Viewer</button>
      <button class="sub-tab"       onclick="switchSub('viewer-stats')">📊 Statistics</button>
    </div>

    <!-- File loader (shared) -->
    <div class="load-panel" style="margin-bottom:16px;">
      <div class="config-title">📁 Load PGN Files</div>
      <label for="folderInput" class="file-btn-label">📁 Load Folder with PGN Files</label>
      <input type="file" id="folderInput" webkitdirectory directory multiple accept=".pgn">

      <div class="monitor-bar" id="monitorBar">
        <div class="mon-indicator" id="monIndicator"></div>
        <span id="monStatus">Monitoring stopped</span>
        <label class="mon-label">
          <input type="checkbox" id="autoMonitor" checked> Auto-refresh
        </label>
        <label class="mon-label">
          Every <input type="number" id="refreshInterval" class="mon-num" min="1" max="60" value="3"> sec
        </label>
        <span id="refreshInfo"></span>
      </div>

      <div class="game-sel-row" style="margin-top:12px;">
        <span>Game #</span>
        <input type="number" class="game-num-input" id="gameNumber" min="1" placeholder="—" disabled>
        <span id="totalGames">of 0</span>
      </div>
    </div>

    <!-- Sub-content: game viewer -->
    <div id="sub-viewer-game" class="sub-content active">
      <div class="viewer-layout">
        <div class="viewer-left">
          <div class="v-board-wrap">
            <div class="v-board" id="vBoard"></div>
            <div class="v-nav">
              <button class="v-nav-btn" onclick="vFirstMove()" id="vBtnFirst">⏮</button>
              <button class="v-nav-btn" onclick="vPrevMove()"  id="vBtnPrev">◀</button>
              <div class="v-move-disp" id="vMovDisp">Move 0/0</div>
              <button class="v-nav-btn" onclick="vNextMove()"  id="vBtnNext">▶</button>
              <button class="v-nav-btn" onclick="vLastMove()"  id="vBtnLast">⏭</button>
            </div>
          </div>
        </div>

        <div class="v-info-panel" id="vInfoPanel">
          <div class="no-data">📁 Load PGN folder<br>to begin</div>
        </div>
      </div>
    </div>

    <!-- Sub-content: stats -->
    <div id="sub-viewer-stats" class="sub-content">
      <div id="statsArea"><div class="no-data">📊 Load games to see statistics</div></div>
    </div>

  </div><!-- /page-viewer -->

</div><!-- /container -->

<script>
// ═══════════════════════════════════════════════════════════════════════
// PAGE SWITCHING
// ═══════════════════════════════════════════════════════════════════════
function switchPage(name) {
  document.querySelectorAll('.page-tab').forEach((t,i) => {
    t.classList.toggle('active', ['game','viewer'][i] === name);
  });
  document.querySelectorAll('.page-content').forEach(c => c.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
}

function switchSub(name) {
  document.querySelectorAll('.sub-tab').forEach((t,i) => {
    t.classList.toggle('active', ['viewer-game','viewer-stats'][i] === name);
  });
  document.querySelectorAll('.sub-content').forEach(c => c.classList.remove('active'));
  document.getElementById('sub-'+name).classList.add('active');
  if (name === 'viewer-stats') renderStatistics();
}

// ═══════════════════════════════════════════════════════════════════════
// GAME (Play tab)
// ═══════════════════════════════════════════════════════════════════════
const EMPTY = null;
let board = Array(16).fill(EMPTY);
let currentPlayer = 'X';
let gameOver = false;
let moveHistory = [];
let aiShowing = false;

function initBoard() {
  const boardEl = document.getElementById('board');
  boardEl.innerHTML = '';
  for (let i = 0; i < 16; i++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.id = `cell-${i}`;
    cell.innerHTML = `
      <div class="ai-bg" id="ai-bg-${i}"></div>
      <span class="piece" id="piece-${i}"></span>
      <span class="ai-prob" id="ai-prob-${i}"></span>`;
    cell.addEventListener('click', () => handleCellClick(i));
    boardEl.appendChild(cell);
  }
}

function renderBoard() {
  for (let i = 0; i < 16; i++) {
    const cell = document.getElementById(`cell-${i}`);
    const piece = document.getElementById(`piece-${i}`);
    const v = board[i];
    if (v) { piece.textContent = v; piece.className = `piece ${v}`; cell.classList.add('taken'); }
    else   { piece.textContent = ''; piece.className = 'piece'; cell.classList.remove('taken'); }
    if (gameOver) cell.classList.add('game-over');
    else          cell.classList.remove('game-over');
  }
  if (!gameOver) {
    const tp = document.getElementById('turnPlayer');
    tp.textContent = currentPlayer;
    tp.className = `turn-player ${currentPlayer}`;
  }
}

function getLines() {
  const lines = [];
  for (let r = 0; r < 4; r++) for (let c = 0; c <= 1; c++) lines.push([r*4+c, r*4+c+1, r*4+c+2]);
  for (let c = 0; c < 4; c++) for (let r = 0; r <= 1; r++) lines.push([r*4+c, (r+1)*4+c, (r+2)*4+c]);
  for (let r = 0; r <= 1; r++) for (let c = 0; c <= 1; c++) lines.push([r*4+c, (r+1)*4+c+1, (r+2)*4+c+2]);
  for (let r = 0; r <= 1; r++) for (let c = 2; c < 4; c++)  lines.push([r*4+c, (r+1)*4+c-1, (r+2)*4+c-2]);
  return lines;
}
const LINES = getLines();

function checkLoser() {
  for (const [a,b,c] of LINES)
    if (board[a] && board[a]===board[b] && board[b]===board[c])
      return { loser: board[a], line: [a,b,c] };
  return null;
}
function isDraw() { return board.every(c => c !== EMPTY); }

function handleCellClick(idx) {
  if (gameOver || board[idx] !== EMPTY) return;
  placeMove(idx, currentPlayer);
}

function placeMove(idx, player) {
  board[idx] = player;
  moveHistory.push({ player, idx, move: moveNumber(idx) });
  addHistory(player, idx);
  const result = checkLoser();
  if (result) {
    renderBoard();
    endGame(result);
  } else if (isDraw()) {
    renderBoard();
    endGame(null);
  } else {
    currentPlayer = player === 'X' ? 'O' : 'X'; // update BEFORE render so indicator is correct
    clearAIOverlay();
    renderBoard();
  }
}

function moveNumber(idx) { return `R${Math.floor(idx/4)+1}C${(idx%4)+1}`; }

function endGame(result) {
  gameOver = true;
  const resultEl = document.getElementById('gameResult');
  const winnerText = document.getElementById('winnerText');
  const winnerSub  = document.getElementById('winnerSub');
  resultEl.classList.add('visible');
  if (!result) {
    winnerText.textContent = 'DRAW'; winnerText.className = 'winner-text draw';
    winnerSub.textContent = 'No three in a row — board full';
  } else {
    const { loser, line } = result;
    const winner = loser === 'X' ? 'O' : 'X';
    winnerText.textContent = `${winner} WINS`; winnerText.className = `winner-text ${loser}`;
    winnerSub.textContent = `${loser} formed 3-in-a-row and loses!`;
    line.forEach(i => document.getElementById(`cell-${i}`).classList.add('losing'));
  }
  const tp = document.getElementById('turnPlayer');
  tp.textContent = result ? (result.loser==='X'?'O':'X')+' WIN' : 'DRAW';
  document.getElementById('btnAI').disabled = true;
}

function addHistory(player, idx) {
  const list = document.getElementById('historyList');
  const item = document.createElement('div');
  item.className = `history-item ${player}`;
  item.textContent = `${player}:${moveNumber(idx)}`;
  list.appendChild(item);
  list.scrollTop = list.scrollHeight;
}

function resetGame() {
  board = Array(16).fill(EMPTY);
  currentPlayer = 'X'; gameOver = false; moveHistory = []; aiShowing = false;
  document.getElementById('gameResult').classList.remove('visible');
  document.getElementById('historyList').innerHTML = '';
  document.getElementById('valuePanel').classList.remove('visible');
  document.getElementById('btnAI').disabled = false;
  const tp = document.getElementById('turnPlayer');
  tp.textContent = 'X'; tp.className = 'turn-player X';
  initBoard(); renderBoard();
}

function boardToFloat(player) {
  const out = new Float32Array(48);
  const plane2val = player === 'X' ? 0.0 : 1.0;
  for (let x = 0; x < 4; x++) for (let y = 0; y < 4; y++) {
    const idx = x*4+y; const cell = board[idx];
    if (cell === player) out[0*16+idx] = 1.0;
    else if (cell !== null) out[1*16+idx] = 1.0;
    out[2*16+idx] = plane2val;
  }
  return out;
}
function getLegalMask() {
  const out = new Float32Array(16);
  for (let i=0;i<16;i++) out[i] = board[i]===EMPTY ? 1.0 : 0.0;
  return out;
}

async function callTriton(boardFloat, maskFloat) {
  const host = document.getElementById('tritonHost').value.trim();
  const url = `${host}/v2/models/AlphaZero/infer`;
  const payload = {
    inputs: [
      { name:"boards", shape:[1,48], datatype:"FP32", data: Array.from(boardFloat) },
      { name:"mask",   shape:[1,16], datatype:"FP32", data: Array.from(maskFloat)  }
    ],
    outputs: [{ name:"policy" }, { name:"value" }]
  };
  const resp = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
  if (!resp.ok) { const t = await resp.text(); throw new Error(`Triton error ${resp.status}: ${t}`); }
  const data = await resp.json();
  return {
    policy: data.outputs.find(o=>o.name==='policy').data,
    value:  data.outputs.find(o=>o.name==='value').data[0]
  };
}

async function requestAI() {
  if (gameOver) return;
  const btn = document.getElementById('btnAI');
  btn.disabled = true; btn.classList.add('loading'); btn.textContent = '⏳ Thinking...';
  setStatus('loading','Querying Triton...');
  try {
    const { policy, value } = await callTriton(boardToFloat(currentPlayer), getLegalMask());
    setStatus('ok','Connected');
    showAIOverlay(policy, value);
    let bestIdx=-1, bestProb=-1;
    for (let i=0;i<16;i++) if (board[i]===EMPTY && policy[i]>bestProb) { bestProb=policy[i]; bestIdx=i; }
    setTimeout(() => { if (!gameOver && bestIdx>=0) placeMove(bestIdx, currentPlayer); }, 800);
  } catch(err) { setStatus('err', err.message.slice(0,60)); console.error(err); }
  finally { btn.disabled=gameOver; btn.classList.remove('loading'); btn.textContent='⚡ Ask AI'; }
}

function showAIOverlay(policy, value) {
  const maxP = Math.max(...policy);
  for (let i=0;i<16;i++) {
    const cell=document.getElementById(`cell-${i}`);
    const bg=document.getElementById(`ai-bg-${i}`);
    const probEl=document.getElementById(`ai-prob-${i}`);
    const norm = maxP>0 ? policy[i]/maxP : 0;
    bg.style.background = `rgba(0,${Math.round(norm*201)},${Math.round(norm*100)},${0.1+norm*0.35})`;
    if (board[i]===EMPTY) { probEl.textContent=(policy[i]*100).toFixed(1)+'%'; cell.classList.add('ai-shown'); }
    else { probEl.textContent=''; cell.classList.remove('ai-shown'); bg.style.background='transparent'; }
  }
  const vPanel=document.getElementById('valuePanel');
  const vDisplay=document.getElementById('valueDisplay');
  const vInterp=document.getElementById('valueInterp');
  const vBar=document.getElementById('valueBar');
  vPanel.classList.add('visible');
  vDisplay.textContent = value.toFixed(3);
  const vPct = ((value+1)/2)*100;
  if (value>0.15) { vDisplay.className='value-display value-good'; vInterp.textContent='Favourable position for current player'; vBar.style.background='var(--accent-o)'; }
  else if (value<-0.15) { vDisplay.className='value-display value-bad'; vInterp.textContent='Unfavourable — be careful!'; vBar.style.background='var(--accent-x)'; }
  else { vDisplay.className='value-display value-neutral'; vInterp.textContent='Balanced / unclear position'; vBar.style.background='var(--accent-warn)'; }
  vBar.style.width=`${Math.max(4,Math.min(96,vPct))}%`;
  aiShowing=true;
}

function clearAIOverlay() {
  if (!aiShowing) return;
  for (let i=0;i<16;i++) {
    document.getElementById(`cell-${i}`).classList.remove('ai-shown');
    document.getElementById(`ai-bg-${i}`).style.background='transparent';
    document.getElementById(`ai-prob-${i}`).textContent='';
  }
  aiShowing=false;
}

function setStatus(state, msg) {
  document.getElementById('statusDot').className=`status-dot ${state}`;
  document.getElementById('statusText').textContent=msg;
}

// ═══════════════════════════════════════════════════════════════════════
// VIEWER (Game Viewer tab)
// ═══════════════════════════════════════════════════════════════════════
let vGames=[], vCurrentGame=-1, vCurrentMove=0, vStatistics=null;
let folderHandle=null, monitoringInterval=null, lastGameCount=0;
const vBoard2d = Array(4).fill(null).map(()=>Array(4).fill(null));
const vRowMap = {'a':0,'b':1,'c':2,'d':3};
const vColLabels=['1','2','3','4'];
const vRowLabels=['a','b','c','d'];

// DOM refs (safe — accessed after DOMContentLoaded equivalent, i.e. end of script)
const folderInput    = () => document.getElementById('folderInput');
const gameNumber     = () => document.getElementById('gameNumber');
const totalGamesEl   = () => document.getElementById('totalGames');
const monitorBar     = () => document.getElementById('monitorBar');
const monIndicator   = () => document.getElementById('monIndicator');
const monStatus      = () => document.getElementById('monStatus');
const autoMonitor    = () => document.getElementById('autoMonitor');
const refreshInterval= () => document.getElementById('refreshInterval');
const refreshInfo    = () => document.getElementById('refreshInfo');

function parseVMove(s) { return { row: vRowMap[s[1]], col: parseInt(s[0])-1 }; }

function parsePGN(content, filename) {
  const lines = content.trim().split('\n');
  const g = { filename, event:'', playerX:'', playerO:'', result:'', moves:[] };
  lines.forEach(line => {
    if (line.startsWith('[Event')) g.event = line.match(/"(.+)"/)?.[1]||'';
    else if (line.startsWith('[X'))  g.playerX = line.match(/"(.+)"/)?.[1]||'';
    else if (line.startsWith('[O'))  g.playerO = line.match(/"(.+)"/)?.[1]||'';
    else if (line.match(/^[01]-[01]|^1\/2-1\/2$/)) g.result = line;
    else if (line.match(/^\d+\./)) {
      for (const m of line.matchAll(/\d+\.(\w+)/g)) g.moves.push(m[1]);
    }
  });
  return g;
}

function boardToStr(b) { return b.map(r=>r.map(c=>c||'.').join('')).join(''); }

function calculateStatistics() {
  const s = { totalGames:vGames.length, xWins:0, oWins:0, draws:0, totalMoves:0,
              gameLengths:[], positions:new Set(), boardRepetitions:{}, uniqueMoves:new Set(),
              endingBoards:new Set(), openingSequences:new Set(), totalPositionOccurrences:0 };
  vGames.forEach(g => {
    if      (g.result==='1-0')     s.xWins++;
    else if (g.result==='0-1')     s.oWins++;
    else if (g.result==='1/2-1/2') s.draws++;
    s.totalMoves += g.moves.length;
    s.gameLengths.push(g.moves.length);
    g.moves.forEach(m=>s.uniqueMoves.add(m));
    // Opening: first 4 moves as a sequence key
    const opening = g.moves.slice(0,4).join('-');
    if (g.moves.length >= 4) s.openingSequences.add(opening);
    const tb=Array(4).fill(null).map(()=>Array(4).fill(null));
    g.moves.forEach((m,i)=>{
      const p=parseVMove(m); tb[p.row][p.col]=i%2===0?'X':'O';
      const bs=boardToStr(tb); s.positions.add(bs);
      s.boardRepetitions[bs]=(s.boardRepetitions[bs]||0)+1;
      s.totalPositionOccurrences++; // every board state reached across all games
    });
    s.endingBoards.add(boardToStr(tb));
  });
  s.xWinPct=(s.xWins/s.totalGames*100).toFixed(1);
  s.oWinPct=(s.oWins/s.totalGames*100).toFixed(1);
  s.drawPct=(s.draws/s.totalGames*100).toFixed(1);
  s.avgMoves=(s.totalMoves/s.totalGames).toFixed(1);
  const lb={}; s.gameLengths.forEach(l=>{lb[l]=(lb[l]||0)+1;});
  s.lengthDistribution=lb;
  s.repeatedBoards=Object.values(s.boardRepetitions).filter(c=>c>1).length;
  // Opening diversity: games with >=4 moves
  const gamesWithOpening = vGames.filter(g=>g.moves.length>=4).length;
  s.openingPct = gamesWithOpening > 0 ? (s.openingSequences.size / gamesWithOpening * 100).toFixed(1) : '0.0';
  s.openingGamesTotal = gamesWithOpening;
  return s;
}

function renderStatistics() {
  const area = document.getElementById('statsArea');
  if (!vStatistics || vGames.length===0) {
    area.innerHTML='<div class="no-data">📊 Load games to see statistics</div>'; return;
  }
  const s=vStatistics;
  const maxCount=Math.max(...Object.values(s.lengthDistribution));
  area.innerHTML=`
    <div class="stats-grid">
      <div class="stat-card">
        <h3>Win Rates</h3>
        <div class="bar-item">
          <div class="bar-lbl"><span>❌ X (Player 1)</span><span>${s.xWins} (${s.xWinPct}%)</span></div>
          <div class="bar-bg"><div class="bar-fill x-color" style="width:${s.xWinPct}%">${s.xWinPct}%</div></div>
        </div>
        <div class="bar-item">
          <div class="bar-lbl"><span>⭕ O (Player 2)</span><span>${s.oWins} (${s.oWinPct}%)</span></div>
          <div class="bar-bg"><div class="bar-fill o-color" style="width:${s.oWinPct}%">${s.oWinPct}%</div></div>
        </div>
        <div class="bar-item">
          <div class="bar-lbl"><span>🤝 Draws</span><span>${s.draws} (${s.drawPct}%)</span></div>
          <div class="bar-bg"><div class="bar-fill draw-color" style="width:${s.drawPct}%">${s.drawPct}%</div></div>
        </div>
      </div>
      <div class="stat-card">
        <h3>Avg Game Length</h3>
        <div class="stat-value">${s.avgMoves}</div>
        <div class="stat-label">moves per game</div>
        <div style="margin-top:12px;font-size:0.65rem;color:var(--text-dim);">
          <div>Shortest: ${Math.min(...s.gameLengths)} moves</div>
          <div>Longest:  ${Math.max(...s.gameLengths)} moves</div>
        </div>
      </div>
      <div class="stat-card">
        <h3>Unique Positions</h3>
        <div class="stat-value" style="font-size:1.8rem;line-height:1.2;">${s.positions.size.toLocaleString()} <span style="color:var(--text-dim);font-size:1.1rem;">/ ${s.totalPositionOccurrences.toLocaleString()}</span> <span style="color:var(--accent-o);font-size:1rem;">(${(s.positions.size/s.totalPositionOccurrences*100).toFixed(1)}%)</span></div>
        <div class="stat-label">distinct board states</div>
        <div style="margin-top:12px;font-size:0.65rem;color:var(--text-dim);">
          <div>Repeated: ${s.repeatedBoards.toLocaleString()}</div>
          <div>Avg/game: ${(s.positions.size/s.totalGames).toFixed(1)}</div>
        </div>
      </div>
      <div class="stat-card">
        <h3>Opening Diversity</h3>
        <div class="stat-value">${s.openingSequences.size.toLocaleString()}</div>
        <div class="stat-label">unique opening sequences (first 4 moves)</div>
        <div style="margin-top:12px;font-size:0.65rem;color:var(--text-dim);">
          <div>Out of ${s.openingGamesTotal.toLocaleString()} games with ≥4 moves</div>
          <div style="margin-top:5px;">
            <div class="bar-bg" style="margin-top:4px;">
              <div class="bar-fill o-color" style="width:${Math.min(100,s.openingPct)}%">${s.openingPct}%</div>
            </div>
          </div>
        </div>
      </div>
      <div class="stat-card">
        <h3>Ending Board Diversity</h3>
        <div class="stat-value">${s.endingBoards.size.toLocaleString()}</div>
        <div class="stat-label">unique game endings</div>
        <div style="margin-top:12px;font-size:0.65rem;color:var(--text-dim);">
          <div>Out of ${s.totalGames.toLocaleString()} total games</div>
          <div style="margin-top:5px;">
            <div class="bar-bg" style="margin-top:4px;">
              <div class="bar-fill x-color" style="width:${Math.min(100,(s.endingBoards.size/s.totalGames*100).toFixed(1))}%">${(s.endingBoards.size/s.totalGames*100).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      </div>
      <div class="stat-card">
        <h3>Game Length Distribution</h3>
        <div class="histogram">
          ${Object.keys(s.lengthDistribution).sort((a,b)=>a-b).map(len=>{
            const cnt=s.lengthDistribution[len];
            const h=(cnt/maxCount*100);
            return `<div class="hbar" style="height:${h}%">
              <div class="hbar-val">${cnt}</div>
              <div class="hbar-lbl">${len}m</div>
            </div>`;
          }).join('')}
        </div>
      </div>
    </div>`;
}

function vResetBoard() { for(let i=0;i<4;i++) for(let j=0;j<4;j++) vBoard2d[i][j]=null; }

function vApplyMoves(count) {
  vResetBoard();
  const g=vGames[vCurrentGame];
  for (let i=0;i<count&&i<g.moves.length;i++) {
    const p=parseVMove(g.moves[i]);
    vBoard2d[p.row][p.col]=i%2===0?'X':'O';
  }
}

function renderViewerBoard() {
  const vb=document.getElementById('vBoard');
  const info=document.getElementById('vInfoPanel');
  if (vCurrentGame===-1||vGames.length===0) {
    vb.innerHTML=''; info.innerHTML='<div class="no-data">📁 Load PGN folder<br>to begin</div>'; return;
  }
  const g=vGames[vCurrentGame];
  vApplyMoves(vCurrentMove);

  vb.innerHTML = vBoard2d.map((row,i)=>row.map((cell,j)=>{
    const lbl=`${vColLabels[j]}${vRowLabels[i]}`;
    const isLast=vCurrentMove>0 && g.moves[vCurrentMove-1]===lbl;
    return `<div class="v-cell ${cell?cell.toLowerCase():''} ${isLast?'highlight':''}">
      <span class="v-cell-label">${lbl}</span>${cell||''}
    </div>`;
  }).join('')).join('');

  // Nav buttons
  document.getElementById('vBtnFirst').disabled = vCurrentMove===0;
  document.getElementById('vBtnPrev').disabled  = vCurrentMove===0;
  document.getElementById('vBtnNext').disabled  = vCurrentMove>=g.moves.length;
  document.getElementById('vBtnLast').disabled  = vCurrentMove>=g.moves.length;
  document.getElementById('vMovDisp').textContent = `Move ${vCurrentMove}/${g.moves.length}`;

  const rClass = g.result==='1-0'?'win-x': g.result==='0-1'?'win-o':'draw';
  const rText  = g.result==='1-0'?'X Wins': g.result==='0-1'?'O Wins':'Draw';
  info.innerHTML=`
    <div class="config-title" style="margin-bottom:12px;">📋 Game Info</div>
    <div class="v-info-row"><span class="v-info-label">File</span><span class="v-info-value">${g.filename}</span></div>
    <div class="v-info-row"><span class="v-info-label">Event</span><span class="v-info-value">${g.event||'—'}</span></div>
    <div class="v-info-row"><span class="v-info-label">Result</span><span class="v-result ${rClass}">${rText} (${g.result})</span></div>
    <div class="v-info-row"><span class="v-info-label">Last Move</span><span class="v-info-value">${vCurrentMove>0?g.moves[vCurrentMove-1]:'—'}</span></div>
    <div class="v-info-row"><span class="v-info-label">Next Move</span><span class="v-info-value">${vCurrentMove<g.moves.length?g.moves[vCurrentMove]:'—'}</span></div>
  `;
}

function vLoadGame(idx) { vCurrentGame=idx; vCurrentMove=0; gameNumber().value=idx+1; renderViewerBoard(); }
function vFirstMove() { vCurrentMove=0; renderViewerBoard(); }
function vPrevMove()  { if(vCurrentMove>0){vCurrentMove--;renderViewerBoard();} }
function vNextMove()  { if(vCurrentGame>=0&&vCurrentMove<vGames[vCurrentGame].moves.length){vCurrentMove++;renderViewerBoard();} }
function vLastMove()  { if(vCurrentGame>=0){vCurrentMove=vGames[vCurrentGame].moves.length;renderViewerBoard();} }

async function loadGamesFromFiles(files) {
  const pgnFiles=Array.from(files).filter(f=>f.name.endsWith('.pgn'));
  const gs=[];
  for (const f of pgnFiles) { gs.push(parsePGN(await f.text(), f.name)); }
  return gs;
}

async function refreshGames() {
  if (!folderHandle) return;
  try {
    const files=[];
    for await (const entry of folderHandle.values())
      if (entry.kind==='file'&&entry.name.endsWith('.pgn')) files.push(await entry.getFile());
    const newGames=await loadGamesFromFiles(files);
    newGames.sort((a,b)=>a.filename.localeCompare(b.filename));
    const added=newGames.length-lastGameCount;
    if (added>0) {
      vGames=newGames; lastGameCount=newGames.length;
      totalGamesEl().textContent=`of ${vGames.length}`;
      gameNumber().max=vGames.length;
      vStatistics=calculateStatistics();
      const now=new Date();
      refreshInfo().innerHTML=`<span class="new-badge">+${added} new</span> ${now.toLocaleTimeString()}`;
      if (document.getElementById('sub-viewer-stats').classList.contains('active')) renderStatistics();
      if (vCurrentGame>=0&&vCurrentGame<vGames.length) renderViewerBoard();
    }
    monStatus().textContent=`Monitoring active (${vGames.length} games)`;
  } catch(e) { console.error(e); monStatus().textContent='Monitoring error'; }
}

function startMonitoring() {
  if (monitoringInterval) return;
  const interval=parseInt(refreshInterval().value)*1000;
  monitoringInterval=setInterval(refreshGames, interval);
  monIndicator().classList.add('active');
  monStatus().textContent=`Monitoring active (${vGames.length} games)`;
}
function stopMonitoring() {
  if (monitoringInterval) { clearInterval(monitoringInterval); monitoringInterval=null; }
  monIndicator().classList.remove('active');
  monStatus().textContent='Monitoring stopped';
}

folderInput().addEventListener('change', async (e) => {
  const files=Array.from(e.target.files).filter(f=>f.name.endsWith('.pgn'));
  if (!files.length) { alert('No PGN files found'); return; }
  if (e.target.files[0].webkitRelativePath) {
    try { folderHandle = await window.showDirectoryPicker(); } catch(_) {}
  }
  vGames = await loadGamesFromFiles(files);
  vGames.sort((a,b)=>a.filename.localeCompare(b.filename));
  lastGameCount=vGames.length;
  totalGamesEl().textContent=`of ${vGames.length}`;
  gameNumber().disabled=false; gameNumber().max=vGames.length; gameNumber().value=1;
  vCurrentGame=0; vStatistics=calculateStatistics();
  monitorBar().classList.add('visible');
  if (autoMonitor().checked && folderHandle) startMonitoring();
  else if (!folderHandle) refreshInfo().innerHTML='<span style="color:var(--accent-warn)">⚠ Auto-refresh requires modern browser</span>';
  vLoadGame(0);
});

autoMonitor().addEventListener('change', e => { if(e.target.checked&&folderHandle) startMonitoring(); else stopMonitoring(); });
refreshInterval().addEventListener('change', () => { if(monitoringInterval){stopMonitoring();if(autoMonitor().checked)startMonitoring();} });
gameNumber().addEventListener('input', e => { const n=parseInt(e.target.value); if(n>=1&&n<=vGames.length) vLoadGame(n-1); });
gameNumber().addEventListener('keydown', e => { if(e.key==='Enter') e.target.blur(); });

// Keyboard nav (only when in viewer tab)
document.addEventListener('keydown', e => {
  if (!document.getElementById('page-viewer').classList.contains('active')) return;
  if (document.activeElement.tagName==='INPUT') return;
  if (vGames.length===0) return;
  switch(e.key) {
    case 'ArrowLeft':  vPrevMove(); break;
    case 'ArrowRight': vNextMove(); break;
    case 'Home':       vFirstMove();break;
    case 'End':        vLastMove(); break;
    case 'ArrowUp':    if(vCurrentGame>0) vLoadGame(vCurrentGame-1); break;
    case 'ArrowDown':  if(vCurrentGame<vGames.length-1) vLoadGame(vCurrentGame+1); break;
  }
});
window.addEventListener('beforeunload', stopMonitoring);

// ─── Init ───
initBoard();
renderBoard();
renderViewerBoard();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HANDLER  (serves HTML + proxies Triton)
# ─────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[server] {args[0]} {args[1]} {args[2]}")

    def send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors()
        self.end_headers()

    def do_GET(self):
        # Serve the game UI
        if self.path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Proxy everything else → Triton
        try:
            target = TRITON_URL + self.path
            req = urllib.request.urlopen(target, timeout=10)
            resp_body = req.read()
            self.send_response(req.status)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp_body)
        except Exception as e:
            self.send_response(502)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            target = TRITON_URL + self.path
            req = urllib.request.Request(
                target, data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            resp = urllib.request.urlopen(req, timeout=10)
            resp_body = resp.read()
            self.send_response(resp.status)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as e:
            self.send_response(502)
            self.send_cors()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PROXY_PORT), Handler)
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  MISÈRE 4×4 — Neural Tic-Tac-Toe            ║")
    print(f"╠══════════════════════════════════════════════╣")
    print(f"║  UI   →  http://localhost:{PROXY_PORT}               ║")
    print(f"║  Fwds →  Triton at {TRITON_URL}       ║")
    print(f"╚══════════════════════════════════════════════╝")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")