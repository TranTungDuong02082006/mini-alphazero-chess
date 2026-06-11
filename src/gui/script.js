// Global variables
let board = null;
let game = new Chess();
let moveStartTime = Date.now();
let moveHistory = [];

// Clocks state
let whiteTime = 180;
let blackTime = 180;
let timerInterval = null;
let currentTurn = "w"; // 'w' or 'b'

// Player configuration
let humanColor = "w"; // "w" or "b"
let botName = "AlphaZero (Black)";
let humanName = "Human (White)";

const statusEl = document.getElementById("status");
const API_URL = "http://127.0.0.1:8000/move";

// Initialize Board and Dashboard on DOM Load
function init() {
  setupEventListeners();
  resetGame();
}

// Set up UI event listeners
function setupEventListeners() {
  document.getElementById("resetBtn").addEventListener("click", resetGame);
  
  document.getElementById("playerSide").addEventListener("change", function () {
    const side = this.value;
    if (side === "white") {
      humanColor = "w";
      botName = "AlphaZero (Black)";
      humanName = "Human (White)";
      document.getElementById("whitePlayerName").textContent = humanName;
      document.getElementById("blackPlayerName").textContent = botName;
      board.orientation("white");
    } else {
      humanColor = "b";
      botName = "AlphaZero (White)";
      humanName = "Human (Black)";
      document.getElementById("whitePlayerName").textContent = botName;
      document.getElementById("blackPlayerName").textContent = humanName;
      board.orientation("black");
    }
    resetGame();
  });
}

// Reset game state and clocks
function resetGame() {
  clearInterval(timerInterval);
  game.reset();
  
  whiteTime = 180;
  blackTime = 180;
  currentTurn = "w";
  moveHistory = [];
  moveStartTime = Date.now();

  updateClockDisplay();
  updateCapturedPiecesDisplay();
  updateStatus();
  renderMoveHistory();
  clearAIThoughts();
  updateEvalBar(0.0);

  // Initialize board widget
  if (board) {
    board.destroy();
  }
  
  board = Chessboard("board", {
    draggable: true,
    position: "start",
    onDragStart,
    onDrop,
    onSnapEnd,
    moveSpeed: "slow",
    pieceTheme: "./chessboardjs/img/chesspieces/wikipedia/{piece}.png"
  });

  startClock();

  // If human plays Black, Bot (White) moves first!
  if (humanColor === "b") {
    setTimeout(requestBotMove, 800);
  }
}

// Prevent dragging pieces if game is over or it's not human turn
function onDragStart(source, piece) {
  if (game.game_over()) return false;

  // Check turn: turn must match humanColor, and piece color must match humanColor
  const turn = game.turn(); // 'w' or 'b'
  if (turn !== humanColor) return false;
  if (piece.charAt(0) !== humanColor) return false;
}

// Handle human move drop
async function onDrop(source, target) {
  // Try to make the move (assume promotion to Queen for simplicity)
  const moveObj = game.move({ from: source, to: target, promotion: "q" });
  if (!moveObj) return "snapback";

  // Visual updates
  board.position(game.fen());
  updateCapturedPiecesDisplay();
  updateStatus();

  // Turn time tracking
  clearInterval(timerInterval);
  const duration = ((Date.now() - moveStartTime) / 1000).toFixed(2);
  updateMoveHistory(moveObj, duration, humanColor);
  moveStartTime = Date.now();

  // Switch turns
  currentTurn = game.turn();
  updateActivePlayerHighlight();
  startClock();

  // Request Bot response if game is not over
  if (!game.game_over()) {
    await requestBotMove();
  }
}

// Snapback transition helper
function onSnapEnd() {
  board.position(game.fen());
}

// Request Bot move from the Python FastAPI server
async function requestBotMove() {
  const fen = game.fen();
  const method = document.getElementById("engineMethod").value;
  statusEl.innerHTML = `<span class="status-active">Bot is thinking...</span>`;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen, method }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.status === "ok" && data.move) {
      // Play Bot move
      const botMoveObj = game.move(data.move, { sloppy: true });
      board.position(game.fen());
      updateCapturedPiecesDisplay();
      
      // Update Eval & Thoughts
      if (data.root_value !== undefined) {
        updateEvalBar(data.root_value);
      }
      if (data.top_moves) {
        renderAIThoughts(data.top_moves, data.root_value);
      }

      updateStatus();

      // Clock logic
      clearInterval(timerInterval);
      const duration = ((Date.now() - moveStartTime) / 1000).toFixed(2);
      // Bot's color is opposite to human color
      const botColor = humanColor === "w" ? "b" : "w";
      updateMoveHistory(botMoveObj, duration, botColor);
      moveStartTime = Date.now();

      // Switch turns
      currentTurn = game.turn();
      updateActivePlayerHighlight();
      startClock();
    } else {
      const reason = data.reason || "unknown";
      if (data.status === "game_over") {
        statusEl.innerHTML = `<span class="status-alert">Game Over: ${reason}</span>`;
      } else {
        statusEl.innerHTML = `<span class="status-alert">Bot Error: ${reason}</span>`;
      }
      clearInterval(timerInterval);
    }
  } catch (error) {
    console.error("API fetch failed:", error);
    statusEl.innerHTML = `<span class="status-alert">Cannot connect to API server. Verify it is running.</span>`;
    clearInterval(timerInterval);
  }
}

// --- CLOCK & ROW HILIGHTS ---
function updateClockDisplay() {
  document.getElementById("whiteClock").textContent = formatTime(whiteTime);
  document.getElementById("blackClock").textContent = formatTime(blackTime);
}

function formatTime(seconds) {
  let m = Math.floor(seconds / 60);
  let s = seconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function startClock() {
  clearInterval(timerInterval);
  updateActivePlayerHighlight();
  
  timerInterval = setInterval(() => {
    if (currentTurn === "w") {
      whiteTime--;
      if (whiteTime <= 0) {
        whiteTime = 0;
        endGame("⏰ White ran out of time — Black wins!");
      }
    } else {
      blackTime--;
      if (blackTime <= 0) {
        blackTime = 0;
        endGame("⏰ Black ran out of time — White wins!");
      }
    }
    updateClockDisplay();
  }, 1000);
}

// Update clock row border highlight
function updateActivePlayerHighlight() {
  const whiteRow = document.getElementById("whitePlayerRow");
  const blackRow = document.getElementById("blackPlayerRow");
  
  if (currentTurn === "w") {
    whiteRow.classList.add("active");
    blackRow.classList.remove("active");
  } else {
    blackRow.classList.add("active");
    whiteRow.classList.remove("active");
  }
}

function endGame(msg) {
  clearInterval(timerInterval);
  statusEl.innerHTML = `<span class="status-alert">${msg}</span>`;
  alert(msg);
}

// --- ROBUST CAPTURED PIECES DISPLAY (Board Scanning) ---
function updateCapturedPiecesDisplay() {
  const startingCounts = {
    w: { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 },
    b: { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 }
  };
  const currentCounts = {
    w: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 },
    b: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 }
  };

  // Scan current board
  const boardState = game.board();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const square = boardState[r][c];
      if (square) {
        currentCounts[square.color][square.type]++;
      }
    }
  }

  // Find diffs
  const capturedWhite = []; // White pieces captured by Black
  for (const piece in startingCounts.w) {
    const diff = startingCounts.w[piece] - currentCounts.w[piece];
    for (let i = 0; i < diff; i++) {
      capturedWhite.push('w' + piece.toUpperCase());
    }
  }

  const capturedBlack = []; // Black pieces captured by White
  for (const piece in startingCounts.b) {
    const diff = startingCounts.b[piece] - currentCounts.b[piece];
    for (let i = 0; i < diff; i++) {
      capturedBlack.push('b' + piece.toUpperCase());
    }
  }

  // Image helper
  const pieceImg = (piece) =>
    `<img src="./chessboardjs/img/chesspieces/wikipedia/${piece}.png" alt="${piece}"/>`;

  // Render to respective spots
  document.getElementById("capturedBlack").innerHTML = capturedBlack.map(pieceImg).join("");
  document.getElementById("capturedWhite").innerHTML = capturedWhite.map(pieceImg).join("");
}

// --- EVALUATION BAR & THOUGHTS ---
function updateEvalBar(rootValue) {
  // Convert current player's perspective to White's perspective
  const turnSign = game.turn() === "w" ? 1 : -1;
  const evalWhite = rootValue * turnSign;

  // Map eval [-1, 1] to height percentage [2%, 98%]
  let fillPercent = 50 + (evalWhite * 50);
  fillPercent = Math.max(2, Math.min(98, fillPercent));

  const scoreStr = (evalWhite >= 0 ? "+" : "") + evalWhite.toFixed(2);

  document.getElementById("evalFill").style.height = fillPercent + "%";
  document.getElementById("evalLabel").textContent = scoreStr;
  
  // Set label position relative to evalFill height (bottom: fillPercent)
  document.getElementById("evalLabel").style.bottom = `calc(${fillPercent}% - 10px)`;
}

function clearAIThoughts() {
  document.getElementById("posEvalPrediction").textContent = "0.00 (Equal)";
  document.getElementById("thoughtList").innerHTML = `<div class="no-data-msg" id="noThoughtMsg">Make a move to see AI analysis...</div>`;
}

function renderAIThoughts(topMoves, rootValue) {
  const turnSign = game.turn() === "w" ? 1 : -1;
  const evalWhite = rootValue * turnSign;

  let textRating = "Equal";
  if (evalWhite > 0.6) textRating = "White is winning";
  else if (evalWhite > 0.15) textRating = "White is slightly better";
  else if (evalWhite < -0.6) textRating = "Black is winning";
  else if (evalWhite < -0.15) textRating = "Black is slightly better";

  document.getElementById("posEvalPrediction").textContent = `${(evalWhite >= 0 ? "+" : "")}${evalWhite.toFixed(2)} (${textRating})`;

  const listContainer = document.getElementById("thoughtList");
  listContainer.innerHTML = "";

  if (topMoves.length === 0) {
    listContainer.innerHTML = `<div class="no-data-msg">No move search stats available.</div>`;
    return;
  }

  // Find max visit count to scale bars relatively
  const maxVisits = Math.max(...topMoves.map(m => m.visit_count), 1);

  topMoves.forEach(m => {
    const percentWidth = (m.visit_count / maxVisits) * 100;
    const probabilityText = (m.probability * 100).toFixed(1) + "%";

    const row = document.createElement("div");
    row.className = "thought-row";
    row.innerHTML = `
      <div class="thought-move">${m.move}</div>
      <div class="thought-bar-outer">
        <div class="thought-bar-inner" style="width: ${percentWidth}%"></div>
      </div>
      <div class="thought-value">${probabilityText} (${m.visit_count})</div>
    `;
    listContainer.appendChild(row);
  });
}

// --- MOVE HISTORY ---
function updateMoveHistory(move, duration, color) {
  const historyLength = game.history().length;
  const turnIndex = Math.floor((historyLength - 1) / 2);

  if (color === "w") {
    moveHistory[turnIndex] = {
      turn: turnIndex + 1,
      white: move.san,
      whiteTime: `${duration}s`,
      black: "",
      blackTime: ""
    };
  } else {
    if (!moveHistory[turnIndex]) {
      moveHistory[turnIndex] = { turn: turnIndex + 1, white: "", whiteTime: "", black: "", blackTime: "" };
    }
    moveHistory[turnIndex].black = move.san;
    moveHistory[turnIndex].blackTime = `${duration}s`;
  }

  renderMoveHistory();
}

function renderMoveHistory() {
  const tbody = document.querySelector("#moveHistory tbody");
  tbody.innerHTML = "";

  moveHistory.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.turn}</td>
      <td class="${row.white ? 'active-move' : ''}">${row.white}</td>
      <td style="color: var(--text-muted); font-size: 0.85em;">${row.whiteTime}</td>
      <td class="${row.black ? 'active-move' : ''}">${row.black}</td>
      <td style="color: var(--text-muted); font-size: 0.85em;">${row.blackTime}</td>
    `;
    tbody.appendChild(tr);
  });

  // Scroll to bottom of history container
  const container = document.querySelector(".history-table-container");
  container.scrollTop = container.scrollHeight;
}

// --- STATUS LABEL ---
function updateStatus() {
  let status = "";
  const moveColor = game.turn() === "w" ? "White" : "Black";
  const activeName = game.turn() === humanColor ? "Your turn" : "Bot turn";

  if (game.in_checkmate()) {
    const winner = game.turn() === "w" ? "Black" : "White";
    status = `🏆 Game over! ${winner} wins by checkmate.`;
    statusEl.className = "status-container status-alert";
    clearInterval(timerInterval);
  } else if (game.in_draw()) {
    status = "🤝 Game drawn (Stalemate, Repetition, or 50 moves).";
    statusEl.className = "status-container status-muted";
    clearInterval(timerInterval);
  } else {
    status = `${moveColor} to move (${activeName})`;
    if (game.in_check()) status += ` — ⚠️ CHECK!`;
    statusEl.className = "status-container status-active";
  }

  statusEl.textContent = status;
}

// Hook onload
document.addEventListener("DOMContentLoaded", init);
