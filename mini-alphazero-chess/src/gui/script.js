// Khởi tạo bàn cờ và trò chơi
let board = null;
let game = new Chess();
let capturedWhite = []; // Danh sách các quân ĐEN bị TRẮNG bắt (hiển thị dưới đồng hồ Trắng)
let capturedBlack = []; // Danh sách các quân TRẮNG bị ĐEN bắt (hiển thị dưới đồng hồ Đen)
let moveStartTime = Date.now();
let moveHistory = [];

// thời gian ban đầu (giây)
let whiteTime = 180;
let blackTime = 180;
let whiteClock
let blackClock 
let timerInterval = null;

let currentTurn = "w"; // Lượt hiện tại để đồng hồ đếm

const statusEl = document.getElementById("status");
const API_URL = "http://127.0.0.1:8000/move";

// Kiểm tra điều kiện bắt đầu kéo quân
function onDragStart(source, piece) {
  if (game.game_over()) return false;
  // CHỈ cho phép Trắng đi (player là Trắng)
  if (game.turn() !== "w" || piece.startsWith("b")) return false;
}

// Xử lý khi người chơi thả quân (Lượt Trắng)
async function onDrop(source, target) {
  const move = game.move({ from: source, to: target, promotion: "q" });
  if (!move) return "snapback";

  updateCapturedPieces(move);
  board.position(game.fen());
  updateStatus(); 

  // Dừng đồng hồ Trắng và ghi thời gian
  clearInterval(timerInterval); 
  const duration = ((Date.now() - moveStartTime) / 1000).toFixed(2);
  updateMoveHistory(move, duration, "w");
  moveStartTime = Date.now();

  // Chuyển lượt sang Đen (Bot) và bắt đầu đồng hồ Đen
  currentTurn = "b"; 
  startClock(); 

  // Gọi Bot
  await requestBotMove();
}

// Cập nhật vị trí sau khi quân được thả
function onSnapEnd() {
  board.position(game.fen());
}

// Gửi FEN cho bot và xử lý phản hồi (Lượt Đen)
async function requestBotMove() {
  const fen = game.fen();
  statusEl.textContent = "Bot is thinking...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen, method: "best" }),
    });

    if (!response.ok) {
      statusEl.textContent = `Error: HTTP status ${response.status}. Cannot reach API server.`;
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.status === "ok" && data.move) {
      const botMove = game.move(data.move, { sloppy: true });
      updateCapturedPieces(botMove);
      board.position(game.fen());
      updateStatus();

      // Dừng đồng hồ Đen và ghi thời gian
      clearInterval(timerInterval);
      const duration = ((Date.now() - moveStartTime) / 1000).toFixed(2);
      updateMoveHistory(botMove, duration, "b");
      moveStartTime = Date.now();

      // Đổi lượt sang trắng và bắt đầu đồng hồ Trắng
      currentTurn = "w";
      startClock();
    } else {
      const reason = data.reason || "unknown";
      statusEl.textContent = data.status === "game_over"
        ? `Game over: ${reason}`
        : `Error: Bot response status: ${reason}`;
      console.error("Bot response error:", data);
      clearInterval(timerInterval); 
    }
  } catch (error) {
    console.error("API fetch failed:", error);
    statusEl.textContent = "Cannot reach API server. Check your bot server/CORS.";
    clearInterval(timerInterval); 
  }
}

// ======================== CLOCK ===========================
function updateClockDisplay() {
  whiteClock.textContent = formatTime(whiteTime);
  blackClock.textContent = formatTime(blackTime);
}

function formatTime(seconds) {
  let m = Math.floor(seconds / 60);
  let s = seconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

function startClock() {
  clearInterval(timerInterval); // Đảm bảo chỉ có một interval chạy
  timerInterval = setInterval(() => {
    if (currentTurn === "w") {
      whiteTime--;
      if (whiteTime <= 0) endGame("⏰ White ran out of time — Black wins!");
    } else {
      blackTime--;
      if (blackTime <= 0) endGame("⏰ Black ran out of time — White wins!");
    }
    updateClockDisplay();
  }, 1000);
}

function endGame(msg) {
  clearInterval(timerInterval);
  alert(msg);
  statusEl.textContent = msg;
}

// ==================== CAPTURED PIECES =====================
function updateCapturedPieces(move) {
  if (move.captured) {
    const piece = move.captured; // Tên quân cờ: p, n, b, r, q, k
    const colorOfMover = move.color; // 'w' (Trắng) hoặc 'b' (Đen)
    
    // Màu của quân bị bắt là NGƯỢC LẠI với màu của quân di chuyển
    const colorOfCaptured = colorOfMover === 'w' ? 'b' : 'w';
    
    // Tên file ảnh: bR, wP, v.v. (Màu quân bị bắt + Tên quân in hoa)
    const pieceImgName = 'w' + piece.toUpperCase(); 

    if (colorOfMover === 'w') {
      // Trắng ăn -> Đen bị bắt -> Thêm vào danh sách Trắng bắt (capturedWhite)
      capturedBlack.push(pieceImgName); 
    } else {
      // Đen ăn -> Trắng bị bắt -> Thêm vào danh sách Đen bắt (capturedBlack)
      capturedWhite.push(pieceImgName); 
    }
    renderCaptured();
  }
}

function renderCaptured() {
  const pieceImg = (piece) =>
    `<img src="./chessboardjs/img/chesspieces/wikipedia/${piece}.png" />`;

  // Xóa nội dung cũ trước khi render
  document.getElementById("capturedWhite").innerHTML = "";
  document.getElementById("capturedBlack").innerHTML = "";

  // Render các quân bị bắt bởi Trắng (quân Đen)
  capturedWhite.forEach(piece => {
    document.getElementById("capturedWhite").insertAdjacentHTML("beforeend", pieceImg(piece));
  });

  // Render các quân bị bắt bởi Đen (quân Trắng)
  capturedBlack.forEach(piece => {
    document.getElementById("capturedBlack").insertAdjacentHTML("beforeend", pieceImg(piece));
  });
}

// ==================== MOVE HISTORY =======================
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
      <td>${row.white}</td>
      <td>${row.whiteTime}</td>
      <td>${row.black}</td>
      <td>${row.blackTime}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ==================== STATUS =============================
function updateStatus() {
  let status = "";
  const moveColor = game.turn() === "w" ? "White" : "Black";

  if (game.in_checkmate()) {
    status = `Game over, ${moveColor === 'White' ? 'Black' : 'White'} wins by checkmate!`;
    clearInterval(timerInterval);
  } else if (game.in_draw()) {
    status = "Game over, drawn position.";
    clearInterval(timerInterval);
  } else {
    status = `${moveColor} to move`;
    if (game.in_check()) status += `, ${moveColor} is in check.`;
  }

  statusEl.textContent = status;
}

// ==================== BOARD INIT =========================
function init() {
  whiteClock = document.getElementById("whiteClock");
  blackClock = document.getElementById("blackClock");

  board = Chessboard("board", {
    draggable: true,
    position: "start",
    onDragStart,
    onDrop,
    onSnapEnd,
    moveSpeed: "slow",
    pieceTheme: "./chessboardjs/img/chesspieces/wikipedia/{piece}.png"
  });
  
  // Nút reset
  document.getElementById("resetBtn").addEventListener("click", () => {
    game.reset();
    board.start();
    capturedWhite = [];
    capturedBlack = [];
    moveHistory = [];

    document.getElementById("capturedWhite").innerHTML = "";
    document.getElementById("capturedBlack").innerHTML = "";

    whiteTime = 180;
    blackTime = 180;
    currentTurn = "w";
    updateClockDisplay();
    updateStatus(); 
    startClock();
    moveStartTime = Date.now();
    renderMoveHistory(); 
  });

  renderCaptured(); // Đảm bảo hiển thị ban đầu sạch sẽ
  updateClockDisplay();
  startClock();
}

document.addEventListener("DOMContentLoaded", init);
