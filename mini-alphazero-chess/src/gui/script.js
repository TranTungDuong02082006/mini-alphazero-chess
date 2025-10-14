// Khởi tạo bàn cờ và trò chơi
let board = null;
let game = new Chess();
let capturedWhite = [];
let capturedBlack = [];

const statusEl = document.getElementById("status");
const API_URL = "http://127.0.0.1:8000/move";

// Kiểm tra điều kiện bắt đầu kéo quân
function onDragStart(source, piece) {
  if (game.game_over()) return false;
  if (game.turn() !== "w" || piece.startsWith("b")) return false;
}

// Xử lý khi người chơi thả quân
async function onDrop(source, target) {
  const move = game.move({ from: source, to: target, promotion: "q" });

  if (!move) return "snapback";

  updateCapturedPieces(move);
  board.position(game.fen());
  updateStatus();

  await requestBotMove();
}

// Cập nhật vị trí sau khi quân được thả
function onSnapEnd() {
  board.position(game.fen());
}

// Gửi FEN cho bot và xử lý phản hồi
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
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.status === "ok" && data.move) {
      const botMove = game.move(data.move, { sloppy: true });
      updateCapturedPieces(botMove);
      board.position(game.fen());
      updateStatus();
    } else {
      const reason = data.reason || "unknown";
      statusEl.textContent = data.status === "game_over"
        ? `Game over: ${reason}`
        : `Error: ${reason}`;
      console.error(data);
    }
  } catch (error) {
    console.error("API fetch failed:", error);
    statusEl.textContent = "Cannot reach API server";
  }
}

// Cập nhật trạng thái trò chơi
function updateStatus() {
  let status = "";
  const moveColor = game.turn() === "w" ? "White" : "Black";

  if (game.in_checkmate()) {
    status = `Game over, ${moveColor} is checkmated.`;
  } else if (game.in_draw()) {
    status = "Game over, drawn position.";
  } else {
    status = `${moveColor} to move`;
    if (game.in_check()) {
      status += `, ${moveColor} is in check.`;
    }
  }

  statusEl.textContent = status;
}

function updateCapturedPieces(move) {
  if (move.captured) {
    const pieceSymbol = move.captured;
    const capturedPiece = move.color === "w" ? "b" + pieceSymbol : "w" + pieceSymbol;

    if (move.color === "w") {
      capturedBlack.push(capturedPiece);
    } else {
      capturedWhite.push(capturedPiece);
    }

    renderCaptured();
  }
}

function renderCaptured() {
  const pieceImg = (piece) =>
    `<img src="./chessboardjs/img/chesspieces/wikipedia/${piece}.png";" />`;

  // Chỉ thêm quân mới vào DOM
  if (capturedWhite.length > 0) {
    const lastWhite = capturedWhite[capturedWhite.length - 1];
    document.getElementById("capturedWhite").insertAdjacentHTML("beforeend", pieceImg(lastWhite));
  }

  if (capturedBlack.length > 0) {
    const lastBlack = capturedBlack[capturedBlack.length - 1];
    document.getElementById("capturedBlack").insertAdjacentHTML("beforeend", pieceImg(lastBlack));
  }
}
// Tạo bàn cờ
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
  renderCaptured();
  updateStatus();
  document.getElementById("capturedWhite").innerHTML = "";
  document.getElementById("capturedBlack").innerHTML = "";
});

// Khởi tạo trạng thái ban đầu
updateStatus();