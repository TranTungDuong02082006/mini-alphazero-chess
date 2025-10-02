import chess

def render_board_from_fen(fen_string: str):
    """
    Hàm này nhận một chuỗi FEN và hiển thị trạng thái bàn cờ tương ứng ra console.

    Args:
        fen_string (str): Chuỗi FEN biểu diễn một thế cờ.
    
    Returns:
        None: Hàm này chỉ in ra bàn cờ và không trả về giá trị nào.
    
    Raises:
        ValueError: Ném ra lỗi nếu chuỗi FEN không hợp lệ.
    """
    try:
        # 1. Tạo một đối tượng Board từ chuỗi FEN
        board = chess.Board(fen_string)
        
        # 2. In đối tượng board ra console. 
        # Thư viện chess đã định nghĩa sẵn cách hiển thị bàn cờ một cách đẹp mắt.
        print("+" + "---+" * 8)
        print(board)
        print("+" + "---+" * 8)
        
        # 3. In thêm một số thông tin hữu ích từ FEN
        print(f"Lượt đi của: {'Trắng (White)' if board.turn == chess.WHITE else 'Đen (Black)'}")
        print(f"Quyền nhập thành: {board.castling_xfen()}")
        
    except ValueError:
        print(f"Lỗi: Chuỗi FEN '{fen_string}' không hợp lệ.")

# --- Ví dụ sử dụng ---

if __name__ == "__main__":
    # Ví dụ 1: Thế cờ bắt đầu mặc định
    print("--- Ví dụ 1: Thế cờ bắt đầu ---")
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    render_board_from_fen(start_fen)
    print("\n" + "="*30 + "\n")

    # Ví dụ 2: Một thế cờ sau vài nước đi (e4 c5)
    print("--- Ví dụ 2: Sau nước 1. e4 c5 ---")
    sicilian_defense_fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    render_board_from_fen(sicilian_defense_fen)
    print("\n" + "="*30 + "\n")
    
    # Ví dụ 3: Một chuỗi FEN không hợp lệ để kiểm tra lỗi
    print("--- Ví dụ 3: FEN không hợp lệ ---")
    invalid_fen = "this is not a valid fen"
    render_board_from_fen(invalid_fen)