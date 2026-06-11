# generate_data.py (Phiên bản Multiprocessing)

import chess
import chess.pgn
import chess.engine
import numpy as np
import os
import sys
import time
import traceback
import argparse
import multiprocessing # Thêm thư viện multiprocessing
from functools import partial # Dùng để truyền tham số cố định cho worker
from typing import List, Tuple, Optional, Dict, Any

# --- Import các thành phần cốt lõi từ dự án của bạn ---
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    from src.utils.replay_buffer import ReplayBuffer
    from src.game.chess_game import ChessGame
    from src.mcts.mcts_action_indexer import UCIActionIndexer
    from src.utils.adapter import build_action_maps
except ImportError as e:
    print(f"LỖI NGHIÊM TRỌNG: Không thể import các module cần thiết từ 'src'. Lỗi: {e}")
    sys.exit(1)

# --- Các hàm phụ trợ (stable_softmax, generate_policy_vector) ---
# (Giữ nguyên các hàm này như phiên bản trước)
_board_for_uci_generation = chess.Board()

def stable_softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    # ... (code stable_softmax giữ nguyên) ...
    if x.size == 0: return np.array([], dtype=np.float32)
    x = x / max(temp, 1e-6); x = x - np.max(x)
    e_x = np.exp(x); sum_ex = e_x.sum()
    if sum_ex == 0 or not np.isfinite(sum_ex): return np.ones_like(x, dtype=np.float32) / x.size
    return e_x / sum_ex

def generate_policy_vector(
    analysis_results: List[Dict[str, Any]],
    action_indexer: UCIActionIndexer, # Sẽ được truyền vào worker
    temperature: float,
    policy_size: int
) -> Tuple[Optional[np.ndarray], List[str]]:
    # ... (code generate_policy_vector giữ nguyên như phiên bản trước, sử dụng _board_for_uci_generation) ...
    policy_vector = np.zeros(policy_size, dtype=np.float32)
    moves_data = []; failed_ucis = []
    for info in analysis_results:
        move_obj: Optional[chess.Move] = info.get('pv', [None])[0]
        score_obj = info.get('score')
        if not isinstance(move_obj, chess.Move) or score_obj is None: continue
        cp_score = score_obj.relative.score(mate_score=10000)
        if cp_score is None: continue
        try:
            uci_str = _board_for_uci_generation.uci(move_obj)
            if not isinstance(uci_str, str) or not (4 <= len(uci_str) <= 5): continue
            moves_data.append({"uci": uci_str, "score_cp": float(cp_score)})
        except Exception: continue # Bỏ qua nếu lỗi tạo UCI
    if not moves_data: return policy_vector, failed_ucis
    moves_data.sort(key=lambda x: x["score_cp"], reverse=True)
    scores_cp_array = np.array([m["score_cp"] for m in moves_data])
    probabilities = stable_softmax(scores_cp_array, temp=temperature)
    for move_info, prob in zip(moves_data, probabilities):
        uci = move_info["uci"]
        try:
            idx = action_indexer.action_to_idx(uci) #
            if idx is None:
                print(f"[generate_policy_vector] Không tìm thấy idx cho UCI: {uci}")
                if uci not in failed_ucis: failed_ucis.append(uci)
            elif not (0 <= idx < policy_size):
                 if uci not in failed_ucis: failed_ucis.append(uci)
            else:
                policy_vector[idx] = prob
        except Exception:
             if uci not in failed_ucis: failed_ucis.append(uci)
    current_sum = policy_vector.sum()
    if current_sum > 1e-6 and not np.isclose(current_sum, 1.0):
        policy_vector /= current_sum
    return policy_vector, failed_ucis

# =================================================================
# HÀM WORKER CHO MULTIPROCESSING
# =================================================================
def process_game_chunk(
    game_indices: List[int], # <--- ĐƯA LÊN LÀM THAM SỐ ĐẦU TIÊN
    pgn_file_path: str,
    stockfish_path: str,
    action_indexer: UCIActionIndexer,
    args: argparse.Namespace
) -> List[Tuple]:
    """
    Hàm được chạy bởi mỗi tiến trình worker.
    Xử lý một danh sách các game PGN được chỉ định bởi game_indices.
    Trả về một danh sách các tuple (state, policy, value).
    """
    worker_pid = os.getpid()
    # print(f"[Worker {worker_pid}] Bắt đầu xử lý {len(game_indices)} games.") # Giảm log

    # --- Mỗi worker cần khởi tạo engine Stockfish riêng ---
    engine: Optional[chess.engine.SimpleEngine] = None
    try:
        engine_options = {"UCI_LimitStrength": "false", "Threads": 1}
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure(engine_options)
    except Exception as e:
        print(f"[Worker {worker_pid}] LỖI: Không thể khởi chạy Stockfish: {e}")
        return []

    game_wrapper = ChessGame() #
    all_generated_data = []
    processed_count = 0
    skipped_count = 0
    total_games = len(game_indices)
    try:
        # Mở file PGN (mỗi worker tự mở)
        # Sử dụng 'seek' để đến đúng vị trí game nếu PGN lớn và bạn có offset (phức tạp hơn)
        # Cách đơn giản: đọc tuần tự và bỏ qua game không cần thiết
        with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            game_indices_set = set(game_indices) # Chuyển sang set để kiểm tra nhanh hơn
            current_game_num = -1

            while True: # Đọc cho đến hết file hoặc hết game cần xử lý
                current_game_num += 1
                try:
                    game_pgn = chess.pgn.read_game(pgn_file)
                except Exception as read_err:
                    # print(f"[Worker {worker_pid}] Lỗi đọc game PGN {current_game_num}, bỏ qua: {read_err}")
                    continue # Bỏ qua game lỗi

                if game_pgn is None:
                    # print(f"[Worker {worker_pid}] Đã đọc hết file PGN.")
                    break # Hết file

                # Chỉ xử lý game nếu chỉ số của nó nằm trong danh sách được giao
                if current_game_num not in game_indices_set:
                    continue
                print(f"[Worker {worker_pid}] Đã xử lý {processed_count}/{total_games}, bỏ qua {skipped_count}, {((processed_count/total_games))*100}%...")
                # --- Logic xử lý game (giữ nguyên như trước) ---
                game_valid = True
                game_data_for_buffer: List[Tuple] = []
                try:
                    result_str = game_pgn.headers.get("Result", "*")
                    if result_str == "1-0": final_value = 1.0
                    elif result_str == "0-1": final_value = -1.0
                    elif result_str == "1/2-1/2": final_value = 0.0
                    else: game_valid = False

                    if game_valid:
                        game_wrapper.board = game_pgn.board() # Xử lý FEN
                        node = game_pgn
                        while node.variations:
                            next_node = node.variation(0)
                            move = next_node.move
                            current_board_state = game_wrapper.board.copy()

                            if move not in current_board_state.legal_moves:
                                game_valid = False; break

                            state_tensor = game_wrapper.encode_state() #
                            value = final_value * (1 if current_board_state.turn == chess.WHITE else -1)
                            limit = chess.engine.Limit(depth=args.depth, time=args.time_limit)
                            analysis = engine.analyse(current_board_state, limit, multipv=args.multi_pv)

                            policy_vector, _ = generate_policy_vector(
                                analysis, action_indexer, args.temperature, 4672
                            )
                            if policy_vector is not None and policy_vector.sum() > 1e-6:
                                game_data_for_buffer.append((state_tensor, policy_vector, float(value)))

                            game_wrapper.play_move(move) #
                            node = next_node

                except Exception as e_game:
                    game_valid = False
                    print(f"[Worker {worker_pid}] Lỗi xử lý game số {current_game_num}, bỏ qua: {e_game}")

                if game_valid and game_data_for_buffer:
                    all_generated_data.extend(game_data_for_buffer)
                    processed_count += 1
                else:
                    print(f"[Worker {worker_pid}] Bỏ qua game số {current_game_num} do không hợp lệ hoặc không có dữ liệu.")
                    skipped_count += 1

                # Đã xử lý xong game này, xóa khỏi set để dừng sớm nếu hết việc
                game_indices_set.remove(current_game_num)
                if not game_indices_set:
                    print(f"[Worker {worker_pid}] Đã xử lý hết game được giao.")
                    break # Dừng sớm

    except Exception as e_worker:
        print(f"[Worker {worker_pid}] LỖI NGHIÊM TRỌNG: {e_worker}")
    finally:
        if engine:
            engine.quit()

    # print(f"[Worker {worker_pid}] Hoàn thành. Thành công: {processed_count}, Bỏ qua: {skipped_count}. Vị trí: {len(all_generated_data)}")
    return all_generated_data

# --- Hàm Main đã sửa đổi để dùng Multiprocessing ---
def main(args):
    """Hàm chính điều khiển quá trình tạo dataset bằng multiprocessing."""

    # --- Kiểm tra file đầu vào ---
    if not os.path.exists(args.stockfish_path):
        print(f"LỖI: Không tìm thấy Stockfish tại: {args.stockfish_path}")
        return
    if not os.path.exists(args.pgn_path):
        print(f"LỖI: Không tìm thấy file PGN tại: {args.pgn_path}")
        return

    # --- Khởi tạo các thành phần dùng chung (chỉ 1 lần) ---
    print("Xây dựng bản đồ action indexer (chỉ 1 lần)...")
    try:
        build_action_maps() #
        action_indexer = UCIActionIndexer() #
        policy_size = len(getattr(action_indexer, 'all_actions', []))
        if policy_size != 4672:
             print(f"CẢNH BÁO: Kích thước action space là {policy_size} thay vì 4672.")
        print(f"Action indexer được tạo với policy size = {policy_size}")
    except Exception as e:
        print(f"Lỗi khi khởi tạo Action Indexer: {e}")
        return

    # --- Xác định công việc cho các worker ---
    print(f"Đang xác định các game cần xử lý (tối đa {args.num_games})...")
    # Đọc nhanh qua file PGN để đếm số game (cách đơn giản) hoặc lấy offsets (phức tạp hơn)
    # Cách đơn giản: Tạo danh sách chỉ số từ 0 đến num_games - 1
    all_game_indices = list(range(args.num_games))
    if not all_game_indices:
        print("Không có game nào để xử lý.")
        return

    # Chia danh sách chỉ số game cho các worker
    num_workers = min(args.workers, len(all_game_indices)) # Không dùng nhiều worker hơn số game
    if num_workers <= 0: num_workers = 1 # Ít nhất 1 worker
    chunk_size = len(all_game_indices) // num_workers
    worker_tasks = []
    start_idx = 0
    for i in range(num_workers):
        end_idx = start_idx + chunk_size
        # Worker cuối cùng lấy phần còn lại
        if i == num_workers - 1:
            end_idx = len(all_game_indices)
        worker_tasks.append(all_game_indices[start_idx:end_idx])
        start_idx = end_idx

    print(f"Sẽ sử dụng {num_workers} worker(s) để xử lý {len(all_game_indices)} games.")

    # --- Chạy các worker song song ---
    start_time = time.time()
    # Sử dụng partial để cố định các tham số không thay đổi cho worker
    worker_func = partial(
        process_game_chunk, # Vẫn là tên hàm
        # KHÔNG cần truyền game_indices ở đây
        pgn_file_path=args.pgn_path,
        stockfish_path=args.stockfish_path,
        action_indexer=action_indexer,
        args=args
    )

    all_results = []
    try:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            # imap_unordered sẽ truyền từng list trong worker_tasks
            # làm tham số đầu tiên (game_indices) cho worker_func
            results_iterator = pool.imap_unordered(worker_func, worker_tasks)
            # Thu thập kết quả
            for worker_result in results_iterator:
                all_results.extend(worker_result)
                print(f"\rĐã nhận {len(all_results)} vị trí từ các worker...", end="")

    except Exception as e_pool:
        print(f"\nLỖI NGHIÊM TRỌNG trong quá trình multiprocessing: {e_pool}")
        traceback.print_exc()
        return # Thoát nếu pool lỗi

    print("\nThu thập kết quả hoàn tất.")
    total_positions_added = len(all_results)
    elapsed = time.time() - start_time
    pos_per_sec = total_positions_added / elapsed if elapsed > 0 else 0

    # --- Tổng kết và Lưu Buffer ---
    print("\n--- TỔNG KẾT ---")
    print(f"Tổng thời gian xử lý: {elapsed:.2f} giây.")
    print(f"Tổng số thế cờ hợp lệ được thêm vào: {total_positions_added}")
    print(f"Tốc độ trung bình: {pos_per_sec:.0f} pos/giây.")
    # (Có thể thêm lại phần thống kê lỗi UCI nếu cần, nhưng giờ lỗi đã ít hơn)

    if total_positions_added > 0:
        print(f"Đang tạo và lưu buffer vào {args.output_path}...")
        replay_buffer = ReplayBuffer(max_size=args.buffer_size) #
        replay_buffer.extend(all_results) # Thêm tất cả vào buffer
        try:
            replay_buffer.save(args.output_path) #
            print(f"Lưu buffer hoàn tất. Kích thước buffer: {len(replay_buffer)}")
        except Exception as save_err:
            print(f"LỖI NGHIÊM TRỌNG KHI LƯU BUFFER: {save_err}")
    else:
        print("Không có dữ liệu nào được tạo, không lưu buffer.")

    print("Tạo dataset hoàn tất.")

# --- Parser và gọi main ---
if __name__ == "__main__":
    # --- Cấu hình mặc định ---
    # --- CẤU HÌNH CỦA BẠN ---

    # CẬP NHẬT ĐƯỜNG DẪN TỚI FILE THỰC THI STOCKFISH CỦA BẠN
    STOCKFISH_PATH = r"stockfish/stockfish-windows-x86-64-avx2.exe"
    # CẬP NHẬT ĐƯỜNG DẪN TỚI FILE PGN LỚN BẠN ĐÃ TẢI VỀ
    PGN_FILE_PATH = r"GameDatabase/lichess_elite_2022-02/lichess_elite_2022-02.pgn"
    # Nơi lưu trữ dataset đầu ra
    OUTPUT_BUFFER_PATH = "Dataset/stockfish_dataset_gen0.pkl.gz"
    # Số lượng ván cờ cần xử lý từ file PGN
    GAMES_TO_PROCESS = 5000
    # Độ sâu Stockfish phân tích
    STOCKFISH_DEPTH = 12
    # Số lượng nước đi hàng đầu cần lấy (Multi-PV)
    MULTI_PV = 5
    # Nhiệt độ cho hàm softmax
    POLICY_TEMPERATURE = 100.0

    parser = argparse.ArgumentParser(description="Tạo bộ dữ liệu bootstrapping từ PGN và Stockfish bằng Multiprocessing.")
    parser.add_argument("--stockfish_path", type=str, default=STOCKFISH_PATH)
    parser.add_argument("--pgn_path", type=str, default=PGN_FILE_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_BUFFER_PATH)
    parser.add_argument("--num_games", type=int, default=GAMES_TO_PROCESS)
    parser.add_argument("--depth", type=int, default=STOCKFISH_DEPTH)
    parser.add_argument("--time_limit", type=float, default=0.1)
    parser.add_argument("--multi_pv", type=int, default=MULTI_PV)
    parser.add_argument("--temperature", type=float, default=POLICY_TEMPERATURE)
    parser.add_argument("--buffer_size", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2), help="Số lượng tiến trình worker song song.") # Mặc định dùng ít hơn 2 core so với tổng số core
    parsed_args = parser.parse_args()

    # !!! QUAN TRỌNG: Đảm bảo build_action_maps chạy trước khi fork process !!!
    # Gọi build_action_maps ở đây để các bản đồ toàn cục được tạo sẵn
    print("Khởi tạo Action Maps một lần...")
    try:
        build_action_maps() #
    except Exception as e_map:
        print(f"LỖI NGHIÊM TRỌNG khi tạo Action Maps: {e_map}")
        sys.exit(1)

    main(parsed_args)