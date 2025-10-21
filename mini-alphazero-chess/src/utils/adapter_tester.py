from src.mcts.mcts_action_indexer import UCIActionIndexer
def test_action_indexer():
    """
    Test UCIActionIndexer for consistency between action_to_idx and idx_to_action.
    """
    indexer = UCIActionIndexer()

    # Test that every action maps to a valid index and back
    for idx, action in enumerate(indexer.all_actions):
        mapped_idx = indexer.action_to_idx(action)
        assert mapped_idx == idx, f"Action to idx failed for action {action}: expected {idx}, got {mapped_idx}"

        mapped_action = indexer.idx_to_action(idx)
        if isinstance(mapped_action, str):
            assert mapped_action == action, f"Idx to action failed for idx {idx}: expected {action}, got {mapped_action}"
        else:
            assert mapped_action.uci() == action, f"Idx to action failed for idx {idx}: expected {action}, got {mapped_action.uci()}"

    print("All UCIActionIndexer tests passed.")
if __name__ == "__main__":
    test_action_indexer()