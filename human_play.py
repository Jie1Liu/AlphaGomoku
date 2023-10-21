from __future__ import print_function
import pickle
from game2 import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from policy_value_net_pytorch import PolicyValueNet  # Pytorch

def run():
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5.model2'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')
        best_policy = PolicyValueNetNumpy(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)

        game.start_play( mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')
if __name__ == '__main__':
    run()
