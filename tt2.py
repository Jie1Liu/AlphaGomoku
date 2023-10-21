import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()

# 输入层
G.add_node('Input (4 x board_width x board_height)')

# 卷积层
G.add_node('Conv1 (3x3, 32 filters)')
G.add_node('ReLU')
G.add_node('Conv2 (3x3, 64 filters)')
G.add_node('ReLU')
G.add_node('Conv3 (3x3, 128 filters)')
G.add_node('ReLU')

# 策略头
G.add_node('PolicyHead-Conv1 (1x1, 4 filters)')
G.add_node('ReLU')
G.add_node('PolicyHead-FC1 (Linear, board_width x board_height)')
G.add_node('LogSoftmax')

# 价值头
G.add_node('ValueHead-Conv1 (1x1, 2 filters)')
G.add_node('ReLU')
G.add_node('ValueHead-FC1 (Linear, 64)')
G.add_node('ValueHead-FC2 (Linear, 1)')

# 连接
G.add_edge('Input (4 x board_width x board_height)', 'Conv1 (3x3, 32 filters)')
G.add_edge('Conv1 (3x3, 32 filters)', 'ReLU')
# ....连接其他层

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.savefig('neural_net.png')
plt.show()