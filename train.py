from model import RNNModel

num_layers = 2
num_units = 256
num_feature = 28 # or 29? including blank label?

model = RNNModel(num_layers, num_units, num_feature, keep_prob=0.95, is_training=True)

model.build_graph()