import numpy as np
from shapley import PermutationSampler
from ensambler import retrieve_weights

# W = np.array([[0.99, 0.01, 0.29843549, 0.15379455, 0.51862131, 0.34891333,
#   0.1075523  ,0.77481699 ,0.27726541 ,0.88820124 ,0.43517843 ,0.49584311,
#   0.97304657 ,0.54722007 ,0.87338943 ,0.37438674 ,0.15430086 ,0.90116497,
#   0.90280463 ,0.21883294 ,0.27128867 ,0.88007508 ,0.72142279 ,0.20474932,
#   0.58962011 ,0.18098523 ,0.10759176 ,0.19045006 ,0.77667651 ,0.50374377,
#   0.14314454 ,0.28396882 ,0.72998475 ,0.86541802 ,0.69480103 ,0.82359642,
#   0.34568706 ,0.59626615 ,0.45889262 ,0.47196615 ,0.18462066 ,0.27060618,
#   0.14513884 ,0.97889627 ,0.57690177 ,0.86666917 ,0.81444338 ,0.76350051,
#   0.84789274 ,0.97741937 ,0.68141679 ,0.48756238 ,0.57635841 ,0.69576718,
#   0.45225732 ,0.76294495 ,0.98842759 ,0.16288184 ,0.58601026 ,0.28860111,
#   0.30734203 ,0.64452474 ,0.56254204 ,0.65260114 ,0.63892126 ,0.13173215,
#   0.41435691 ,0.41736315 ,0.13721167 ,0.46936669]])


weights = retrieve_weights()
W = np.array([weights[0]])
W = W/W.sum()
# print("size:",W.shape)
# print("sommatoria:", W.sum())
q = 0.5
solver = PermutationSampler()

print("quote ", q)
solver.solve_game(W,q)
shapley_values = solver.get_solution()
# avg_shapley = solver.get_average_shapley()
# print(avg_shapley)
print("shapley values:", shapley_values)
