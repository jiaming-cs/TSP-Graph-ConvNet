import numpy as np


class PartialRefinement:
    def __init__(self):
        return None
        
    def greedy_search(self, probs):
        
        problem_size = probs.shape[0]
        greedy_solution = [0]
        while len(greedy_solution) != problem_size:
            for i in np.argsort(-probs[greedy_solution[-1]]):
                if i not in greedy_solution:
                    greedy_solution.append(i)
                    break
        
        y_edges = np.zeros(probs.shape)
        
        for i in range(problem_size):
            y_edges[greedy_solution[i]][greedy_solution[(i+1)%problem_size]] = 1

        

        
        
        return greedy_solution, y_edges
    
    
    def find_confused_parts(self, probs, greedy_solution, threshold=0.3):
        y_edges_bin = probs.copy()
        
        for i in range(y_edges_bin.shape[0]):
            for j in range(y_edges_bin.shape[1]):
                if y_edges_bin[i, j] < threshold:
                    y_edges_bin[i, j] = 0
                else:
                    y_edges_bin[i, j] = 1
                    
        confused_nodes_index = [i for i, vec in enumerate(y_edges_bin) if sum(vec)>2]
        
        
        
        confused_nodes = [i if i in confused_nodes_index else -1 for i in greedy_solution]
        
        if confused_nodes[0] != -1:
            i = len(confused_nodes) - 1
            while confused_nodes[i] != -1:
                i -= 1
            confused_nodes = confused_nodes[i:] + confused_nodes[:i]
        confused_nodes += [-1]
        start, end = [], []
        
        for i, n in enumerate(confused_nodes):
            if n != -1 and confused_nodes[i-1] == -1:
                start.append(i)
                
            if n != -1 and confused_nodes[i+1] == -1:
                end.append(i)
                
        return [confused_nodes[s: e+1] for s, e in zip(start, end)]            
        
        
    
    
    
    
    
    
    
    
