def measure_strength(G, node_i, node_j):
    list1 = set(G.neighbors(node_i))
    list1.add(node_i)
    list2 = set(G.neighbors(node_j))
    list2.add(node_j)
    strength =  len(list1 & list2) / ((len(list1)  * len(list2)) ** 0.5)
    return strength

