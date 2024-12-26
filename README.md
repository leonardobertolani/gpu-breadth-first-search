# gpu-breadth-first-search
Implementation of Breadth First Search algorithm for graph traversal on GPU architecture using CUDA


# How it works

## Algorithm breakdown
Given the Breadth First Search problem, the basic idea is to build a *frontier* of nodes currently visited, that separates the already visited nodes from the ones that haven't been visited yet. At each iteration, one node from the frontier is explored and its neighbours (if not already visited) are added to the frontier. The algorithm terminates when a frontier with 0 elements inside is reached.

## GPU parallelism
By means of a GPU it is possible to speed up the process of exploring nodes in a frontier. In particular, we might associate each GPU thread with a node in the frontier and make them compute the associated node neighbours all at once. After each thread have finished its execution, the next frontier is built using the neighbours found.
