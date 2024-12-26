# gpu-breadth-first-search
Implementation of Breadth First Search algorithm for graph traversal on GPU architecture using CUDA


# How it works

## Algorithm breakdown
Given the Breadth First Search problem, the basic idea is to build a *frontier* of nodes currently visited, that separates the already visited nodes from the ones that haven't been visited yet. At each iteration, one node from the frontier is explored and its neighbours (if not already visited) are added to the frontier. The algorithm terminates when a frontier with 0 elements inside is reached.

<p align="center">
![Frame 14temp_network](https://github.com/user-attachments/assets/714d5b3a-4807-4d9a-b98f-080059c5759d)
![Frame 9temp_network](https://github.com/user-attachments/assets/e288f723-e64d-431f-8457-9467de4cb754)
![Frame 10temp_network](https://github.com/user-attachments/assets/f63dd26c-390a-4cd0-b427-546a7848f4e0)
</p>



## GPU parallelism
By means of a GPU it is possible to speed up the process of exploring nodes in a frontier. In particular, we might associate each GPU thread with a node in the frontier and make them compute the associated node neighbours all at once. After each thread have finished its execution, the next frontier is built using the neighbours found.

![Frame 19temp_network](https://github.com/user-attachments/assets/e93f35c7-5a70-4729-8fb4-e64e211562ef)
![Frame 22temp_network](https://github.com/user-attachments/assets/f5beb511-bb6d-4701-8383-3f6571fe6801)
![Frame 24temp_network](https://github.com/user-attachments/assets/89a93734-f0b0-4aef-b5e2-6679eb161006)


