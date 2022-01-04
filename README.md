# DEQ-Mnist
[Deep Equilibrium model](https://arxiv.org/abs/1909.01377) is a kind of implicit layer model. These layers(implicit layers) have shown impressive results on NLP and vision tasks. One of the advantages of the implicit layer is memory efficiency which is based on implicit differentiation.

This graph shows the memories that were used in our DEQ model when we didn't implement implicit differentiation:


![alt text](https://raw.githubusercontent.com/manishemirani/DEQ-Mnist/main/graphs/without_implicit_differentiation.png)


This graph shows the memories when we were implemented implicit differentiation:



![alt text](https://raw.githubusercontent.com/manishemirani/DEQ-Mnist/main/graphs/with_implicit_differentiation.png)

Since these models are based on implicit layers we need a fixed point solver to find the fixed point that satisfies the desired condition. For fixed-point solvers, I used [Anderson acceleration](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf) and forward solver which is a forward pass layer that satisfies a condition.
