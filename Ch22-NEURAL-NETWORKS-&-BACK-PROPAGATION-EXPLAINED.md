# NEURAL NETWORKS
## Neuron
1. Use activate function to mimic the Neuron mechnism.
    * composed by : weight(w), input(x), bias($\epsilon$)
    * kinds of activate function.
        * sigmoid : $y = \dfrac{1}{1+e^{-x}}$, and derivative of it is $y\prime = y(1-y)$
        * Relu : $y = max(0, x)$, and derivative of it is $if y>0, y\prime = 1 ; if y<0, y\prime = 0$
        * softmax : $y= \dfrac{e^{z_l}}{\sum_{k = 1}^{n} e^{z_k}}$, and derivative of it is $\dfrac{\partial}{\partial z_i}y_l = \begin{cases} y_l(1-y_l), if \space i = l \\ 
    -y_ly_i, if \space i \space \neq l\end{cases}$


# BACK PROPAGATION
1. Derivate the forward steps from end to the target point to update.

# Loss function
1. MSE
2. Cross-entropy 
3. KL divergence
