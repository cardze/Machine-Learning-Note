# OPTIMIZATION AND GRADIENT DESCENT BASICS
## Simple concept about optimization
1. If we want to go to (6 , 2) by swimming and running from (0, 0). Coast is only on axis-x, so we need to swimming any way, please schedule a best distribution of these two ways to arrive there in shortest time.
![](./media/coast-to-island-problem.png)
## Lagrange multiplier
1. To solve a function $f(x, y, z)$ with constraint $g(x,y, z)=k$ 
2. Do parital derivative to $f = \lambda g$ for $x, y , z$ and solve it.
3. We get x, y, z from last step, put it into $g()$ to find $\lambda$
4. Finally, don't forget to check the $x, y,z$ are still in the constraint!  
### example
1. Maximize $𝑓(x, y) = 𝑥 + 𝑦$ subject to $𝑥^2 + 𝑦 ^2 = 1$
2. $L(𝑥, 𝑦, \lambda) = 𝑥 + 𝑦 + \lambda(𝑥^2 + 𝑦^2 − 1)$
3. Solve $\dfrac{\partial}{\partial x} L = 0$, $\dfrac{\partial}{\partial y} L = 0$ , $\dfrac{\partial}{\partial \lambda} L = 0$
# What if there are no constraint?
## Gradient Descent Method
1. $x_{k+1} = x_{k} - \eta \nabla f(x_k)$ where $\eta$  is a small positive number.
2. set a initial point $x_k$ to slowly approach to the answer.

## Regularization
1. [L1 L2 normalization](https://hackmd.io/@kk6333/BkIDyLikj)
    * L1 : $||w_1|| = \sum_{i} |w_i|$
    * L2 : $||w_2|| = \sum_{i} (w_i)^{2}$
2. Rewrite it to Lagrange multiplier
    * $L(w, \lambda) = Loss(w) + \lambda ||w_2||$ 

