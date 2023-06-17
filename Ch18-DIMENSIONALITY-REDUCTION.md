# DIMENSIONALITY REDUCTION
## PCA(Principal components analysis)
1. The idea is simple: We want to find an “axis” $w_1$, when projected on it, the sum of squares of $x_i$ projections are maximum.
2. Λ = diag(𝜆1, ⋯ , 𝜆𝑝) is a diagonal matrix consisting
of eigenvalues of $X^TX$ where X is the data.

### Score of reducing the N dimension to k
1. PoV(Proportion of Variance) $$PoV(k) = \dfrac{\lambda_1 + ... + \lambda_k}{\lambda_1 + ... + \lambda_p}$$
2. Typically pick k where PoV(k) > 0.9, where the number can be decided by the user(?).

## FA(Factor analysis)
1. See if the observed variables are effecting the hiden factor(latent factor)
2. First we need to set our goal => How many latent we should choose.
3. Then we have to divide the factors into groups by checking their covarience to each others.
4. Use pseudo inverse to do the transformation.

## ICA(Independent components analysis)
1. For distinguish the source from Mixed Data => Blind Source Separation.
2. ICA will choose the base vector which more fit on the data distribution.
![](./media/ICA_vs_PCA.png)

## LDA(Linear discriminant analysis)
1. If we are dealing with the data which has k classes, the max number of dimension after LDA is (k-1). => because we only need k-1 lines devide the data into k classes.