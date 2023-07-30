# Idea for FMPINN
Solving a class of elliptic partial differential equations(PDEs) with multiple scales utilizing Fourier-based mixed physics informed neural networks(dubbed FMPINN), the solver of FMPINN is configured as a multi-scale deep neural networks.

# Title of paper
Solving a class of multi-scale elliptic PDEs by means of Fourier-based mixed physics informed neural networks

created by Xi'an Li, Jinran Wu, You-Gan Wang, Xin Tai and Jianhua Xu

[[Paper]](https://arxiv.org/pdf/2306.13385.pdf)

# Abstract

Deep neural networks have received widespread attention due to their simplicity and flexibility in the fields of engineering and scientific calculation. In this work, we probe into solving a class of elliptic partial differential equations(PDEs) with multiple scales utilizing Fourier-based mixed physics informed neural networks(dubbed FMPINN), the solver of FMPINN is configured as a multi-scale deep neural networks. Unlike the classical PINN method, a dual (flux) variable about the rough coefficient of PDEs is introduced to avoid the ill-condition of neural tangent kernel matrix that resulted from the oscillating coefficient of multi-scale PDEs. Therefore, apart from the physical conservation laws, the discrepancy between the auxiliary variables and the gradients of multi-scale coefficients is incorporated into the cost function, then obtaining a satisfactory solution of PDEs by minimizing the defined loss through some optimization methods. Additionally, a trigonometric activation function is introduced for FMPINN, which is suited for representing the derivatives of complex target functions. Handling the input data by Fourier feature mapping will effectively improve the capacity of deep neural networks to solve high-frequency problems.  Finally, by introducing several numerical examples of multi-scale problems in various dimensional Euclidean spaces, we validate the efficiency and robustness of the proposed FMPINN algorithm in both low-frequency and high-frequency oscillation cases.

# Remarks

## 1.The MSPDE2d_FMPINN_TensorForm is faster than MSPDE2d_FMPINN_TensorForm_X_Y, but the performance of MSPDE2d_FMPINN_TensorForm_X_Y is superior to that of MSPDE2d_FMPINN_TensorForm

## 2.The LHS sampler is better than Random sampler for interior and boundary.

10 ---  0.00751(LHS)

10 ---- 0.00801(Random)

5----0.00651(LHS)

5----0.00701(Random)

## 3.For 2D problem, changing the penalty of loss for flux variable in MSPDE2d_FMPINN_TensorForm_X_Y, 

10 --- 0.00351

5----0.00651

20----0.0115
