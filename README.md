# Idea for FMPINN
Solving a class of elliptic partial differential equations(PDEs) with multiple scales utilizing Fourier-based mixed physics informed neural networks(dubbed FMPINN), the solver of FMPINN is configured as a multi-scale deep neural networks.

# Title of paper
Solving a class of multi-scale elliptic PDEs by Fourier-based mixed physics informed neural networks

created by Xi'an Li, Jinran Wu, Xin Tai, Jianhua Xu and You-Gan Wang

[[Paper]](https://www.researchgate.net/publication/371855350_Solving_a_class_of_multi-scale_elliptic_PDEs_by_Fourier-based_mixed_physics_informed_neural_networks)

# Abstract

Deep neural networks have garnered widespread attention due to their simplicity and flexibility in the fields of engineering and scientific calculation. In this study, we probe into solving a class of elliptic partial differential equations(PDEs) with multiple scales by utilizing Fourier-based mixed physics informed neural networks(dubbed FMPINN), its solver is configured as a multi-scale deep neural network. In contrast to the classical PINN method, a dual (flux) variable about the rough coefficient of PDEs is introduced to avoid the ill-condition of neural tangent kernel matrix caused by the oscillating coefficient of multi-scale PDEs. Therefore, apart from the physical conservation laws, the discrepancy between the auxiliary variables and the gradients of multi-scale coefficients is incorporated into the cost function, then obtaining a satisfactory solution of PDEs by minimizing the defined loss through some optimization methods. Additionally, a trigonometric activation function is introduced for FMPINN, which is suited for representing the derivatives of complex target functions. Handling the input data by Fourier feature mapping will effectively improve the capacity of deep neural networks to solve high-frequency problems.  Finally, to validate the efficiency and robustness of the proposed FMPINN algorithm, we present several numerical examples of multi-scale problems in various dimensional Euclidean spaces. These examples cover both low-frequency and high-frequency oscillation cases, demonstrating the effectiveness of our approach.

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

# Remarks
1. The all networks in DNN_base are modified for initializing their corresponding Weights and Biases, and 
   configuring the type of float and the device(cpu or gpu) when initializing the network. Unfortually, the 
   performance of Fourier_Subnets3D will be decrease for solving the elliptic multi-scale PDEs when eps=0.001.
   This is a open problem, it is necessary to study.
2. The codes for various networks in DNN_base_old, FMPINN1d using these codes to solve multi-scale PDEs can 
   obtain wonderful results. The difference of codes in DNN_base_old and DNN_base are the initialization of 
   networks with device.
