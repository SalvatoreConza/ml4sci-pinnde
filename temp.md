# Project: A PINN Literature Based Approach for the Damped Harmonic Oscillator
**Author:** Conza Salvatore Angelo

## Abstract
In this notebook, I implement a **Physics-Informed Neural Network (PINN)** to solve the parametric damped harmonic oscillator equation:
$$
\frac{d^2x}{dz^2} + 2\xi \frac{dx}{dz} + x = 0
$$
The network approximates the solution $x(z, \xi)$ over the domain $z \in [0, 20]$ for damping ratios $\xi \in [0.1, 0.4]$. The network approximates the solution $x(z, \xi)$ by minimizing a loss function composed of the ODE residual and initial condition constraints ($x_0=0.7, v_0=1.2$).
It



# Methodology
## Experiment 1
* **Not so trivial Observation/Premise**: Since what I am plotting is basically a function that oscillating more or less towards the zero and is a lot smooth I can imagine that the model will get less precise in the points where the derivative change its sign, and this was the case for all the experiments: the graphs where I plotted the absolute error over time (so the |exact analytic solution - PINN solution|) show a bigger absolute error in the points where the function change the sign of its derivative. This also explain why a damped harmonic oscillator with bigger xi would get predicted better from my PINN: there are less points where the function changes its derivative so its easier to predict. This in literature is called "spectral bias" or "failure modes". A way of solving this is to sample fresh random points at each steps, this would help us to "fill" better the domain, this works but it will add some gradient noise.
  
* **Architecture:** Feedforward fully connected network (Input: 2, Hidden: 3x50, Output: 1). This from a literature review seemed a good start, I quote (and highlight the most important part):
> Since Raissi et al (2019) original vanilla PINN, **the majority of solutions have used feed-forward neural networks.**
> 
> — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.1, Page 11*

>While the ideal DNN architecture is still an ongoing research; papers implementing PINN have attempted to empirically optimise the architecture’s
characteristics, such as the number of layers and neurons in each layer. Smaller
DNNs may be unable to effectively approximate unknown functions, whereas
too large DNNs may be difficult to train, particularly with small datasets.
Raissi et al (2019) used different typologies of DNN, for each problem, like a
12 Scientific Machine Learning through PINNs
5-layer deep neural network with 100 neurons per layer, an DNN with 4 hidden layers and 200 neurons per layer or a 9 layers with 20 neuron each layer.
**Tartakovsky et al (2020) empirically determine the feedforward network size, in particular they use three hidden layers and 50 units per layer, all with an hyperbolic tangent activation function.**
> —*S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.1.1, Page 11 & 12*
* **Activation:** I chosed Tanh, this because of what I quoted before but, also, I quote:
    > Most authors tend to use the infinitely differentiable hyperbolic tangent activation function $\alpha(x) = \tanh(x)$.
    >
    > — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.1.1, Page 13*
* **Loss Function:** Physics Residual + Initial Conditions for velocity and position. This method of simpling adding those two losses is kinda bad, I quote:
  > BC [Boundary Condition] constraints can be regarded as penalty terms (soft BC enforcement) (Zhu et al, 2019), or they can be encoded into the network design (hard BC enforcement) (Sun et al, 2020a).
    >
    > Many existing PINN frameworks use a soft approach to constrain the BCs by creating extra loss components defined on the collocation points of borders. The disadvantages of this technique are twofold:
    > 1.  Satisfying the BCs accurately is not guaranteed;
    > 2.  The assigned weight of BC loss might effect learning efficiency, and no theory exists to guide determining the weights at this time.
    >
    > Zhu et al (2021) address the Dirichlet BC in a hard approach by employing a specific component of the neural network to purely meet the specified Dirichlet BC. Therefore, the initial boundary conditions are regarded as part of the labeled data constraint.
    >
    > — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.3, Page 23*
* **Optimizer:** Adam Optimizer, this should work better than stochastic gradient for random distribution of points. I quote:
>The Adam approach, which combines adaptive learning rate and momentum methods, is employed in Zhu et al (2021) to increase convergence speed,
>because stochastic gradient descent (SGD) hardly manages random collocation
>points, especially in 3D setup.
>
> — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.3, Page 24*

but this method Adam Optimizer alone doesnt seems the best thing to do. Spedicically, I quote
> He et al (2020) propose a two-step training approach in which the loss function is minimized first by the Adam algorithm with a predefined stop condition, then by the L-BFGS-B optimizer. According to the aforementioned paper, for cases with a little amount of training data and/or residual points, L-BFGS-B, performs better with a faster rate of convergence and reduced computing cost.
> 
>  — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.3, Page 24*

* **Pre processing:** We are using a neural networks with only tanh as activation functions so the normalization is highly suggested, in fact if the inputs and weights are big the tanh would go to regions where its constant (and equal to 1 or -1) and its gradient is constant, and this is bad.
* **Stopping criteria:** A simple patience based early stopping will work poorly, because we are sampling the points randomically, we decide to NOT compare the current loss with the best loss. We decided to calculate a average loss on a windows of 100 epochs and to compare this new loss to the best loss.   

* **Result comparison:** plots of the neural network solution and the analytic function, loss function and absolute error over time (|exact analytic solution - PINN solution|) will be given to make fast comments on the results. In addition, three table with metric results is given for make more rigorous the comparison beetween experiments for three chosed xi as reference (0.1, 0.25 and 0.4). 




```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# define the PINN architecture model and forward pass
# input 2 features, z and xi and ouput 1 feature x

class OscillatorPINN(nn.Module):
    def __init__(self):
        super(OscillatorPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
    def forward(self, z, xi):
        # normalization step
        z_norm = 2.0 * (z - 0.0) / (20.0 - 0.0) - 1.0
        xi_norm = 2.0 * (xi - 0.1) / (0.4 - 0.1) - 1.0
        # I feed the nromalized values into the layers
        inputs = torch.cat([z_norm, xi_norm], dim=1)
        return self.net(inputs)

# define the physics informed loss function 
def physics_loss(model, z, xi):
    
    # enable gradient tracking for z
    z.requires_grad = True
    
    # predict x
    x = model(z, xi)
    
    # compute first derivative (dx/dz)
    dx_dz = torch.autograd.grad(
        x, z, 
        grad_outputs=torch.ones_like(x), 
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # compute second derivative (d^2x/dz^2)
    d2x_dz2 = torch.autograd.grad(
        dx_dz, z, 
        grad_outputs=torch.ones_like(dx_dz), 
        create_graph=True, 
        retain_graph=True
    )[0]
                                            # this is just the left hand side of the PDE specified in the
    residual = d2x_dz2 + 2 * xi * dx_dz + x # test on the ML4SC website and, since is = 0, the residual
                                            # just tell me how much off the prediction x is from the 0.

                                    # the mean of one positive and one negative 
    return torch.mean(residual**2)  # value of the same module is 0, so it's
                                    # required to square to account them as penalty


# training loop 
def train_pinn(model, steps=100000, lr=1e-3, patience=2000, min_delta=1e-6):
    # choosed optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # initial conditions targets (specified in the test on the ML4SC website)
    x0_target = 0.7
    v0_target = 1.2
   
    # windows for the average loss calculation
    window = 100
    loss_window = []

    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    print("starting training...")
    
    for step in range(steps):
        optimizer.zero_grad()
         
        # sample random points for z and xi in given domains (specified in the test on the ML4SC website) at each step
        # this help us to cover the domain well over time
        batch_size = 5000
        z_col = torch.rand(batch_size, 1) * 20.0
        xi_col = torch.rand(batch_size, 1) * (0.4 - 0.1) + 0.1
        # at each of this random points we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col, xi_col)
        
        # initial condition loss (boundary loss)
        # Sample random xi for the boundary condition
        xi_bc = torch.rand(batch_size // 4, 1) * (0.4 - 0.1) + 0.1 # we pick a random xi again, no matter what xi I took, the oscillator always start in the same spot
        z_bc = torch.zeros_like(xi_bc) # z is always 0 here
        z_bc.requires_grad = True # we still need to track z if I want to track how much fast x change with z, so dx/dz

        # guess of the position at t=0:
        x_bc = model(z_bc, xi_bc)
        
        # then compute velocity at t=0:
        dx_bc = torch.autograd.grad(
            x_bc, z_bc,
            grad_outputs=torch.ones_like(x_bc),
            create_graph=True,
            retain_graph=True
        )[0]
        # then calculate the penalty for position and velocity:
        loss_ic_x = torch.mean((x_bc - x0_target)**2)
        loss_ic_v = torch.mean((dx_bc - v0_target)**2)
        
                                                    # total loss, infact without the penalty for position and velocity
        loss = loss_physics + loss_ic_x + loss_ic_v # I would have that x=0 and v=0 for every time step minimize the loss and  
                                                    # so is the solution but that cannot be since of ICs different from zero
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        loss_history.append(loss.item())

        # early stopping criteria: if the average loss doesnt improve at least of min_delta, for a number of consecutive steps equal to the patience, it stops early.
        loss_window.append(current_loss)
        if len(loss_window) > window:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
    
        if patience_counter >= patience:
            print(f"\nEarly stopping at step {step}")
            print(f"Best loss: {best_loss:.6f}, Current loss: {current_loss:.6f}")
            break
    
        if step % 1000 == 0:
            print(f"Step {step}: Loss = {current_loss:.6f}, Avg Loss of last 100 steps = {avg_loss:.6f} (Best: {best_loss:.6f}, Patience: {patience_counter}/{patience})")
    return loss_history

# validation & visualization 
def analytical_solution(z, xi, x0=0.7, v0=1.2):
    omega_n = 1.0 # from wikipedia page  on Harmonic oscillator, this is the natural frequency, this is just the coeff of x in the PDE so it's 1
    omega_d = np.sqrt(omega_n**2 - xi**2) # this is the damped natural frequency in the case of omega_n=1
    
    # integrations constants A and B to be founded with ICs, since the solution from  
    # wikipedia is x(z)=exp(-xi*omega_n*z)*(A*cos(omega_d*t)+B*sin(omega_d*z)) I would have:
    # x(0) = A = x0
    # x'(0) = -xi*omega_n*A + omega_d*B = v0
    A = x0
    B = (v0 + xi * omega_n * x0) / omega_d
    
    return np.exp(-xi * omega_n * z) * (A * np.cos(omega_d * z) + B * np.sin(omega_d * z))

# plotting results for specific xi values and testing with xi = 0.1 (low damping), xi=0.25 (medium damping), xi = 0.4 (high damping)
def plot_predictions(model, model_name="PINN", test_xis=[0.1, 0.25, 0.4]): 
    z_test = torch.linspace(0, 20, 200).view(-1, 1)
    plt.figure(figsize=(12, 5))
    for i, xi_val in enumerate(test_xis):
        xi_test = torch.full_like(z_test, xi_val)
        with torch.no_grad():
            x_pred = model(z_test, xi_test).numpy()
        x_exact = analytical_solution(z_test.numpy(), xi_val)
        plt.subplot(1, 3, i+1)
        plt.plot(z_test, x_exact, 'k--', label="exact analytical solution", linewidth=2)
        plt.plot(z_test, x_pred, 'r', label=f"{model_name} prediction", alpha=0.8)
        plt.title(f"solution for $\\xi={xi_val}$")
        plt.xlabel("z")
        plt.ylabel("x")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# plot loss curve
def plot_loss(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.yscale('log') # the error drops drastically at first and then very slowly so its better use this scale
    plt.title("training loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()
    
# plot the error analysis
def plot_error(model, model_name="PINN", test_xis=[0.1, 0.25, 0.4]):
    z_test = torch.linspace(0, 20, 200).view(-1, 1)
    plt.figure(figsize=(10, 4))
    for xi_val in test_xis:
        xi_test = torch.full_like(z_test, xi_val)
        with torch.no_grad():
            x_pred = model(z_test, xi_test).numpy()
        x_exact = analytical_solution(z_test.numpy(), xi_val)
        # calculate absolute error
        error = np.abs(x_exact - x_pred)
        plt.plot(z_test, error, label=f"error ($\\xi={xi_val}$)")
    plt.yscale('log')
    plt.title(f"absolute error over time (|exact analytic solution - PINN solution|) — {model_name}")
    plt.xlabel("z")
    plt.ylabel("absolute error")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()


pinn = OscillatorPINN()
history = train_pinn(pinn, steps=100000)
plot_predictions(pinn, "tanh PINN")
plot_loss(history)
plot_error(pinn, "tanh PINN")

```

    starting training...
    Step 0: Loss = 1.823138, Avg Loss of last 100 steps = 1.823138 (Best: 1.823138, Patience: 0/2000)
    Step 1000: Loss = 0.002900, Avg Loss of last 100 steps = 0.003426 (Best: 0.003426, Patience: 0/2000)
    Step 2000: Loss = 0.001225, Avg Loss of last 100 steps = 0.001409 (Best: 0.001409, Patience: 0/2000)
    Step 3000: Loss = 0.001285, Avg Loss of last 100 steps = 0.000911 (Best: 0.000911, Patience: 0/2000)
    Step 4000: Loss = 0.000564, Avg Loss of last 100 steps = 0.000685 (Best: 0.000587, Patience: 61/2000)
    Step 5000: Loss = 0.000455, Avg Loss of last 100 steps = 0.000622 (Best: 0.000495, Patience: 519/2000)
    Step 6000: Loss = 0.000416, Avg Loss of last 100 steps = 0.000463 (Best: 0.000427, Patience: 126/2000)
    Step 7000: Loss = 0.000380, Avg Loss of last 100 steps = 0.000532 (Best: 0.000404, Patience: 122/2000)
    Step 8000: Loss = 0.000349, Avg Loss of last 100 steps = 0.000482 (Best: 0.000364, Patience: 311/2000)
    Step 9000: Loss = 0.000362, Avg Loss of last 100 steps = 0.000463 (Best: 0.000339, Patience: 41/2000)
    Step 10000: Loss = 0.000300, Avg Loss of last 100 steps = 0.000314 (Best: 0.000314, Patience: 2/2000)
    Step 11000: Loss = 0.000388, Avg Loss of last 100 steps = 0.000373 (Best: 0.000294, Patience: 556/2000)
    Step 12000: Loss = 0.000263, Avg Loss of last 100 steps = 0.000339 (Best: 0.000271, Patience: 723/2000)
    Step 13000: Loss = 0.000231, Avg Loss of last 100 steps = 0.000328 (Best: 0.000236, Patience: 307/2000)
    Step 14000: Loss = 0.000156, Avg Loss of last 100 steps = 0.000323 (Best: 0.000183, Patience: 101/2000)
    Step 15000: Loss = 0.000429, Avg Loss of last 100 steps = 0.000179 (Best: 0.000158, Patience: 280/2000)
    Step 16000: Loss = 0.000191, Avg Loss of last 100 steps = 0.000250 (Best: 0.000158, Patience: 1280/2000)
    Step 17000: Loss = 0.000127, Avg Loss of last 100 steps = 0.000224 (Best: 0.000137, Patience: 83/2000)
    Step 18000: Loss = 0.000118, Avg Loss of last 100 steps = 0.000220 (Best: 0.000126, Patience: 330/2000)
    Step 19000: Loss = 0.000305, Avg Loss of last 100 steps = 0.000186 (Best: 0.000117, Patience: 799/2000)
    Step 20000: Loss = 0.000090, Avg Loss of last 100 steps = 0.000182 (Best: 0.000107, Patience: 122/2000)
    Step 21000: Loss = 0.000088, Avg Loss of last 100 steps = 0.000209 (Best: 0.000091, Patience: 300/2000)
    Step 22000: Loss = 0.000993, Avg Loss of last 100 steps = 0.000113 (Best: 0.000088, Patience: 8/2000)
    Step 23000: Loss = 0.000074, Avg Loss of last 100 steps = 0.000138 (Best: 0.000072, Patience: 513/2000)
    Step 24000: Loss = 0.000121, Avg Loss of last 100 steps = 0.000229 (Best: 0.000071, Patience: 748/2000)
    Step 25000: Loss = 0.000056, Avg Loss of last 100 steps = 0.000120 (Best: 0.000064, Patience: 916/2000)
    Step 26000: Loss = 0.000082, Avg Loss of last 100 steps = 0.000139 (Best: 0.000064, Patience: 1916/2000)
    
    Early stopping at step 26084
    Best loss: 0.000064, Current loss: 0.000086
    


    
![png](temp_files/temp_2_1.png)
    



    
![png](temp_files/temp_2_2.png)
    



    
![png](temp_files/temp_2_3.png)
    


## Experiment 2
* **Changes with respect experiment 1**: I have change the activation function from nn.Tanh() to nn.Relu()
* **Observations**: ReLu activation function is unsuitable for this second order ODEs because its second derivative is zero everywhere. This causes the physics loss term $\frac{d^2x}{dz^2}$ to vanish, preventing the model from learning the acceleration. This confirms that for this problem smooth activation functions like Tanh must be used.


```python
class ReluPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    def forward(self, z, xi):
        # normalization step
        z_norm = 2.0 * (z - 0.0) / (20.0 - 0.0) - 1.0
        xi_norm = 2.0 * (xi - 0.1) / (0.4 - 0.1) - 1.0
        # I feed the nromalized values into the layers
        inputs = torch.cat([z_norm, xi_norm], dim=1)
        return self.net(inputs)

relu_model=ReluPINN()
hist_relu = train_pinn(relu_model, steps=100000)

# plotting results for specific xi values
plot_predictions(relu_model, "relu PINN")

# plot loss curve
plot_loss(hist_relu)

# plot the error analysis
plot_error(relu_model, "relu  PINN")


```

    starting training...
    Step 0: Loss = 2.114813, Avg Loss of last 100 steps = 2.114813 (Best: 2.114813, Patience: 0/2000)
    Step 1000: Loss = 1.603648, Avg Loss of last 100 steps = 1.596320 (Best: 1.446555, Patience: 536/2000)
    Step 2000: Loss = 1.622082, Avg Loss of last 100 steps = 1.621351 (Best: 1.446555, Patience: 1536/2000)
    
    Early stopping at step 2464
    Best loss: 1.446555, Current loss: 1.619362
    


    
![png](temp_files/temp_4_1.png)
    



    
![png](temp_files/temp_4_2.png)
    



    
![png](temp_files/temp_4_3.png)
    


## Experiment 3 A and 3 B
* **Changes with respect experiment 1**: I have make a set of two experiments where I have taken a deeper model (EXPERIMENT 3 A) and a wider one (experiment 3 B). 
* **Observations**: The deeper model helped us to predict better. The wider model needs more max epochs to hit the patience limit, and it's worth a shot to continue to play and research with it, since, I quote:
> When compared to the shallow architecture, more hidden layers aid in the modeling of complicated
> nonlinear relationships (Sengupta et al, 2020), however, using PINNs for real
> problems can result in deep networks with many layers associated with high
> training costs and efficiency issues. For this reason, not only deep neural networks have been employed for PINNs but also shallow ANN are reported > in the literature. X-TFC, developed by Schiassi et al (2021), employs a single-layer
> NN trained using the ELM algorithm. While PIELM (Dwivedi and Srinivasan,
> 2020) is proposed as a faster alternative, using a hybrid neural network-based
> method that combines two ideas from PINN and ELM. ELM only updates the
> weights of the outer layer, leaving the weights of the inner layer unchanged.
> Finally, in Ramabathiran and Ramachandran (2021) a Sparse, Physics-based,
> and partially Interpretable Neural Networks (SPINN) is proposed
> 
>  — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.1.1, Page 23*


```python
class DeeperPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),   
            nn.Tanh(),
            nn.Linear(20, 20),  
            nn.Tanh(),
            nn.Linear(20, 20),  
            nn.Tanh(),
            nn.Linear(20, 20),  
            nn.Tanh(),
            nn.Linear(20, 20),  
            nn.Tanh(),
            nn.Linear(20, 20),  
            nn.Tanh(),
            nn.Linear(20, 1)         
        )
    def forward(self, z, xi):
        # normalization step
        z_norm = 2.0 * (z - 0.0) / (20.0 - 0.0) - 1.0
        xi_norm = 2.0 * (xi - 0.1) / (0.4 - 0.1) - 1.0
        # I feed the nromalized values into the layers
        inputs = torch.cat([z_norm, xi_norm], dim=1)
        return self.net(inputs)

deeper_model=DeeperPINN()
hist_deeper = train_pinn(deeper_model, steps=100000)

# plotting results for specific xi values
plot_predictions(deeper_model, "deeper PINN")

# plot loss curve
plot_loss(hist_deeper)

# plot the error analysis
plot_error(deeper_model, "deeper PINN")

```

    starting training...
    Step 0: Loss = 1.765270, Avg Loss of last 100 steps = 1.765270 (Best: 1.765270, Patience: 0/2000)
    Step 1000: Loss = 0.005075, Avg Loss of last 100 steps = 0.005663 (Best: 0.005663, Patience: 0/2000)
    Step 2000: Loss = 0.002699, Avg Loss of last 100 steps = 0.002838 (Best: 0.002836, Patience: 17/2000)
    Step 3000: Loss = 0.001453, Avg Loss of last 100 steps = 0.001610 (Best: 0.001588, Patience: 98/2000)
    Step 4000: Loss = 0.001040, Avg Loss of last 100 steps = 0.001024 (Best: 0.001024, Patience: 0/2000)
    Step 5000: Loss = 0.000812, Avg Loss of last 100 steps = 0.000995 (Best: 0.000832, Patience: 146/2000)
    Step 6000: Loss = 0.001301, Avg Loss of last 100 steps = 0.000779 (Best: 0.000768, Patience: 508/2000)
    Step 7000: Loss = 0.000513, Avg Loss of last 100 steps = 0.000690 (Best: 0.000557, Patience: 136/2000)
    Step 8000: Loss = 0.001051, Avg Loss of last 100 steps = 0.000620 (Best: 0.000533, Patience: 476/2000)
    Step 9000: Loss = 0.000462, Avg Loss of last 100 steps = 0.000683 (Best: 0.000440, Patience: 191/2000)
    Step 10000: Loss = 0.000476, Avg Loss of last 100 steps = 0.000548 (Best: 0.000427, Patience: 138/2000)
    Step 11000: Loss = 0.000393, Avg Loss of last 100 steps = 0.000558 (Best: 0.000411, Patience: 689/2000)
    Step 12000: Loss = 0.000784, Avg Loss of last 100 steps = 0.000579 (Best: 0.000411, Patience: 1689/2000)
    
    Early stopping at step 12311
    Best loss: 0.000411, Current loss: 0.000644
    


    
![png](temp_files/temp_6_1.png)
    



    
![png](temp_files/temp_6_2.png)
    



    
![png](temp_files/temp_6_3.png)
    



```python
class WiderPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200),  
            nn.Tanh(),
            nn.Linear(200, 1)   
        )
    def forward(self, z, xi):
        # normalization step
        z_norm = 2.0 * (z - 0.0) / (20.0 - 0.0) - 1.0
        xi_norm = 2.0 * (xi - 0.1) / (0.4 - 0.1) - 1.0
        # I feed the nromalized values into the layers
        inputs = torch.cat([z_norm, xi_norm], dim=1)
        return self.net(inputs)


wider_model=WiderPINN()
hist_wider = train_pinn(wider_model, steps=100000)


# plotting results for specific xi values
plot_predictions(wider_model, "wideer PINN")

# plot loss curve
plot_loss(hist_wider)

# plot the error analysis
plot_error(wider_model, "wider PINN")

```

    starting training...
    Step 0: Loss = 1.741606, Avg Loss of last 100 steps = 1.741606 (Best: 1.741606, Patience: 0/2000)
    Step 1000: Loss = 0.560047, Avg Loss of last 100 steps = 0.622980 (Best: 0.622980, Patience: 0/2000)
    Step 2000: Loss = 0.105576, Avg Loss of last 100 steps = 0.111493 (Best: 0.111493, Patience: 0/2000)
    Step 3000: Loss = 0.034848, Avg Loss of last 100 steps = 0.035588 (Best: 0.035588, Patience: 0/2000)
    Step 4000: Loss = 0.010255, Avg Loss of last 100 steps = 0.011412 (Best: 0.011412, Patience: 0/2000)
    Step 5000: Loss = 0.004188, Avg Loss of last 100 steps = 0.004592 (Best: 0.004592, Patience: 0/2000)
    Step 6000: Loss = 0.002458, Avg Loss of last 100 steps = 0.002539 (Best: 0.002539, Patience: 0/2000)
    Step 7000: Loss = 0.001661, Avg Loss of last 100 steps = 0.001820 (Best: 0.001820, Patience: 2/2000)
    Step 8000: Loss = 0.001787, Avg Loss of last 100 steps = 0.001546 (Best: 0.001491, Patience: 115/2000)
    Step 9000: Loss = 0.001331, Avg Loss of last 100 steps = 0.001363 (Best: 0.001346, Patience: 144/2000)
    Step 10000: Loss = 0.001269, Avg Loss of last 100 steps = 0.001289 (Best: 0.001280, Patience: 15/2000)
    Step 11000: Loss = 0.001152, Avg Loss of last 100 steps = 0.001253 (Best: 0.001221, Patience: 132/2000)
    Step 12000: Loss = 0.001168, Avg Loss of last 100 steps = 0.001214 (Best: 0.001178, Patience: 106/2000)
    Step 13000: Loss = 0.001234, Avg Loss of last 100 steps = 0.001141 (Best: 0.001139, Patience: 6/2000)
    Step 14000: Loss = 0.001035, Avg Loss of last 100 steps = 0.001117 (Best: 0.001106, Patience: 462/2000)
    Step 15000: Loss = 0.001121, Avg Loss of last 100 steps = 0.001116 (Best: 0.001077, Patience: 158/2000)
    Step 16000: Loss = 0.001096, Avg Loss of last 100 steps = 0.001097 (Best: 0.001065, Patience: 397/2000)
    Step 17000: Loss = 0.001069, Avg Loss of last 100 steps = 0.001051 (Best: 0.001019, Patience: 443/2000)
    Step 18000: Loss = 0.000956, Avg Loss of last 100 steps = 0.001054 (Best: 0.001005, Patience: 237/2000)
    Step 19000: Loss = 0.001112, Avg Loss of last 100 steps = 0.001018 (Best: 0.000972, Patience: 149/2000)
    Step 20000: Loss = 0.000947, Avg Loss of last 100 steps = 0.000984 (Best: 0.000949, Patience: 297/2000)
    Step 21000: Loss = 0.000916, Avg Loss of last 100 steps = 0.000966 (Best: 0.000935, Patience: 479/2000)
    Step 22000: Loss = 0.000964, Avg Loss of last 100 steps = 0.000964 (Best: 0.000925, Patience: 903/2000)
    Step 23000: Loss = 0.000914, Avg Loss of last 100 steps = 0.000932 (Best: 0.000896, Patience: 177/2000)
    Step 24000: Loss = 0.000916, Avg Loss of last 100 steps = 0.000907 (Best: 0.000882, Patience: 268/2000)
    Step 25000: Loss = 0.000876, Avg Loss of last 100 steps = 0.000876 (Best: 0.000866, Patience: 406/2000)
    Step 26000: Loss = 0.000789, Avg Loss of last 100 steps = 0.000888 (Best: 0.000846, Patience: 179/2000)
    Step 27000: Loss = 0.000877, Avg Loss of last 100 steps = 0.000849 (Best: 0.000846, Patience: 1179/2000)
    Step 28000: Loss = 0.000747, Avg Loss of last 100 steps = 0.000854 (Best: 0.000817, Patience: 43/2000)
    Step 29000: Loss = 0.000734, Avg Loss of last 100 steps = 0.000835 (Best: 0.000808, Patience: 305/2000)
    Step 30000: Loss = 0.000818, Avg Loss of last 100 steps = 0.000828 (Best: 0.000797, Patience: 617/2000)
    Step 31000: Loss = 0.000715, Avg Loss of last 100 steps = 0.000813 (Best: 0.000776, Patience: 411/2000)
    Step 32000: Loss = 0.000705, Avg Loss of last 100 steps = 0.000792 (Best: 0.000767, Patience: 93/2000)
    Step 33000: Loss = 0.000736, Avg Loss of last 100 steps = 0.000785 (Best: 0.000764, Patience: 196/2000)
    Step 34000: Loss = 0.000712, Avg Loss of last 100 steps = 0.000763 (Best: 0.000741, Patience: 239/2000)
    Step 35000: Loss = 0.000775, Avg Loss of last 100 steps = 0.000779 (Best: 0.000738, Patience: 589/2000)
    Step 36000: Loss = 0.000638, Avg Loss of last 100 steps = 0.000785 (Best: 0.000712, Patience: 676/2000)
    Step 37000: Loss = 0.000695, Avg Loss of last 100 steps = 0.000742 (Best: 0.000712, Patience: 1676/2000)
    
    Early stopping at step 37324
    Best loss: 0.000712, Current loss: 0.000632
    


    
![png](temp_files/temp_7_1.png)
    



    
![png](temp_files/temp_7_2.png)
    



    
![png](temp_files/temp_7_3.png)
    


## Experiment 4
* **Changes with respect experiment 1**: In the experiment 1 the lossess for position and velocity have the same weight as the physic loss. I now propose to modify that in experiment 4 going from the old loss:
                                                  <center>loss = loss_physics + loss_ic_x + loss_ic_v</center>                                                           
    to a new loss:
                                            <center>loss = loss_physics +$\lambda$*(loss_ic_x + loss_ic_v)</center>
  where I have choosed $\lambda$=10                                                        
* **Observations**: Since the Experiments 1, 3A and 3B were able to plot the solution almost identical for the first time steps the problem is to make it able to be accurate also for the last time steps and by enforcing the weight of the position and velocity this should be possible. But in my experiments the term loss_ic_x + loss_ic_v goes dropped to much fast in the first 1000 steps. This suggest that I need to implement some other methods on the loss or take a bigger $\lambda$. 



```python
def train_pinn_1_to_10_weights(model, steps=10000, lr=1e-3, patience=1000, min_delta=1e-6):
    # choosed optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # initial conditions targets (specified in the test on the ML4SC website)
    x0_target = 0.7
    v0_target = 1.2

    # windows for the average loss calculation
    window = 100
    loss_window = []
    
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    print("starting training...")
    
    
    for step in range(steps):
        optimizer.zero_grad()
         
        # sample random points for z and xi in given domains (specified in the test on the ML4SC website)
        batch_size = 1000
        z_col = torch.rand(batch_size, 1) * 20.0
        xi_col = torch.rand(batch_size, 1) * (0.4 - 0.1) + 0.1
        # at each of this random points we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col, xi_col)
        
        # initial condition loss (boundary loss)
        # Sample random xi for the boundary condition
        xi_bc = torch.rand(batch_size // 4, 1) * (0.4 - 0.1) + 0.1 # we pick a random xi again, no matter what xi I took, the oscillator always start in the same spot
        z_bc = torch.zeros_like(xi_bc) # z is always 0 here
        z_bc.requires_grad = True # we still need to track z if I want to track how much fast x change with z, so dx/dz

        # guess of the position at t=0:
        x_bc = model(z_bc, xi_bc)
        
        # then compute velocity at t=0:
        dx_bc = torch.autograd.grad(
            x_bc, z_bc,
            grad_outputs=torch.ones_like(x_bc),
            create_graph=True,
            retain_graph=True
        )[0]
        # then calculate the penalty for position and velocity:
        loss_ic_x = torch.mean((x_bc - x0_target)**2)
        loss_ic_v = torch.mean((dx_bc - v0_target)**2)
        
                                                                # total loss, infact without the penalty for position and velocity
        loss = loss_physics + 1000*loss_ic_x + 1000*loss_ic_v   # I would have that x=0 and v=0 for every time step minimize the loss and  
                                                                # so is the solution but that cannot be since of ICs different from zero
        
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(loss.item())
        
        # early stopping criteria: if the average loss doesnt improve at least of min_delta, for a number of consecutive steps equal to the patience, it stops early.
        loss_window.append(current_loss)
        if len(loss_window) > window:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
    
        if patience_counter >= patience:
            print(f"\nEarly stopping at step {step}")
            print(f"Best loss: {best_loss:.6f}, Current loss: {current_loss:.6f}")
            break
    
        if step % 1000 == 0:
            print(f"Step {step}: Loss = {current_loss:.6f}, Avg Loss of last 100 steps = {avg_loss:.6f} (Best: {best_loss:.6f}, Patience: {patience_counter}/{patience})")
    return loss_history

pinn_1_to_10_weights = OscillatorPINN()
hist_1_to_10_weights= train_pinn_1_to_10_weights(pinn_1_to_10_weights, steps=10000, lr=0.001)

deeper_model=DeeperPINN()
hist_deeper = train_pinn(deeper_model, steps=100000)

# plotting results for specific xi values
plot_predictions(deeper_model, "1to10weghts PINN")

# plot loss curve
plot_loss(hist_deeper)

# plot the error analysis
plot_error(deeper_model, "1to10weight PINN")

```

    starting training...
    Step 0: Loss = 2172.214355, Avg Loss of last 100 steps = 2172.214355 (Best: 2172.214355, Patience: 0/1000)
    Step 1000: Loss = 5.410488, Avg Loss of last 100 steps = 6.163869 (Best: 6.163869, Patience: 0/1000)
    Step 2000: Loss = 0.210933, Avg Loss of last 100 steps = 0.308982 (Best: 0.302743, Patience: 48/1000)
    Step 3000: Loss = 0.082788, Avg Loss of last 100 steps = 0.131517 (Best: 0.065026, Patience: 55/1000)
    Step 4000: Loss = 0.382795, Avg Loss of last 100 steps = 0.108378 (Best: 0.055401, Patience: 482/1000)
    Step 5000: Loss = 0.413036, Avg Loss of last 100 steps = 0.106390 (Best: 0.047408, Patience: 289/1000)
    Step 6000: Loss = 0.044447, Avg Loss of last 100 steps = 0.139151 (Best: 0.042487, Patience: 78/1000)
    Step 7000: Loss = 0.385513, Avg Loss of last 100 steps = 0.192703 (Best: 0.039323, Patience: 166/1000)
    Step 8000: Loss = 0.037546, Avg Loss of last 100 steps = 0.120418 (Best: 0.037243, Patience: 409/1000)
    Step 9000: Loss = 0.499904, Avg Loss of last 100 steps = 0.074321 (Best: 0.036361, Patience: 373/1000)
    starting training...
    Step 0: Loss = 1.869770, Avg Loss of last 100 steps = 1.869770 (Best: 1.869770, Patience: 0/2000)
    Step 1000: Loss = 0.006567, Avg Loss of last 100 steps = 0.006394 (Best: 0.006341, Patience: 25/2000)
    Step 2000: Loss = 0.003618, Avg Loss of last 100 steps = 0.003828 (Best: 0.003828, Patience: 0/2000)
    Step 3000: Loss = 0.002180, Avg Loss of last 100 steps = 0.002371 (Best: 0.002359, Patience: 38/2000)
    Step 4000: Loss = 0.001182, Avg Loss of last 100 steps = 0.001373 (Best: 0.001373, Patience: 0/2000)
    Step 5000: Loss = 0.000942, Avg Loss of last 100 steps = 0.000976 (Best: 0.000967, Patience: 54/2000)
    Step 6000: Loss = 0.000921, Avg Loss of last 100 steps = 0.000676 (Best: 0.000642, Patience: 137/2000)
    Step 7000: Loss = 0.000418, Avg Loss of last 100 steps = 0.000576 (Best: 0.000504, Patience: 261/2000)
    Step 8000: Loss = 0.001421, Avg Loss of last 100 steps = 0.000445 (Best: 0.000389, Patience: 102/2000)
    Step 9000: Loss = 0.000448, Avg Loss of last 100 steps = 0.000316 (Best: 0.000307, Patience: 181/2000)
    Step 10000: Loss = 0.000290, Avg Loss of last 100 steps = 0.000375 (Best: 0.000296, Patience: 105/2000)
    Step 11000: Loss = 0.000317, Avg Loss of last 100 steps = 0.000407 (Best: 0.000284, Patience: 242/2000)
    Step 12000: Loss = 0.000355, Avg Loss of last 100 steps = 0.000319 (Best: 0.000236, Patience: 88/2000)
    Step 13000: Loss = 0.000228, Avg Loss of last 100 steps = 0.000309 (Best: 0.000230, Patience: 780/2000)
    Step 14000: Loss = 0.000257, Avg Loss of last 100 steps = 0.000300 (Best: 0.000230, Patience: 1780/2000)
    Step 15000: Loss = 0.000222, Avg Loss of last 100 steps = 0.000320 (Best: 0.000223, Patience: 239/2000)
    Step 16000: Loss = 0.000214, Avg Loss of last 100 steps = 0.000336 (Best: 0.000223, Patience: 1239/2000)
    
    Early stopping at step 16761
    Best loss: 0.000223, Current loss: 0.000241
    


    
![png](temp_files/temp_9_1.png)
    



    
![png](temp_files/temp_9_2.png)
    



    
![png](temp_files/temp_9_3.png)
    


# Task ideas to do during GSoC 2026 : Physics-Informed Neural Network Diffusion Equation (PINNDE)
- Map a 3D zero mean, unit variance, diagonal normal to a 3D non-Gaussian density using a PINN. The inputs to the PINN are t, x, y, z — that is, the reverse time t ∈ [1, 0] and a point sampled from the 3D normal. The output of the PINN is the vector solution u(t, x, y, z). Since the PINN is conditioned on x, y, z, during training the points can be sampled from any convenient distribution, including quasi-random sampling such as Sobol sampling. (Of course, when used we must sample from a 3D normal.)
- Repeat with increasingly complex 3D non-Gaussian densities.
- Optional: Apply what has been learned from 1 and 2 to build a fast calorimeter simulator. Use Dataset 1 from the Fast Calorimeter Simulation Challenge 2022 [4].
- Publish the results in an ML paper.
## Things that I have learned during the exercise/test
- I used torch.rand and noticed that the loss fluctuated and I needed many points (1000+) to get a stable result. In 4 dimension, take 1000+ points in each dimension would put me to have 1'000'000'000'000+ points. If I take just 1000 random points in the whole quadridimensional space the gaps beetween the points would be bigger. So as mentioned I need to use the Sobol sequences, this would give me more omogeneurs cover of the space.
- In the xi=0.1 case the function became more complex with wiggles. The Calomiter Data would likely suffer of this spiky behaviour because of sharp peaks where the particles hit the detector, a standard PINN as we have seen in this tests/experiments would smoorth these out and fail.
- With PINNs even a simple problem as this one is not trivial: just taking a deeper PINN didnt improve my results, this suggest that even for simple problems PINN are very complex because I should implement even for this easy case some residual connections. The complexity of the architecture required in the problem of the calorimeter would be even greater. 
  


# Wrap up everything in a single plot graph, a single loss graph and a single difference graph

Let's now put all together on a single diagram for better understanding, here I can play with the different models I have discussed up. Relu model will be discarded from so on, since it's useless. Also the comparison would be more rigorous by the introduction of three tables for the metrics for the three damping terms.

# First set of experiments
We will now rerun the 4 experiments with the same parameters and put them on the same graph:
 * **BasePINN**: "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":2000
 * **DeeperPINN**: "lambda": 1.0, "dim": 50, "layers": 6, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":2000
 * **WiderPINN**: "lambda": 1.0, "dim": 200, "layers": 1, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":2000
 * **WeightedPINN**: "lambda": 10.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":2000


```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# define the PINN architecture model and forward pass
# input 2 features z and xi and ouput 1 feature x
class BasePINN(nn.Module):
    def __init__(self, activation=nn.Tanh, hidden_dim=50, hidden_layers=3):
        super().__init__()
        layers = [] # I initialize the number of layers as a empty list, I will add more and more layers with .append
        
        # input layer, is made by a linear layer and a layer with the activation function
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(activation())
        
        # a tot number of hidden layer, each of a linear layer and a layer with the activation function
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
            
        # Output layer (hidden -> 1)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, z, xi):
        # normalization step
        z_norm = 2.0 * (z - 0.0) / (20.0 - 0.0) - 1.0
        xi_norm = 2.0 * (xi - 0.1) / (0.4 - 0.1) - 1.0
        # I feed the nromalized values into the layers
        inputs = torch.cat([z_norm, xi_norm], dim=1)
        return self.net(inputs)


# define the physics informed loss function 
def physics_loss(model, z, xi):
        
    # enable gradient tracking for z
    z.requires_grad = True
    
    # predict x
    x = model(z, xi)
    
    # compute first derivative (dx/dz)
    dx_dz = torch.autograd.grad(
        x, z,
        torch.ones_like(x),
        create_graph=True
    )[0]
    
    # compute second derivative (d^2x/dz^2)
    d2x_dz2 = torch.autograd.grad(
        dx_dz, z,
        torch.ones_like(dx_dz), 
        create_graph=True
    )[0]
    
                                            # this is just the left hand side of the PDE specified in the
    residual = d2x_dz2 + 2 * xi * dx_dz + x # test on the ML4SC website and, since is = 0, the residual
                                            # just tell me how much off the prediction x is from the 0.

                                    # the mean of one positive and one negative
    return torch.mean(residual**2)  # value of the same module is 0, so it's
                                    # required to square to account them as penalty
    
# training loop 
def train_experiment(model_name, activation, lambda_ic=1.0, hidden_dim=50, layers=3, steps=10000, batch_size=1000, lr=1e-3, patience=1000, min_delta=1e-6):

    print(f"--- starting Experiment: {model_name} (loss lambda_ICs={lambda_ic}) ---")
    
    # I use the previous class to initialize the model
    model = BasePINN(activation=activation, hidden_dim=hidden_dim, hidden_layers=layers)
    
    # choosed optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # initial conditions targets (specified in the test on the ML4SC website)
    x0_target = 0.7
    v0_target = 1.2
    
    # windows for the average loss calculation
    window = 100
    loss_window = []
    
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    for step in range(steps):
        optimizer.zero_grad()
        
        # sample random points for z and xi in given domains (specified in the test on the ML4SC website)
        z_col = torch.rand(batch_size, 1) * 20.0
        xi_col = torch.rand(batch_size, 1) * (0.4 - 0.1) + 0.1
        # at each of this random points we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col, xi_col)
        
        # initial condition loss (boundary loss)
        # Sample random xi for the boundary condition
        xi_bc = torch.rand(batch_size//4, 1) * (0.4 - 0.1) + 0.1 # we pick a random xi again, no matter what xi I took, the oscillator always start in the same spot
        z_bc = torch.zeros_like(xi_bc) # z is always 0 here
        z_bc.requires_grad = True # we still need to track z if I want to track how much fast x change with z, so dx/dz

        # guess of the position at t=0:
        x_bc = model(z_bc, xi_bc)

        # then compute the velocity at t=0:
        dx_bc = torch.autograd.grad(
            x_bc, z_bc, 
            torch.ones_like(x_bc), 
            create_graph=True
        )[0]
        # than calculate the penalty for position and velocity:
        loss_ic_val = torch.mean((x_bc - x0_target)**2) + torch.mean((dx_bc - v0_target)**2)
        
                                                        # total loss, infact without the penalty for position and velocity        
        loss = loss_physics + (lambda_ic * loss_ic_val) # I would have that x=0 and v=0 for every time step minimize the loss and 
                                                        # so is the solution but that cannot be since of ICs different from zero       
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        loss_history.append(loss.item())
        
        # early stopping criteria: if the average loss doesnt improve at least of min_delta, for a number of consecutive steps equal to the patience, it stops early.
        loss_window.append(current_loss)
        if len(loss_window) > window:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
    
        if patience_counter >= patience:
            print(f"\nEarly stopping at step {step}")
            print(f"Best loss: {best_loss:.6f}, Current loss: {current_loss:.6f}")
            break
    
        if step % 1000 == 0:
            print(f"Step {step}: Loss = {current_loss:.6f}, Avg Loss of last 100 steps = {avg_loss:.6f} (Best: {best_loss:.6f}, Patience: {patience_counter}/{patience})")            
    return model, loss_history

# validation & visualization
def analytical_solution(z, xi, x0=0.7, v0=1.2):
    omega_n = 1.0 # from wikipedia page  on Harmonic oscillator, this is the natural frequency, this is just the coeff of x in the PDE so it's 1
    omega_d = np.sqrt(omega_n**2 - xi**2) # this is the damped natural frequency in the case of omega_n=1
    
    # integrations constants A and B to be founded with ICs, since the solution from  
    # wikipedia is x(z)=exp(-xi*omega_n*z)*(A*cos(omega_d*t)+B*sin(omega_d*z)) I would have:
    # x(0) = A = x0
    # x'(0) = -xi*omega_n*A + omega_d*B = v0    
    A = x0
    B = (v0 + xi * omega_n * x0) / omega_d
    
    return np.exp(-xi * omega_n * z) * (A * np.cos(omega_d * z) + B * np.sin(omega_d * z))

# define the experiments
experiments = [
    {"name": "Standard Tanh", "act": nn.Tanh, "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience": 2000},
    # {"name": "ReLU", "act": nn.ReLU, "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 2000, "bs":1000},
    {"name": "Deeper Network", "act": nn.Tanh, "lambda": 1.0, "dim": 50, "layers": 6, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience": 2000},
    {"name": "Wider Network", "act": nn.Tanh, "lambda": 1.0, "dim": 200, "layers": 1, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience": 2000},
    {"name": "Weighted loss ICs", "act": nn.Tanh, "lambda": 10.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience": 2000}
]


def run_experiments(experiments):
    results = {}
    for exp in experiments:
        model, hist = train_experiment(
            model_name=exp["name"],
            activation=exp["act"],
            lambda_ic=exp["lambda"],
            hidden_dim=exp["dim"],
            layers=exp["layers"],
            steps=exp["number of steps"],
            batch_size=exp["bs"],
            lr=exp["lr"],
            patience=exp["patience"]
        )
        results[exp["name"]] = {"model": model, "hist": hist}
    return results

def plot_all_predictions(results, test_xis=[0.1, 0.25, 0.4]):
    z_plot = torch.linspace(0, 20, 200).view(-1, 1)
    plt.figure(figsize=(12, 5))
    for i, xi_val in enumerate(test_xis):
        xi_plot = torch.full_like(z_plot, xi_val)
        # exact solution 
        x_exact = analytical_solution(z_plot.numpy(), xi_val)
        plt.subplot(1, 3, i+1)
        plt.plot(z_plot, x_exact, 'k--', linewidth=2, label="exact analytical solution", alpha=0.6)
        for name, data in results.items():
            # PINN prediction
            with torch.no_grad():
                pred = data["model"](z_plot, xi_plot).numpy()
            style = ':' if "ReLU" in name else '-'
            plt.plot(z_plot, pred, linestyle=style, linewidth=1.5, label=name)
        plt.title(f"Solution for $\\xi={xi_val}$")
        plt.xlabel("z")
        plt.ylabel("x")
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_losses(results):
    plt.figure(figsize=(6, 4))
    for name, data in results.items():
        plt.plot(data["hist"], label=name)
    plt.yscale('log')
    plt.title("training loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

def plot_all_errors(results, test_xis=[0.1, 0.25, 0.4]):
    z_plot = torch.linspace(0, 20, 200).view(-1, 1)
    plt.figure(figsize=(12, 5))
    for i, xi_val in enumerate(test_xis):
        xi_plot = torch.full_like(z_plot, xi_val)
        x_exact = analytical_solution(z_plot.numpy(), xi_val)
        plt.subplot(1, 3, i+1)
        for name, data in results.items():
            with torch.no_grad():
                pred = data["model"](z_plot, xi_plot).numpy()
            # calculate absolute error
            error = np.abs(x_exact - pred)
            plt.plot(z_plot, error, label=name)
        plt.yscale('log')
        plt.title(f"absolute error over time (|exact analytic solution - PINN solution|) ($\\xi={xi_val}$)")
        plt.xlabel("z")
        plt.ylabel("absolute error")
        plt.grid(True, which="both", ls='-', alpha=0.5)
        plt.legend()
    plt.tight_layout()
    plt.show()

def summary_tables(results, test_xis=[0.1, 0.25, 0.4]):
    z_test = torch.linspace(0, 20, 500).view(-1, 1)
    for xi_val in test_xis:
        xi_test = torch.full_like(z_test, xi_val)
        exact = analytical_solution(z_test.numpy(), xi_val)
        rows = []
        for name, data in results.items():
            with torch.no_grad():
                pred = data["model"](z_test, xi_test).numpy()
            err = np.abs(exact - pred)
            rows.append({
                "Model": name,
                "MAE": np.mean(err),
                "Max Error": np.max(err),
                "L2 Rel Error": np.sqrt(np.sum((exact - pred)**2) / np.sum(exact**2)),
                "Final Loss": data["hist"][-1],
                "Steps": len(data["hist"])
            })
        df = pd.DataFrame(rows)
        print(f"\n{'='*70}")
        print(f"  ξ = {xi_val}")
        print(f"{'='*70}")
        print(df.to_string(index=False, float_format="%.6f"))
    print()


    
results = run_experiments(experiments) # initializzation and training
plot_all_predictions(results) # plotting results for specific xi values
plot_all_losses(results) # plot loss curve
plot_all_errors(results) # plot the error analysis
summary_tables(results) # make a summary table where I have some metrics as max, mar, L2 
```

    --- starting Experiment: Standard Tanh (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.822458, Avg Loss of last 100 steps = 1.822458 (Best: 1.822458, Patience: 0/2000)
    Step 1000: Loss = 0.003223, Avg Loss of last 100 steps = 0.003805 (Best: 0.003805, Patience: 0/2000)
    Step 2000: Loss = 0.001191, Avg Loss of last 100 steps = 0.001416 (Best: 0.001396, Patience: 15/2000)
    Step 3000: Loss = 0.000891, Avg Loss of last 100 steps = 0.001105 (Best: 0.001017, Patience: 123/2000)
    Step 4000: Loss = 0.000909, Avg Loss of last 100 steps = 0.000719 (Best: 0.000690, Patience: 56/2000)
    Step 5000: Loss = 0.000616, Avg Loss of last 100 steps = 0.000628 (Best: 0.000556, Patience: 208/2000)
    Step 6000: Loss = 0.000433, Avg Loss of last 100 steps = 0.000573 (Best: 0.000486, Patience: 185/2000)
    Step 7000: Loss = 0.000530, Avg Loss of last 100 steps = 0.000410 (Best: 0.000403, Patience: 9/2000)
    Step 8000: Loss = 0.000466, Avg Loss of last 100 steps = 0.000457 (Best: 0.000387, Patience: 236/2000)
    Step 9000: Loss = 0.000557, Avg Loss of last 100 steps = 0.000530 (Best: 0.000350, Patience: 325/2000)
    Step 10000: Loss = 0.000470, Avg Loss of last 100 steps = 0.000412 (Best: 0.000315, Patience: 109/2000)
    Step 11000: Loss = 0.000269, Avg Loss of last 100 steps = 0.000365 (Best: 0.000305, Patience: 60/2000)
    Step 12000: Loss = 0.001703, Avg Loss of last 100 steps = 0.000331 (Best: 0.000263, Patience: 13/2000)
    Step 13000: Loss = 0.000211, Avg Loss of last 100 steps = 0.000386 (Best: 0.000243, Patience: 188/2000)
    Step 14000: Loss = 0.000165, Avg Loss of last 100 steps = 0.000285 (Best: 0.000189, Patience: 190/2000)
    Step 15000: Loss = 0.000122, Avg Loss of last 100 steps = 0.000196 (Best: 0.000157, Patience: 39/2000)
    Step 16000: Loss = 0.000270, Avg Loss of last 100 steps = 0.000149 (Best: 0.000128, Patience: 67/2000)
    Step 17000: Loss = 0.000127, Avg Loss of last 100 steps = 0.000209 (Best: 0.000121, Patience: 409/2000)
    Step 18000: Loss = 0.000298, Avg Loss of last 100 steps = 0.000248 (Best: 0.000101, Patience: 34/2000)
    Step 19000: Loss = 0.000082, Avg Loss of last 100 steps = 0.000172 (Best: 0.000101, Patience: 1034/2000)
    Step 20000: Loss = 0.000371, Avg Loss of last 100 steps = 0.000169 (Best: 0.000088, Patience: 472/2000)
    Step 21000: Loss = 0.000078, Avg Loss of last 100 steps = 0.000172 (Best: 0.000082, Patience: 197/2000)
    Step 22000: Loss = 0.000100, Avg Loss of last 100 steps = 0.000153 (Best: 0.000072, Patience: 763/2000)
    Step 23000: Loss = 0.000212, Avg Loss of last 100 steps = 0.000115 (Best: 0.000072, Patience: 1763/2000)
    
    Early stopping at step 23237
    Best loss: 0.000072, Current loss: 0.000079
    --- starting Experiment: Deeper Network (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.757809, Avg Loss of last 100 steps = 1.757809 (Best: 1.757809, Patience: 0/2000)
    Step 1000: Loss = 0.003041, Avg Loss of last 100 steps = 0.004676 (Best: 0.004556, Patience: 64/2000)
    Step 2000: Loss = 0.004497, Avg Loss of last 100 steps = 0.002849 (Best: 0.002104, Patience: 186/2000)
    Step 3000: Loss = 0.001638, Avg Loss of last 100 steps = 0.002071 (Best: 0.001670, Patience: 599/2000)
    Step 4000: Loss = 0.001260, Avg Loss of last 100 steps = 0.001838 (Best: 0.001395, Patience: 101/2000)
    Step 5000: Loss = 0.000883, Avg Loss of last 100 steps = 0.001756 (Best: 0.001395, Patience: 1101/2000)
    Step 6000: Loss = 0.000868, Avg Loss of last 100 steps = 0.001412 (Best: 0.001060, Patience: 777/2000)
    Step 7000: Loss = 0.003551, Avg Loss of last 100 steps = 0.001430 (Best: 0.000810, Patience: 684/2000)
    Step 8000: Loss = 0.002813, Avg Loss of last 100 steps = 0.001031 (Best: 0.000637, Patience: 219/2000)
    Step 9000: Loss = 0.000375, Avg Loss of last 100 steps = 0.000830 (Best: 0.000597, Patience: 253/2000)
    Step 10000: Loss = 0.002109, Avg Loss of last 100 steps = 0.000830 (Best: 0.000597, Patience: 1253/2000)
    Step 11000: Loss = 0.000418, Avg Loss of last 100 steps = 0.000798 (Best: 0.000390, Patience: 328/2000)
    Step 12000: Loss = 0.001452, Avg Loss of last 100 steps = 0.000689 (Best: 0.000390, Patience: 1328/2000)
    
    Early stopping at step 12672
    Best loss: 0.000390, Current loss: 0.001365
    --- starting Experiment: Wider Network (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.709097, Avg Loss of last 100 steps = 1.709097 (Best: 1.709097, Patience: 0/2000)
    Step 1000: Loss = 0.705874, Avg Loss of last 100 steps = 0.751092 (Best: 0.751092, Patience: 0/2000)
    Step 2000: Loss = 0.183508, Avg Loss of last 100 steps = 0.196931 (Best: 0.196931, Patience: 0/2000)
    Step 3000: Loss = 0.072924, Avg Loss of last 100 steps = 0.079635 (Best: 0.079635, Patience: 0/2000)
    Step 4000: Loss = 0.035615, Avg Loss of last 100 steps = 0.034079 (Best: 0.034079, Patience: 0/2000)
    Step 5000: Loss = 0.014301, Avg Loss of last 100 steps = 0.016309 (Best: 0.016309, Patience: 0/2000)
    Step 6000: Loss = 0.008127, Avg Loss of last 100 steps = 0.008688 (Best: 0.008654, Patience: 9/2000)
    Step 7000: Loss = 0.004927, Avg Loss of last 100 steps = 0.004902 (Best: 0.004903, Patience: 1/2000)
    Step 8000: Loss = 0.003293, Avg Loss of last 100 steps = 0.003235 (Best: 0.003235, Patience: 0/2000)
    Step 9000: Loss = 0.003462, Avg Loss of last 100 steps = 0.002394 (Best: 0.002388, Patience: 1/2000)
    Step 10000: Loss = 0.002037, Avg Loss of last 100 steps = 0.001954 (Best: 0.001933, Patience: 13/2000)
    Step 11000: Loss = 0.001844, Avg Loss of last 100 steps = 0.001734 (Best: 0.001699, Patience: 157/2000)
    Step 12000: Loss = 0.001356, Avg Loss of last 100 steps = 0.001547 (Best: 0.001508, Patience: 79/2000)
    Step 13000: Loss = 0.001405, Avg Loss of last 100 steps = 0.001468 (Best: 0.001401, Patience: 117/2000)
    Step 14000: Loss = 0.001352, Avg Loss of last 100 steps = 0.001374 (Best: 0.001352, Patience: 453/2000)
    Step 15000: Loss = 0.001235, Avg Loss of last 100 steps = 0.001396 (Best: 0.001316, Patience: 369/2000)
    Step 16000: Loss = 0.001372, Avg Loss of last 100 steps = 0.001247 (Best: 0.001230, Patience: 46/2000)
    Step 17000: Loss = 0.001220, Avg Loss of last 100 steps = 0.001326 (Best: 0.001205, Patience: 164/2000)
    Step 18000: Loss = 0.001413, Avg Loss of last 100 steps = 0.001223 (Best: 0.001175, Patience: 78/2000)
    Step 19000: Loss = 0.001178, Avg Loss of last 100 steps = 0.001275 (Best: 0.001153, Patience: 179/2000)
    Step 20000: Loss = 0.001118, Avg Loss of last 100 steps = 0.001177 (Best: 0.001126, Patience: 627/2000)
    Step 21000: Loss = 0.000969, Avg Loss of last 100 steps = 0.001105 (Best: 0.001105, Patience: 0/2000)
    Step 22000: Loss = 0.001073, Avg Loss of last 100 steps = 0.001177 (Best: 0.001061, Patience: 166/2000)
    Step 23000: Loss = 0.001597, Avg Loss of last 100 steps = 0.001124 (Best: 0.001059, Patience: 274/2000)
    Step 24000: Loss = 0.001045, Avg Loss of last 100 steps = 0.001053 (Best: 0.001039, Patience: 88/2000)
    Step 25000: Loss = 0.001078, Avg Loss of last 100 steps = 0.001062 (Best: 0.001039, Patience: 1088/2000)
    Step 26000: Loss = 0.001061, Avg Loss of last 100 steps = 0.001056 (Best: 0.000994, Patience: 393/2000)
    Step 27000: Loss = 0.000812, Avg Loss of last 100 steps = 0.001056 (Best: 0.000965, Patience: 115/2000)
    Step 28000: Loss = 0.001298, Avg Loss of last 100 steps = 0.001010 (Best: 0.000965, Patience: 1115/2000)
    Step 29000: Loss = 0.001301, Avg Loss of last 100 steps = 0.000972 (Best: 0.000930, Patience: 58/2000)
    Step 30000: Loss = 0.000818, Avg Loss of last 100 steps = 0.001015 (Best: 0.000930, Patience: 1058/2000)
    Step 31000: Loss = 0.000895, Avg Loss of last 100 steps = 0.000979 (Best: 0.000911, Patience: 162/2000)
    Step 32000: Loss = 0.000875, Avg Loss of last 100 steps = 0.000987 (Best: 0.000899, Patience: 908/2000)
    Step 33000: Loss = 0.001009, Avg Loss of last 100 steps = 0.000973 (Best: 0.000889, Patience: 538/2000)
    Step 34000: Loss = 0.000995, Avg Loss of last 100 steps = 0.000984 (Best: 0.000867, Patience: 817/2000)
    Step 35000: Loss = 0.001047, Avg Loss of last 100 steps = 0.000869 (Best: 0.000849, Patience: 362/2000)
    Step 36000: Loss = 0.001053, Avg Loss of last 100 steps = 0.000913 (Best: 0.000849, Patience: 1362/2000)
    
    Early stopping at step 36638
    Best loss: 0.000849, Current loss: 0.001046
    --- starting Experiment: Weighted loss ICs (loss lambda_ICs=10.0) ---
    Step 0: Loss = 22.016445, Avg Loss of last 100 steps = 22.016445 (Best: 22.016445, Patience: 0/2000)
    Step 1000: Loss = 0.021478, Avg Loss of last 100 steps = 0.022858 (Best: 0.022858, Patience: 0/2000)
    Step 2000: Loss = 0.008247, Avg Loss of last 100 steps = 0.007744 (Best: 0.007645, Patience: 18/2000)
    Step 3000: Loss = 0.004371, Avg Loss of last 100 steps = 0.005641 (Best: 0.004827, Patience: 69/2000)
    Step 4000: Loss = 0.004737, Avg Loss of last 100 steps = 0.004584 (Best: 0.004049, Patience: 138/2000)
    Step 5000: Loss = 0.003708, Avg Loss of last 100 steps = 0.003978 (Best: 0.003530, Patience: 374/2000)
    Step 6000: Loss = 0.003754, Avg Loss of last 100 steps = 0.003583 (Best: 0.002930, Patience: 431/2000)
    Step 7000: Loss = 0.001923, Avg Loss of last 100 steps = 0.002657 (Best: 0.002358, Patience: 261/2000)
    Step 8000: Loss = 0.002179, Avg Loss of last 100 steps = 0.002539 (Best: 0.001765, Patience: 494/2000)
    Step 9000: Loss = 0.001293, Avg Loss of last 100 steps = 0.001450 (Best: 0.001296, Patience: 125/2000)
    Step 10000: Loss = 0.000998, Avg Loss of last 100 steps = 0.001025 (Best: 0.001016, Patience: 17/2000)
    Step 11000: Loss = 0.002205, Avg Loss of last 100 steps = 0.000912 (Best: 0.000886, Patience: 5/2000)
    Step 12000: Loss = 0.001959, Avg Loss of last 100 steps = 0.000934 (Best: 0.000843, Patience: 761/2000)
    Step 13000: Loss = 0.000913, Avg Loss of last 100 steps = 0.001864 (Best: 0.000822, Patience: 265/2000)
    Step 14000: Loss = 0.000653, Avg Loss of last 100 steps = 0.000676 (Best: 0.000676, Patience: 0/2000)
    Step 15000: Loss = 0.003020, Avg Loss of last 100 steps = 0.001486 (Best: 0.000626, Patience: 203/2000)
    Step 16000: Loss = 0.000559, Avg Loss of last 100 steps = 0.000528 (Best: 0.000527, Patience: 2/2000)
    Step 17000: Loss = 0.000470, Avg Loss of last 100 steps = 0.001400 (Best: 0.000489, Patience: 71/2000)
    Step 18000: Loss = 0.000457, Avg Loss of last 100 steps = 0.000710 (Best: 0.000489, Patience: 1071/2000)
    Step 19000: Loss = 0.000598, Avg Loss of last 100 steps = 0.000428 (Best: 0.000420, Patience: 34/2000)
    Step 20000: Loss = 0.002362, Avg Loss of last 100 steps = 0.001054 (Best: 0.000402, Patience: 415/2000)
    Step 21000: Loss = 0.000584, Avg Loss of last 100 steps = 0.001047 (Best: 0.000361, Patience: 324/2000)
    Step 22000: Loss = 0.005941, Avg Loss of last 100 steps = 0.000999 (Best: 0.000323, Patience: 171/2000)
    Step 23000: Loss = 0.000314, Avg Loss of last 100 steps = 0.001245 (Best: 0.000323, Patience: 1171/2000)
    Step 24000: Loss = 0.000292, Avg Loss of last 100 steps = 0.001419 (Best: 0.000306, Patience: 432/2000)
    Step 25000: Loss = 0.000330, Avg Loss of last 100 steps = 0.000913 (Best: 0.000291, Patience: 557/2000)
    Step 26000: Loss = 0.000265, Avg Loss of last 100 steps = 0.001555 (Best: 0.000280, Patience: 797/2000)
    Step 27000: Loss = 0.000363, Avg Loss of last 100 steps = 0.000726 (Best: 0.000273, Patience: 912/2000)
    Step 28000: Loss = 0.000261, Avg Loss of last 100 steps = 0.000269 (Best: 0.000260, Patience: 493/2000)
    Step 29000: Loss = 0.000931, Avg Loss of last 100 steps = 0.001130 (Best: 0.000251, Patience: 941/2000)
    Step 30000: Loss = 0.000267, Avg Loss of last 100 steps = 0.000260 (Best: 0.000251, Patience: 1941/2000)
    
    Early stopping at step 30059
    Best loss: 0.000251, Current loss: 0.000274
    


    
![png](temp_files/temp_13_1.png)
    



    
![png](temp_files/temp_13_2.png)
    



    
![png](temp_files/temp_13_3.png)
    


    
    ======================================================================
      ξ = 0.1
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.058850   0.129833      0.136030    0.000079  23238
       Deeper Network 0.112202   0.265122      0.260097    0.001365  12673
        Wider Network 0.129293   0.269806      0.283853    0.001046  36639
    Weighted loss ICs 0.116259   0.258113      0.267119    0.000274  30060
    
    ======================================================================
      ξ = 0.25
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.004006   0.009675      0.012779    0.000079  23238
       Deeper Network 0.016706   0.027460      0.046987    0.001365  12673
        Wider Network 0.012906   0.043762      0.047240    0.001046  36639
    Weighted loss ICs 0.003904   0.010326      0.012395    0.000274  30060
    
    ======================================================================
      ξ = 0.4
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.003248   0.007970      0.012011    0.000079  23238
       Deeper Network 0.014975   0.038446      0.052118    0.001365  12673
        Wider Network 0.026348   0.105943      0.106554    0.001046  36639
    Weighted loss ICs 0.005744   0.011531      0.019247    0.000274  30060
    
    

I notice how the deeper network stop earlier than the other, even if he doesnt exibit a plateau like the wider network, we decide to then modify the first set of experiment by:
- increasing the batch size in the second set of experiments
- increase the patience in the third set of experiments
those are all ways that we hope it will trigger the patience lately, for a bigger number of steps.

# Second set of experiments
I increase the batch size.
 * **BasePINN**: "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":2000
 * **DeeperPINN**: "lambda": 1.0, "dim": 50, "layers": 6, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":2000
 * **WiderPINN**: "lambda": 1.0, "dim": 200, "layers": 1, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":2000
 * **WeightedPINN**: "lambda": 10.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":2000


```python
# define the experiments
experiments = [
    {"name": "Standard Tanh", "act": nn.Tanh, "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":2000},
    # {"name": "ReLU", "act": nn.ReLU, "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 2000, "bs":1000},
    {"name": "Deeper Network", "act": nn.Tanh, "lambda": 1.0, "dim": 50, "layers": 6, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":2000},
    {"name": "Wider Network", "act": nn.Tanh, "lambda": 1.0, "dim": 200, "layers": 1, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":2000},
    {"name": "Weighted loss ICs", "act": nn.Tanh, "lambda": 10.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":2000}
]

results = run_experiments(experiments) # initializzation and training
plot_all_predictions(results) # plotting results for specific xi values
plot_all_losses(results) # plot loss curve
plot_all_errors(results) # plot the error analysis
summary_tables(results) # make a summary table where I have some metrics as max, mar, L2 
```

    --- starting Experiment: Standard Tanh (loss lambda_ICs=1.0) ---
    Step 0: Loss = 2.179735, Avg Loss of last 100 steps = 2.179735 (Best: 2.179735, Patience: 0/2000)
    Step 1000: Loss = 0.003239, Avg Loss of last 100 steps = 0.003429 (Best: 0.003429, Patience: 0/2000)
    Step 2000: Loss = 0.001294, Avg Loss of last 100 steps = 0.001136 (Best: 0.001132, Patience: 6/2000)
    Step 3000: Loss = 0.000651, Avg Loss of last 100 steps = 0.000658 (Best: 0.000658, Patience: 0/2000)
    Step 4000: Loss = 0.000457, Avg Loss of last 100 steps = 0.000462 (Best: 0.000463, Patience: 46/2000)
    Step 5000: Loss = 0.000443, Avg Loss of last 100 steps = 0.000544 (Best: 0.000438, Patience: 435/2000)
    Step 6000: Loss = 0.000469, Avg Loss of last 100 steps = 0.000472 (Best: 0.000404, Patience: 63/2000)
    Step 7000: Loss = 0.000360, Avg Loss of last 100 steps = 0.000512 (Best: 0.000391, Patience: 805/2000)
    Step 8000: Loss = 0.000616, Avg Loss of last 100 steps = 0.000419 (Best: 0.000391, Patience: 1805/2000)
    
    Early stopping at step 8195
    Best loss: 0.000391, Current loss: 0.000370
    --- starting Experiment: Deeper Network (loss lambda_ICs=1.0) ---
    Step 0: Loss = 2.040576, Avg Loss of last 100 steps = 2.040576 (Best: 2.040576, Patience: 0/2000)
    Step 1000: Loss = 0.003263, Avg Loss of last 100 steps = 0.003410 (Best: 0.002910, Patience: 32/2000)
    Step 2000: Loss = 0.001132, Avg Loss of last 100 steps = 0.001624 (Best: 0.001356, Patience: 46/2000)
    Step 3000: Loss = 0.002188, Avg Loss of last 100 steps = 0.001270 (Best: 0.000995, Patience: 546/2000)
    Step 4000: Loss = 0.000920, Avg Loss of last 100 steps = 0.000884 (Best: 0.000632, Patience: 250/2000)
    Step 5000: Loss = 0.001457, Avg Loss of last 100 steps = 0.001034 (Best: 0.000632, Patience: 1250/2000)
    Step 6000: Loss = 0.000355, Avg Loss of last 100 steps = 0.000875 (Best: 0.000407, Patience: 230/2000)
    Step 7000: Loss = 0.000320, Avg Loss of last 100 steps = 0.000724 (Best: 0.000407, Patience: 1230/2000)
    
    Early stopping at step 7770
    Best loss: 0.000407, Current loss: 0.000357
    --- starting Experiment: Wider Network (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.974374, Avg Loss of last 100 steps = 1.974374 (Best: 1.974374, Patience: 0/2000)
    Step 1000: Loss = 0.623076, Avg Loss of last 100 steps = 0.678158 (Best: 0.678158, Patience: 0/2000)
    Step 2000: Loss = 0.104473, Avg Loss of last 100 steps = 0.110043 (Best: 0.110043, Patience: 0/2000)
    Step 3000: Loss = 0.034057, Avg Loss of last 100 steps = 0.034787 (Best: 0.034787, Patience: 0/2000)
    Step 4000: Loss = 0.009380, Avg Loss of last 100 steps = 0.010131 (Best: 0.010131, Patience: 0/2000)
    Step 5000: Loss = 0.003574, Avg Loss of last 100 steps = 0.003924 (Best: 0.003924, Patience: 0/2000)
    Step 6000: Loss = 0.002091, Avg Loss of last 100 steps = 0.002192 (Best: 0.002192, Patience: 0/2000)
    Step 7000: Loss = 0.001699, Avg Loss of last 100 steps = 0.001655 (Best: 0.001655, Patience: 1/2000)
    Step 8000: Loss = 0.001555, Avg Loss of last 100 steps = 0.001532 (Best: 0.001483, Patience: 115/2000)
    Step 9000: Loss = 0.001462, Avg Loss of last 100 steps = 0.001407 (Best: 0.001395, Patience: 113/2000)
    Step 10000: Loss = 0.001276, Avg Loss of last 100 steps = 0.001332 (Best: 0.001325, Patience: 23/2000)
    Step 11000: Loss = 0.001485, Avg Loss of last 100 steps = 0.001294 (Best: 0.001267, Patience: 244/2000)
    Step 12000: Loss = 0.001367, Avg Loss of last 100 steps = 0.001271 (Best: 0.001234, Patience: 47/2000)
    Step 13000: Loss = 0.001189, Avg Loss of last 100 steps = 0.001196 (Best: 0.001191, Patience: 16/2000)
    Step 14000: Loss = 0.001339, Avg Loss of last 100 steps = 0.001163 (Best: 0.001160, Patience: 7/2000)
    Step 15000: Loss = 0.001034, Avg Loss of last 100 steps = 0.001127 (Best: 0.001111, Patience: 163/2000)
    Step 16000: Loss = 0.001046, Avg Loss of last 100 steps = 0.001083 (Best: 0.001082, Patience: 16/2000)
    Step 17000: Loss = 0.001104, Avg Loss of last 100 steps = 0.001078 (Best: 0.001061, Patience: 297/2000)
    Step 18000: Loss = 0.001067, Avg Loss of last 100 steps = 0.001046 (Best: 0.001021, Patience: 248/2000)
    Step 19000: Loss = 0.000970, Avg Loss of last 100 steps = 0.001022 (Best: 0.001002, Patience: 299/2000)
    Step 20000: Loss = 0.000878, Avg Loss of last 100 steps = 0.001010 (Best: 0.000978, Patience: 289/2000)
    Step 21000: Loss = 0.001025, Avg Loss of last 100 steps = 0.000975 (Best: 0.000944, Patience: 141/2000)
    Step 22000: Loss = 0.001091, Avg Loss of last 100 steps = 0.000958 (Best: 0.000927, Patience: 35/2000)
    Step 23000: Loss = 0.000880, Avg Loss of last 100 steps = 0.000939 (Best: 0.000899, Patience: 283/2000)
    Step 24000: Loss = 0.000926, Avg Loss of last 100 steps = 0.000873 (Best: 0.000872, Patience: 1/2000)
    Step 25000: Loss = 0.000910, Avg Loss of last 100 steps = 0.000927 (Best: 0.000848, Patience: 110/2000)
    Step 26000: Loss = 0.000803, Avg Loss of last 100 steps = 0.000841 (Best: 0.000831, Patience: 150/2000)
    Step 27000: Loss = 0.000822, Avg Loss of last 100 steps = 0.000847 (Best: 0.000800, Patience: 287/2000)
    Step 28000: Loss = 0.001202, Avg Loss of last 100 steps = 0.000799 (Best: 0.000774, Patience: 248/2000)
    Step 29000: Loss = 0.000720, Avg Loss of last 100 steps = 0.000786 (Best: 0.000774, Patience: 1248/2000)
    Step 30000: Loss = 0.000784, Avg Loss of last 100 steps = 0.000795 (Best: 0.000756, Patience: 642/2000)
    Step 31000: Loss = 0.000771, Avg Loss of last 100 steps = 0.000796 (Best: 0.000746, Patience: 164/2000)
    Step 32000: Loss = 0.000773, Avg Loss of last 100 steps = 0.000740 (Best: 0.000727, Patience: 693/2000)
    Step 33000: Loss = 0.000692, Avg Loss of last 100 steps = 0.000735 (Best: 0.000726, Patience: 148/2000)
    Step 34000: Loss = 0.000724, Avg Loss of last 100 steps = 0.000748 (Best: 0.000718, Patience: 246/2000)
    Step 35000: Loss = 0.000622, Avg Loss of last 100 steps = 0.000735 (Best: 0.000710, Patience: 345/2000)
    Step 36000: Loss = 0.000659, Avg Loss of last 100 steps = 0.000718 (Best: 0.000690, Patience: 94/2000)
    Step 37000: Loss = 0.000711, Avg Loss of last 100 steps = 0.000731 (Best: 0.000681, Patience: 635/2000)
    Step 38000: Loss = 0.000641, Avg Loss of last 100 steps = 0.000699 (Best: 0.000680, Patience: 817/2000)
    Step 39000: Loss = 0.000795, Avg Loss of last 100 steps = 0.000684 (Best: 0.000656, Patience: 277/2000)
    Step 40000: Loss = 0.000594, Avg Loss of last 100 steps = 0.000685 (Best: 0.000656, Patience: 1277/2000)
    Step 41000: Loss = 0.000687, Avg Loss of last 100 steps = 0.000648 (Best: 0.000647, Patience: 3/2000)
    Step 42000: Loss = 0.000696, Avg Loss of last 100 steps = 0.000640 (Best: 0.000626, Patience: 36/2000)
    Step 43000: Loss = 0.000685, Avg Loss of last 100 steps = 0.000650 (Best: 0.000620, Patience: 98/2000)
    Step 44000: Loss = 0.000607, Avg Loss of last 100 steps = 0.000643 (Best: 0.000619, Patience: 307/2000)
    Step 45000: Loss = 0.000667, Avg Loss of last 100 steps = 0.000617 (Best: 0.000612, Patience: 478/2000)
    Step 46000: Loss = 0.000580, Avg Loss of last 100 steps = 0.000629 (Best: 0.000609, Patience: 197/2000)
    Step 47000: Loss = 0.000595, Avg Loss of last 100 steps = 0.000617 (Best: 0.000604, Patience: 712/2000)
    Step 48000: Loss = 0.000563, Avg Loss of last 100 steps = 0.000609 (Best: 0.000600, Patience: 192/2000)
    Step 49000: Loss = 0.000567, Avg Loss of last 100 steps = 0.000635 (Best: 0.000587, Patience: 183/2000)
    Step 50000: Loss = 0.000625, Avg Loss of last 100 steps = 0.000609 (Best: 0.000573, Patience: 706/2000)
    Step 51000: Loss = 0.000719, Avg Loss of last 100 steps = 0.000593 (Best: 0.000572, Patience: 361/2000)
    Step 52000: Loss = 0.000583, Avg Loss of last 100 steps = 0.000598 (Best: 0.000565, Patience: 163/2000)
    Step 53000: Loss = 0.000552, Avg Loss of last 100 steps = 0.000569 (Best: 0.000557, Patience: 881/2000)
    Step 54000: Loss = 0.000543, Avg Loss of last 100 steps = 0.000557 (Best: 0.000555, Patience: 255/2000)
    Step 55000: Loss = 0.000674, Avg Loss of last 100 steps = 0.000566 (Best: 0.000544, Patience: 187/2000)
    Step 56000: Loss = 0.000477, Avg Loss of last 100 steps = 0.000556 (Best: 0.000536, Patience: 354/2000)
    Step 57000: Loss = 0.000505, Avg Loss of last 100 steps = 0.000554 (Best: 0.000524, Patience: 703/2000)
    Step 58000: Loss = 0.000515, Avg Loss of last 100 steps = 0.000544 (Best: 0.000524, Patience: 1703/2000)
    
    Early stopping at step 58297
    Best loss: 0.000524, Current loss: 0.000638
    --- starting Experiment: Weighted loss ICs (loss lambda_ICs=10.0) ---
    Step 0: Loss = 23.373379, Avg Loss of last 100 steps = 23.373379 (Best: 23.373379, Patience: 0/2000)
    Step 1000: Loss = 0.025204, Avg Loss of last 100 steps = 0.027354 (Best: 0.027354, Patience: 0/2000)
    Step 2000: Loss = 0.006749, Avg Loss of last 100 steps = 0.007166 (Best: 0.006535, Patience: 27/2000)
    Step 3000: Loss = 0.005228, Avg Loss of last 100 steps = 0.005652 (Best: 0.005324, Patience: 166/2000)
    Step 4000: Loss = 0.005002, Avg Loss of last 100 steps = 0.005002 (Best: 0.004529, Patience: 28/2000)
    Step 5000: Loss = 0.003454, Avg Loss of last 100 steps = 0.004069 (Best: 0.003951, Patience: 505/2000)
    Step 6000: Loss = 0.003131, Avg Loss of last 100 steps = 0.003540 (Best: 0.003226, Patience: 283/2000)
    Step 7000: Loss = 0.002511, Avg Loss of last 100 steps = 0.003255 (Best: 0.002586, Patience: 82/2000)
    Step 8000: Loss = 0.001770, Avg Loss of last 100 steps = 0.001914 (Best: 0.001882, Patience: 127/2000)
    Step 9000: Loss = 0.001208, Avg Loss of last 100 steps = 0.001332 (Best: 0.001332, Patience: 0/2000)
    Step 10000: Loss = 0.004568, Avg Loss of last 100 steps = 0.001218 (Best: 0.001054, Patience: 10/2000)
    Step 11000: Loss = 0.000850, Avg Loss of last 100 steps = 0.001496 (Best: 0.000809, Patience: 60/2000)
    Step 12000: Loss = 0.000638, Avg Loss of last 100 steps = 0.000693 (Best: 0.000666, Patience: 133/2000)
    Step 13000: Loss = 0.000630, Avg Loss of last 100 steps = 0.000603 (Best: 0.000602, Patience: 6/2000)
    Step 14000: Loss = 0.000578, Avg Loss of last 100 steps = 0.001096 (Best: 0.000602, Patience: 1006/2000)
    Step 15000: Loss = 0.000534, Avg Loss of last 100 steps = 0.001183 (Best: 0.000565, Patience: 623/2000)
    Step 16000: Loss = 0.000845, Avg Loss of last 100 steps = 0.000983 (Best: 0.000530, Patience: 439/2000)
    Step 17000: Loss = 0.001835, Avg Loss of last 100 steps = 0.001336 (Best: 0.000500, Patience: 36/2000)
    Step 18000: Loss = 0.000451, Avg Loss of last 100 steps = 0.000477 (Best: 0.000477, Patience: 0/2000)
    Step 19000: Loss = 0.000500, Avg Loss of last 100 steps = 0.000471 (Best: 0.000469, Patience: 976/2000)
    Step 20000: Loss = 0.000427, Avg Loss of last 100 steps = 0.001016 (Best: 0.000443, Patience: 116/2000)
    Step 21000: Loss = 0.000415, Avg Loss of last 100 steps = 0.000978 (Best: 0.000432, Patience: 275/2000)
    Step 22000: Loss = 0.000458, Avg Loss of last 100 steps = 0.001128 (Best: 0.000428, Patience: 372/2000)
    Step 23000: Loss = 0.000469, Avg Loss of last 100 steps = 0.000955 (Best: 0.000428, Patience: 1372/2000)
    
    Early stopping at step 23628
    Best loss: 0.000428, Current loss: 0.000456
    


    
![png](temp_files/temp_16_1.png)
    



    
![png](temp_files/temp_16_2.png)
    



    
![png](temp_files/temp_16_3.png)
    


    
    ======================================================================
      ξ = 0.1
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.127292   0.301547      0.297370    0.000370   8196
       Deeper Network 0.111823   0.275660      0.261520    0.000357   7771
        Wider Network 0.116409   0.257608      0.258571    0.000638  58298
    Weighted loss ICs 0.136388   0.318076      0.315852    0.000456  23629
    
    ======================================================================
      ξ = 0.25
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.005679   0.018856      0.020729    0.000370   8196
       Deeper Network 0.007651   0.028838      0.023933    0.000357   7771
        Wider Network 0.012355   0.048806      0.045323    0.000638  58298
    Weighted loss ICs 0.007596   0.027101      0.028374    0.000456  23629
    
    ======================================================================
      ξ = 0.4
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.005252   0.012668      0.019716    0.000370   8196
       Deeper Network 0.006350   0.013828      0.022531    0.000357   7771
        Wider Network 0.015908   0.069030      0.066712    0.000638  58298
    Weighted loss ICs 0.003341   0.008442      0.011595    0.000456  23629
    
    

# third set of experiments
I increase the patience:
 * **BasePINN**: "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":5000
 * **DeeperPINN**: "lambda": 1.0, "dim": 50, "layers": 6, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":5000
 * **WiderPINN**: "lambda": 1.0, "dim": 200, "layers": 1, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":5000
 * **WeightedPINN**: "lambda": 10.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":5000, "lr":1e-3, "patience":5000


```python
# define the experiments
experiments = [
    {"name": "Standard Tanh", "act": nn.Tanh, "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000},
    # {"name": "ReLU", "act": nn.ReLU, "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 2000, "bs":1000},
    {"name": "Deeper Network", "act": nn.Tanh, "lambda": 1.0, "dim": 50, "layers": 6, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000},
    {"name": "Wider Network", "act": nn.Tanh, "lambda": 1.0, "dim": 200, "layers": 1, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000},
    {"name": "Weighted loss ICs", "act": nn.Tanh, "lambda": 10.0, "dim": 50, "layers": 3, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000}
]

results = run_experiments(experiments) # initializzation and training
plot_all_predictions(results) # plotting results for specific xi values
plot_all_losses(results) # plot loss curve
plot_all_errors(results) # plot the error analysis
summary_tables(results) # make a summary table where I have some metrics as max, mar, L2 
```

    --- starting Experiment: Standard Tanh (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.758102, Avg Loss of last 100 steps = 1.758102 (Best: 1.758102, Patience: 0/5000)
    Step 1000: Loss = 0.003124, Avg Loss of last 100 steps = 0.003459 (Best: 0.003459, Patience: 0/5000)
    Step 2000: Loss = 0.001504, Avg Loss of last 100 steps = 0.001837 (Best: 0.001838, Patience: 5/5000)
    Step 3000: Loss = 0.000933, Avg Loss of last 100 steps = 0.001073 (Best: 0.000980, Patience: 34/5000)
    Step 4000: Loss = 0.000892, Avg Loss of last 100 steps = 0.000782 (Best: 0.000694, Patience: 129/5000)
    Step 5000: Loss = 0.000428, Avg Loss of last 100 steps = 0.000619 (Best: 0.000573, Patience: 314/5000)
    Step 6000: Loss = 0.000531, Avg Loss of last 100 steps = 0.000536 (Best: 0.000492, Patience: 331/5000)
    Step 7000: Loss = 0.000406, Avg Loss of last 100 steps = 0.000671 (Best: 0.000436, Patience: 86/5000)
    Step 8000: Loss = 0.000580, Avg Loss of last 100 steps = 0.000509 (Best: 0.000436, Patience: 1086/5000)
    Step 9000: Loss = 0.000434, Avg Loss of last 100 steps = 0.000506 (Best: 0.000401, Patience: 599/5000)
    Step 10000: Loss = 0.000374, Avg Loss of last 100 steps = 0.000499 (Best: 0.000385, Patience: 165/5000)
    Step 11000: Loss = 0.000454, Avg Loss of last 100 steps = 0.000371 (Best: 0.000367, Patience: 10/5000)
    Step 12000: Loss = 0.000291, Avg Loss of last 100 steps = 0.000514 (Best: 0.000367, Patience: 1010/5000)
    Step 13000: Loss = 0.000473, Avg Loss of last 100 steps = 0.000376 (Best: 0.000347, Patience: 195/5000)
    Step 14000: Loss = 0.000403, Avg Loss of last 100 steps = 0.000402 (Best: 0.000327, Patience: 550/5000)
    Step 15000: Loss = 0.000307, Avg Loss of last 100 steps = 0.000362 (Best: 0.000273, Patience: 36/5000)
    Step 16000: Loss = 0.000241, Avg Loss of last 100 steps = 0.000368 (Best: 0.000273, Patience: 1036/5000)
    Step 17000: Loss = 0.000345, Avg Loss of last 100 steps = 0.000292 (Best: 0.000230, Patience: 354/5000)
    Step 18000: Loss = 0.000227, Avg Loss of last 100 steps = 0.000310 (Best: 0.000230, Patience: 1354/5000)
    Step 19000: Loss = 0.000201, Avg Loss of last 100 steps = 0.000229 (Best: 0.000215, Patience: 668/5000)
    Step 20000: Loss = 0.002125, Avg Loss of last 100 steps = 0.000291 (Best: 0.000215, Patience: 1668/5000)
    Step 21000: Loss = 0.000201, Avg Loss of last 100 steps = 0.000235 (Best: 0.000177, Patience: 625/5000)
    Step 22000: Loss = 0.000315, Avg Loss of last 100 steps = 0.000230 (Best: 0.000177, Patience: 1625/5000)
    Step 23000: Loss = 0.000158, Avg Loss of last 100 steps = 0.000239 (Best: 0.000177, Patience: 2625/5000)
    Step 24000: Loss = 0.000143, Avg Loss of last 100 steps = 0.000198 (Best: 0.000177, Patience: 3625/5000)
    Step 25000: Loss = 0.000342, Avg Loss of last 100 steps = 0.000180 (Best: 0.000158, Patience: 72/5000)
    Step 26000: Loss = 0.001056, Avg Loss of last 100 steps = 0.000270 (Best: 0.000158, Patience: 1072/5000)
    Step 27000: Loss = 0.000127, Avg Loss of last 100 steps = 0.000312 (Best: 0.000144, Patience: 458/5000)
    Step 28000: Loss = 0.000154, Avg Loss of last 100 steps = 0.000243 (Best: 0.000139, Patience: 882/5000)
    Step 29000: Loss = 0.000190, Avg Loss of last 100 steps = 0.000201 (Best: 0.000139, Patience: 1882/5000)
    Step 30000: Loss = 0.000303, Avg Loss of last 100 steps = 0.000167 (Best: 0.000139, Patience: 2882/5000)
    Step 31000: Loss = 0.000148, Avg Loss of last 100 steps = 0.000185 (Best: 0.000139, Patience: 3882/5000)
    Step 32000: Loss = 0.001001, Avg Loss of last 100 steps = 0.000224 (Best: 0.000132, Patience: 306/5000)
    Step 33000: Loss = 0.000226, Avg Loss of last 100 steps = 0.000188 (Best: 0.000117, Patience: 113/5000)
    Step 34000: Loss = 0.000381, Avg Loss of last 100 steps = 0.000152 (Best: 0.000117, Patience: 1113/5000)
    Step 35000: Loss = 0.000122, Avg Loss of last 100 steps = 0.000153 (Best: 0.000112, Patience: 499/5000)
    Step 36000: Loss = 0.000130, Avg Loss of last 100 steps = 0.000179 (Best: 0.000111, Patience: 686/5000)
    Step 37000: Loss = 0.000125, Avg Loss of last 100 steps = 0.000196 (Best: 0.000106, Patience: 656/5000)
    Step 38000: Loss = 0.000123, Avg Loss of last 100 steps = 0.000199 (Best: 0.000106, Patience: 1656/5000)
    Step 39000: Loss = 0.000111, Avg Loss of last 100 steps = 0.000125 (Best: 0.000106, Patience: 2656/5000)
    Step 40000: Loss = 0.000093, Avg Loss of last 100 steps = 0.000164 (Best: 0.000102, Patience: 716/5000)
    Step 41000: Loss = 0.000259, Avg Loss of last 100 steps = 0.000192 (Best: 0.000098, Patience: 224/5000)
    Step 42000: Loss = 0.000108, Avg Loss of last 100 steps = 0.000244 (Best: 0.000090, Patience: 133/5000)
    Step 43000: Loss = 0.000096, Avg Loss of last 100 steps = 0.000123 (Best: 0.000090, Patience: 1133/5000)
    Step 44000: Loss = 0.000088, Avg Loss of last 100 steps = 0.000133 (Best: 0.000090, Patience: 2133/5000)
    Step 45000: Loss = 0.000773, Avg Loss of last 100 steps = 0.000120 (Best: 0.000090, Patience: 3133/5000)
    Step 46000: Loss = 0.000108, Avg Loss of last 100 steps = 0.000109 (Best: 0.000084, Patience: 649/5000)
    Step 47000: Loss = 0.000154, Avg Loss of last 100 steps = 0.000135 (Best: 0.000084, Patience: 1649/5000)
    Step 48000: Loss = 0.000311, Avg Loss of last 100 steps = 0.000152 (Best: 0.000084, Patience: 2649/5000)
    Step 49000: Loss = 0.000100, Avg Loss of last 100 steps = 0.000121 (Best: 0.000084, Patience: 3649/5000)
    Step 50000: Loss = 0.000090, Avg Loss of last 100 steps = 0.000160 (Best: 0.000084, Patience: 4649/5000)
    Step 51000: Loss = 0.000117, Avg Loss of last 100 steps = 0.000208 (Best: 0.000070, Patience: 63/5000)
    Step 52000: Loss = 0.000054, Avg Loss of last 100 steps = 0.000168 (Best: 0.000070, Patience: 1063/5000)
    Step 53000: Loss = 0.000057, Avg Loss of last 100 steps = 0.000125 (Best: 0.000061, Patience: 167/5000)
    Step 54000: Loss = 0.000083, Avg Loss of last 100 steps = 0.000082 (Best: 0.000053, Patience: 42/5000)
    Step 55000: Loss = 0.000055, Avg Loss of last 100 steps = 0.000144 (Best: 0.000053, Patience: 1042/5000)
    Step 56000: Loss = 0.000232, Avg Loss of last 100 steps = 0.000087 (Best: 0.000053, Patience: 2042/5000)
    Step 57000: Loss = 0.000035, Avg Loss of last 100 steps = 0.000156 (Best: 0.000050, Patience: 793/5000)
    Step 58000: Loss = 0.000082, Avg Loss of last 100 steps = 0.000150 (Best: 0.000043, Patience: 939/5000)
    Step 59000: Loss = 0.000078, Avg Loss of last 100 steps = 0.000074 (Best: 0.000040, Patience: 221/5000)
    Step 60000: Loss = 0.000048, Avg Loss of last 100 steps = 0.000143 (Best: 0.000040, Patience: 1221/5000)
    Step 61000: Loss = 0.000053, Avg Loss of last 100 steps = 0.000098 (Best: 0.000036, Patience: 879/5000)
    Step 62000: Loss = 0.000036, Avg Loss of last 100 steps = 0.000135 (Best: 0.000031, Patience: 428/5000)
    Step 63000: Loss = 0.000026, Avg Loss of last 100 steps = 0.000069 (Best: 0.000031, Patience: 1428/5000)
    Step 64000: Loss = 0.000090, Avg Loss of last 100 steps = 0.000120 (Best: 0.000031, Patience: 2428/5000)
    Step 65000: Loss = 0.000028, Avg Loss of last 100 steps = 0.000077 (Best: 0.000027, Patience: 582/5000)
    Step 66000: Loss = 0.000030, Avg Loss of last 100 steps = 0.000022 (Best: 0.000022, Patience: 6/5000)
    Step 67000: Loss = 0.000017, Avg Loss of last 100 steps = 0.000018 (Best: 0.000018, Patience: 243/5000)
    Step 68000: Loss = 0.000022, Avg Loss of last 100 steps = 0.000073 (Best: 0.000015, Patience: 482/5000)
    Step 69000: Loss = 0.000027, Avg Loss of last 100 steps = 0.000048 (Best: 0.000015, Patience: 1482/5000)
    Step 70000: Loss = 0.000011, Avg Loss of last 100 steps = 0.000103 (Best: 0.000011, Patience: 166/5000)
    Step 71000: Loss = 0.000013, Avg Loss of last 100 steps = 0.000150 (Best: 0.000011, Patience: 1166/5000)
    Step 72000: Loss = 0.000056, Avg Loss of last 100 steps = 0.000067 (Best: 0.000010, Patience: 877/5000)
    Step 73000: Loss = 0.000017, Avg Loss of last 100 steps = 0.000102 (Best: 0.000010, Patience: 1877/5000)
    Step 74000: Loss = 0.000047, Avg Loss of last 100 steps = 0.000028 (Best: 0.000009, Patience: 670/5000)
    Step 75000: Loss = 0.000023, Avg Loss of last 100 steps = 0.000016 (Best: 0.000009, Patience: 1670/5000)
    Step 76000: Loss = 0.000007, Avg Loss of last 100 steps = 0.000017 (Best: 0.000009, Patience: 2670/5000)
    Step 77000: Loss = 0.000010, Avg Loss of last 100 steps = 0.000124 (Best: 0.000009, Patience: 3670/5000)
    Step 78000: Loss = 0.000007, Avg Loss of last 100 steps = 0.000057 (Best: 0.000009, Patience: 4670/5000)
    
    Early stopping at step 78330
    Best loss: 0.000009, Current loss: 0.000010
    --- starting Experiment: Deeper Network (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.756607, Avg Loss of last 100 steps = 1.756607 (Best: 1.756607, Patience: 0/5000)
    Step 1000: Loss = 0.003809, Avg Loss of last 100 steps = 0.004696 (Best: 0.004317, Patience: 240/5000)
    Step 2000: Loss = 0.002678, Avg Loss of last 100 steps = 0.002913 (Best: 0.002654, Patience: 106/5000)
    Step 3000: Loss = 0.003106, Avg Loss of last 100 steps = 0.002229 (Best: 0.001619, Patience: 174/5000)
    Step 4000: Loss = 0.001178, Avg Loss of last 100 steps = 0.001732 (Best: 0.001310, Patience: 105/5000)
    Step 5000: Loss = 0.001499, Avg Loss of last 100 steps = 0.001435 (Best: 0.000846, Patience: 135/5000)
    Step 6000: Loss = 0.000644, Avg Loss of last 100 steps = 0.001298 (Best: 0.000780, Patience: 726/5000)
    Step 7000: Loss = 0.000626, Avg Loss of last 100 steps = 0.001095 (Best: 0.000649, Patience: 832/5000)
    Step 8000: Loss = 0.000848, Avg Loss of last 100 steps = 0.001052 (Best: 0.000649, Patience: 1832/5000)
    Step 9000: Loss = 0.000476, Avg Loss of last 100 steps = 0.001147 (Best: 0.000631, Patience: 591/5000)
    Step 10000: Loss = 0.001914, Avg Loss of last 100 steps = 0.001067 (Best: 0.000550, Patience: 355/5000)
    Step 11000: Loss = 0.000500, Avg Loss of last 100 steps = 0.000617 (Best: 0.000550, Patience: 1355/5000)
    Step 12000: Loss = 0.000419, Avg Loss of last 100 steps = 0.000538 (Best: 0.000488, Patience: 147/5000)
    Step 13000: Loss = 0.001645, Avg Loss of last 100 steps = 0.000716 (Best: 0.000488, Patience: 1147/5000)
    Step 14000: Loss = 0.000457, Avg Loss of last 100 steps = 0.000805 (Best: 0.000488, Patience: 2147/5000)
    Step 15000: Loss = 0.000359, Avg Loss of last 100 steps = 0.000454 (Best: 0.000454, Patience: 0/5000)
    Step 16000: Loss = 0.001553, Avg Loss of last 100 steps = 0.000614 (Best: 0.000273, Patience: 106/5000)
    Step 17000: Loss = 0.000832, Avg Loss of last 100 steps = 0.000488 (Best: 0.000273, Patience: 1106/5000)
    Step 18000: Loss = 0.000246, Avg Loss of last 100 steps = 0.000513 (Best: 0.000226, Patience: 359/5000)
    Step 19000: Loss = 0.000189, Avg Loss of last 100 steps = 0.000505 (Best: 0.000226, Patience: 1359/5000)
    Step 20000: Loss = 0.000284, Avg Loss of last 100 steps = 0.000537 (Best: 0.000226, Patience: 2359/5000)
    Step 21000: Loss = 0.000254, Avg Loss of last 100 steps = 0.000348 (Best: 0.000226, Patience: 3359/5000)
    Step 22000: Loss = 0.000135, Avg Loss of last 100 steps = 0.000471 (Best: 0.000226, Patience: 4359/5000)
    Step 23000: Loss = 0.000286, Avg Loss of last 100 steps = 0.000400 (Best: 0.000219, Patience: 950/5000)
    Step 24000: Loss = 0.000312, Avg Loss of last 100 steps = 0.000447 (Best: 0.000215, Patience: 358/5000)
    Step 25000: Loss = 0.000393, Avg Loss of last 100 steps = 0.000338 (Best: 0.000182, Patience: 399/5000)
    Step 26000: Loss = 0.000121, Avg Loss of last 100 steps = 0.000333 (Best: 0.000130, Patience: 488/5000)
    Step 27000: Loss = 0.000181, Avg Loss of last 100 steps = 0.000144 (Best: 0.000128, Patience: 19/5000)
    Step 28000: Loss = 0.000112, Avg Loss of last 100 steps = 0.000191 (Best: 0.000128, Patience: 1019/5000)
    Step 29000: Loss = 0.000159, Avg Loss of last 100 steps = 0.000382 (Best: 0.000128, Patience: 2019/5000)
    Step 30000: Loss = 0.000095, Avg Loss of last 100 steps = 0.000231 (Best: 0.000118, Patience: 205/5000)
    Step 31000: Loss = 0.000084, Avg Loss of last 100 steps = 0.000199 (Best: 0.000097, Patience: 399/5000)
    Step 32000: Loss = 0.000102, Avg Loss of last 100 steps = 0.000288 (Best: 0.000097, Patience: 1399/5000)
    Step 33000: Loss = 0.000072, Avg Loss of last 100 steps = 0.000261 (Best: 0.000097, Patience: 2399/5000)
    Step 34000: Loss = 0.000420, Avg Loss of last 100 steps = 0.000228 (Best: 0.000093, Patience: 583/5000)
    Step 35000: Loss = 0.000166, Avg Loss of last 100 steps = 0.000220 (Best: 0.000093, Patience: 1583/5000)
    Step 36000: Loss = 0.000295, Avg Loss of last 100 steps = 0.000191 (Best: 0.000076, Patience: 160/5000)
    Step 37000: Loss = 0.000735, Avg Loss of last 100 steps = 0.000240 (Best: 0.000047, Patience: 79/5000)
    Step 38000: Loss = 0.000029, Avg Loss of last 100 steps = 0.000148 (Best: 0.000030, Patience: 50/5000)
    Step 39000: Loss = 0.000149, Avg Loss of last 100 steps = 0.000181 (Best: 0.000030, Patience: 1050/5000)
    Step 40000: Loss = 0.000095, Avg Loss of last 100 steps = 0.000232 (Best: 0.000030, Patience: 2050/5000)
    Step 41000: Loss = 0.000122, Avg Loss of last 100 steps = 0.000340 (Best: 0.000030, Patience: 3050/5000)
    Step 42000: Loss = 0.000017, Avg Loss of last 100 steps = 0.000179 (Best: 0.000020, Patience: 879/5000)
    Step 43000: Loss = 0.000029, Avg Loss of last 100 steps = 0.000184 (Best: 0.000020, Patience: 1879/5000)
    Step 44000: Loss = 0.000041, Avg Loss of last 100 steps = 0.000165 (Best: 0.000020, Patience: 2879/5000)
    Step 45000: Loss = 0.000614, Avg Loss of last 100 steps = 0.000232 (Best: 0.000018, Patience: 750/5000)
    Step 46000: Loss = 0.000228, Avg Loss of last 100 steps = 0.000171 (Best: 0.000018, Patience: 1750/5000)
    Step 47000: Loss = 0.000362, Avg Loss of last 100 steps = 0.000120 (Best: 0.000018, Patience: 2750/5000)
    Step 48000: Loss = 0.000014, Avg Loss of last 100 steps = 0.000173 (Best: 0.000018, Patience: 3750/5000)
    Step 49000: Loss = 0.000008, Avg Loss of last 100 steps = 0.000208 (Best: 0.000008, Patience: 282/5000)
    Step 50000: Loss = 0.000103, Avg Loss of last 100 steps = 0.000059 (Best: 0.000008, Patience: 1282/5000)
    Step 51000: Loss = 0.000023, Avg Loss of last 100 steps = 0.000034 (Best: 0.000008, Patience: 2282/5000)
    Step 52000: Loss = 0.000011, Avg Loss of last 100 steps = 0.000092 (Best: 0.000008, Patience: 3282/5000)
    Step 53000: Loss = 0.000145, Avg Loss of last 100 steps = 0.000070 (Best: 0.000008, Patience: 4282/5000)
    
    Early stopping at step 53718
    Best loss: 0.000008, Current loss: 0.000040
    --- starting Experiment: Wider Network (loss lambda_ICs=1.0) ---
    Step 0: Loss = 2.196821, Avg Loss of last 100 steps = 2.196821 (Best: 2.196821, Patience: 0/5000)
    Step 1000: Loss = 0.827856, Avg Loss of last 100 steps = 0.877316 (Best: 0.877316, Patience: 0/5000)
    Step 2000: Loss = 0.175957, Avg Loss of last 100 steps = 0.186460 (Best: 0.186460, Patience: 0/5000)
    Step 3000: Loss = 0.063157, Avg Loss of last 100 steps = 0.067946 (Best: 0.067946, Patience: 0/5000)
    Step 4000: Loss = 0.027609, Avg Loss of last 100 steps = 0.026363 (Best: 0.026363, Patience: 0/5000)
    Step 5000: Loss = 0.008204, Avg Loss of last 100 steps = 0.010237 (Best: 0.010237, Patience: 0/5000)
    Step 6000: Loss = 0.004658, Avg Loss of last 100 steps = 0.004521 (Best: 0.004518, Patience: 1/5000)
    Step 7000: Loss = 0.002357, Avg Loss of last 100 steps = 0.002479 (Best: 0.002463, Patience: 31/5000)
    Step 8000: Loss = 0.001590, Avg Loss of last 100 steps = 0.001725 (Best: 0.001722, Patience: 7/5000)
    Step 9000: Loss = 0.001440, Avg Loss of last 100 steps = 0.001511 (Best: 0.001473, Patience: 120/5000)
    Step 10000: Loss = 0.001110, Avg Loss of last 100 steps = 0.001398 (Best: 0.001347, Patience: 153/5000)
    Step 11000: Loss = 0.001559, Avg Loss of last 100 steps = 0.001299 (Best: 0.001214, Patience: 96/5000)
    Step 12000: Loss = 0.001285, Avg Loss of last 100 steps = 0.001272 (Best: 0.001190, Patience: 94/5000)
    Step 13000: Loss = 0.001019, Avg Loss of last 100 steps = 0.001186 (Best: 0.001175, Patience: 116/5000)
    Step 14000: Loss = 0.001181, Avg Loss of last 100 steps = 0.001147 (Best: 0.001127, Patience: 20/5000)
    Step 15000: Loss = 0.001071, Avg Loss of last 100 steps = 0.001153 (Best: 0.001086, Patience: 192/5000)
    Step 16000: Loss = 0.001149, Avg Loss of last 100 steps = 0.001081 (Best: 0.001059, Patience: 461/5000)
    Step 17000: Loss = 0.001272, Avg Loss of last 100 steps = 0.001071 (Best: 0.001044, Patience: 288/5000)
    Step 18000: Loss = 0.001222, Avg Loss of last 100 steps = 0.001022 (Best: 0.001014, Patience: 15/5000)
    Step 19000: Loss = 0.001547, Avg Loss of last 100 steps = 0.001046 (Best: 0.000980, Patience: 165/5000)
    Step 20000: Loss = 0.000988, Avg Loss of last 100 steps = 0.001000 (Best: 0.000980, Patience: 1165/5000)
    Step 21000: Loss = 0.001017, Avg Loss of last 100 steps = 0.001016 (Best: 0.000970, Patience: 407/5000)
    Step 22000: Loss = 0.000983, Avg Loss of last 100 steps = 0.000945 (Best: 0.000933, Patience: 66/5000)
    Step 23000: Loss = 0.001268, Avg Loss of last 100 steps = 0.000970 (Best: 0.000925, Patience: 197/5000)
    Step 24000: Loss = 0.000952, Avg Loss of last 100 steps = 0.000941 (Best: 0.000911, Patience: 523/5000)
    Step 25000: Loss = 0.000796, Avg Loss of last 100 steps = 0.000938 (Best: 0.000876, Patience: 775/5000)
    Step 26000: Loss = 0.001140, Avg Loss of last 100 steps = 0.000911 (Best: 0.000876, Patience: 1775/5000)
    Step 27000: Loss = 0.000932, Avg Loss of last 100 steps = 0.000932 (Best: 0.000861, Patience: 169/5000)
    Step 28000: Loss = 0.000755, Avg Loss of last 100 steps = 0.000902 (Best: 0.000856, Patience: 714/5000)
    Step 29000: Loss = 0.000821, Avg Loss of last 100 steps = 0.000828 (Best: 0.000827, Patience: 2/5000)
    Step 30000: Loss = 0.001159, Avg Loss of last 100 steps = 0.000844 (Best: 0.000812, Patience: 29/5000)
    Step 31000: Loss = 0.000849, Avg Loss of last 100 steps = 0.000850 (Best: 0.000788, Patience: 219/5000)
    Step 32000: Loss = 0.001020, Avg Loss of last 100 steps = 0.000847 (Best: 0.000788, Patience: 1219/5000)
    Step 33000: Loss = 0.000809, Avg Loss of last 100 steps = 0.000822 (Best: 0.000777, Patience: 150/5000)
    Step 34000: Loss = 0.001047, Avg Loss of last 100 steps = 0.000785 (Best: 0.000777, Patience: 1150/5000)
    Step 35000: Loss = 0.000703, Avg Loss of last 100 steps = 0.000791 (Best: 0.000760, Patience: 260/5000)
    Step 36000: Loss = 0.000687, Avg Loss of last 100 steps = 0.000752 (Best: 0.000748, Patience: 945/5000)
    Step 37000: Loss = 0.000684, Avg Loss of last 100 steps = 0.000789 (Best: 0.000733, Patience: 555/5000)
    Step 38000: Loss = 0.000623, Avg Loss of last 100 steps = 0.000815 (Best: 0.000728, Patience: 678/5000)
    Step 39000: Loss = 0.001031, Avg Loss of last 100 steps = 0.000737 (Best: 0.000728, Patience: 1678/5000)
    Step 40000: Loss = 0.001057, Avg Loss of last 100 steps = 0.000761 (Best: 0.000700, Patience: 181/5000)
    Step 41000: Loss = 0.000824, Avg Loss of last 100 steps = 0.000794 (Best: 0.000700, Patience: 1181/5000)
    Step 42000: Loss = 0.000822, Avg Loss of last 100 steps = 0.000770 (Best: 0.000680, Patience: 126/5000)
    Step 43000: Loss = 0.000758, Avg Loss of last 100 steps = 0.000725 (Best: 0.000680, Patience: 1126/5000)
    Step 44000: Loss = 0.000664, Avg Loss of last 100 steps = 0.000777 (Best: 0.000680, Patience: 2126/5000)
    Step 45000: Loss = 0.000631, Avg Loss of last 100 steps = 0.000693 (Best: 0.000671, Patience: 359/5000)
    Step 46000: Loss = 0.000868, Avg Loss of last 100 steps = 0.000666 (Best: 0.000653, Patience: 28/5000)
    Step 47000: Loss = 0.000659, Avg Loss of last 100 steps = 0.000671 (Best: 0.000653, Patience: 1028/5000)
    Step 48000: Loss = 0.000596, Avg Loss of last 100 steps = 0.000679 (Best: 0.000640, Patience: 409/5000)
    Step 49000: Loss = 0.000654, Avg Loss of last 100 steps = 0.000690 (Best: 0.000640, Patience: 1409/5000)
    Step 50000: Loss = 0.000718, Avg Loss of last 100 steps = 0.000653 (Best: 0.000640, Patience: 2409/5000)
    Step 51000: Loss = 0.000624, Avg Loss of last 100 steps = 0.000668 (Best: 0.000615, Patience: 778/5000)
    Step 52000: Loss = 0.000649, Avg Loss of last 100 steps = 0.000646 (Best: 0.000609, Patience: 549/5000)
    Step 53000: Loss = 0.000548, Avg Loss of last 100 steps = 0.000676 (Best: 0.000609, Patience: 1549/5000)
    Step 54000: Loss = 0.000591, Avg Loss of last 100 steps = 0.000626 (Best: 0.000603, Patience: 795/5000)
    Step 55000: Loss = 0.000588, Avg Loss of last 100 steps = 0.000647 (Best: 0.000591, Patience: 207/5000)
    Step 56000: Loss = 0.000523, Avg Loss of last 100 steps = 0.000626 (Best: 0.000582, Patience: 87/5000)
    Step 57000: Loss = 0.000558, Avg Loss of last 100 steps = 0.000647 (Best: 0.000582, Patience: 1087/5000)
    Step 58000: Loss = 0.000650, Avg Loss of last 100 steps = 0.000640 (Best: 0.000577, Patience: 537/5000)
    Step 59000: Loss = 0.000778, Avg Loss of last 100 steps = 0.000613 (Best: 0.000572, Patience: 494/5000)
    Step 60000: Loss = 0.000550, Avg Loss of last 100 steps = 0.000626 (Best: 0.000572, Patience: 1494/5000)
    Step 61000: Loss = 0.000448, Avg Loss of last 100 steps = 0.000584 (Best: 0.000564, Patience: 818/5000)
    Step 62000: Loss = 0.000636, Avg Loss of last 100 steps = 0.000622 (Best: 0.000564, Patience: 1818/5000)
    Step 63000: Loss = 0.000518, Avg Loss of last 100 steps = 0.000554 (Best: 0.000548, Patience: 67/5000)
    Step 64000: Loss = 0.000534, Avg Loss of last 100 steps = 0.000616 (Best: 0.000548, Patience: 1067/5000)
    Step 65000: Loss = 0.000465, Avg Loss of last 100 steps = 0.000576 (Best: 0.000543, Patience: 730/5000)
    Step 66000: Loss = 0.000796, Avg Loss of last 100 steps = 0.000579 (Best: 0.000535, Patience: 440/5000)
    Step 67000: Loss = 0.000425, Avg Loss of last 100 steps = 0.000588 (Best: 0.000535, Patience: 1440/5000)
    Step 68000: Loss = 0.000796, Avg Loss of last 100 steps = 0.000586 (Best: 0.000531, Patience: 229/5000)
    Step 69000: Loss = 0.000511, Avg Loss of last 100 steps = 0.000579 (Best: 0.000524, Patience: 370/5000)
    Step 70000: Loss = 0.000551, Avg Loss of last 100 steps = 0.000545 (Best: 0.000513, Patience: 844/5000)
    Step 71000: Loss = 0.000552, Avg Loss of last 100 steps = 0.000524 (Best: 0.000513, Patience: 1844/5000)
    Step 72000: Loss = 0.000649, Avg Loss of last 100 steps = 0.000546 (Best: 0.000495, Patience: 195/5000)
    Step 73000: Loss = 0.000462, Avg Loss of last 100 steps = 0.000508 (Best: 0.000495, Patience: 1195/5000)
    Step 74000: Loss = 0.000686, Avg Loss of last 100 steps = 0.000570 (Best: 0.000495, Patience: 2195/5000)
    Step 75000: Loss = 0.000422, Avg Loss of last 100 steps = 0.000493 (Best: 0.000468, Patience: 50/5000)
    Step 76000: Loss = 0.000483, Avg Loss of last 100 steps = 0.000563 (Best: 0.000468, Patience: 1050/5000)
    Step 77000: Loss = 0.000484, Avg Loss of last 100 steps = 0.000520 (Best: 0.000468, Patience: 2050/5000)
    Step 78000: Loss = 0.000509, Avg Loss of last 100 steps = 0.000548 (Best: 0.000468, Patience: 3050/5000)
    Step 79000: Loss = 0.000545, Avg Loss of last 100 steps = 0.000530 (Best: 0.000464, Patience: 870/5000)
    Step 80000: Loss = 0.000469, Avg Loss of last 100 steps = 0.000484 (Best: 0.000464, Patience: 1870/5000)
    Step 81000: Loss = 0.000479, Avg Loss of last 100 steps = 0.000480 (Best: 0.000444, Patience: 221/5000)
    Step 82000: Loss = 0.000521, Avg Loss of last 100 steps = 0.000481 (Best: 0.000442, Patience: 489/5000)
    Step 83000: Loss = 0.000431, Avg Loss of last 100 steps = 0.000463 (Best: 0.000442, Patience: 1489/5000)
    Step 84000: Loss = 0.000424, Avg Loss of last 100 steps = 0.000475 (Best: 0.000430, Patience: 540/5000)
    Step 85000: Loss = 0.000509, Avg Loss of last 100 steps = 0.000456 (Best: 0.000430, Patience: 1540/5000)
    Step 86000: Loss = 0.000362, Avg Loss of last 100 steps = 0.000450 (Best: 0.000427, Patience: 718/5000)
    Step 87000: Loss = 0.000580, Avg Loss of last 100 steps = 0.000426 (Best: 0.000407, Patience: 72/5000)
    Step 88000: Loss = 0.000519, Avg Loss of last 100 steps = 0.000480 (Best: 0.000407, Patience: 1072/5000)
    Step 89000: Loss = 0.000364, Avg Loss of last 100 steps = 0.000461 (Best: 0.000407, Patience: 2072/5000)
    Step 90000: Loss = 0.000620, Avg Loss of last 100 steps = 0.000465 (Best: 0.000407, Patience: 3072/5000)
    Step 91000: Loss = 0.000482, Avg Loss of last 100 steps = 0.000425 (Best: 0.000407, Patience: 4072/5000)
    
    Early stopping at step 91928
    Best loss: 0.000407, Current loss: 0.000502
    --- starting Experiment: Weighted loss ICs (loss lambda_ICs=10.0) ---
    Step 0: Loss = 17.983477, Avg Loss of last 100 steps = 17.983477 (Best: 17.983477, Patience: 0/5000)
    Step 1000: Loss = 0.013339, Avg Loss of last 100 steps = 0.014344 (Best: 0.014329, Patience: 1/5000)
    Step 2000: Loss = 0.007031, Avg Loss of last 100 steps = 0.006488 (Best: 0.006007, Patience: 201/5000)
    Step 3000: Loss = 0.005640, Avg Loss of last 100 steps = 0.005158 (Best: 0.004353, Patience: 403/5000)
    Step 4000: Loss = 0.003183, Avg Loss of last 100 steps = 0.004232 (Best: 0.003532, Patience: 430/5000)
    Step 5000: Loss = 0.004417, Avg Loss of last 100 steps = 0.004205 (Best: 0.003077, Patience: 71/5000)
    Step 6000: Loss = 0.002919, Avg Loss of last 100 steps = 0.002656 (Best: 0.002656, Patience: 0/5000)
    Step 7000: Loss = 0.002512, Avg Loss of last 100 steps = 0.003028 (Best: 0.002409, Patience: 278/5000)
    Step 8000: Loss = 0.001844, Avg Loss of last 100 steps = 0.002600 (Best: 0.001673, Patience: 92/5000)
    Step 9000: Loss = 0.001302, Avg Loss of last 100 steps = 0.001416 (Best: 0.001416, Patience: 0/5000)
    Step 10000: Loss = 0.001388, Avg Loss of last 100 steps = 0.001211 (Best: 0.001211, Patience: 0/5000)
    Step 11000: Loss = 0.001747, Avg Loss of last 100 steps = 0.001975 (Best: 0.001041, Patience: 152/5000)
    Step 12000: Loss = 0.000841, Avg Loss of last 100 steps = 0.000815 (Best: 0.000815, Patience: 0/5000)
    Step 13000: Loss = 0.000678, Avg Loss of last 100 steps = 0.001705 (Best: 0.000805, Patience: 969/5000)
    Step 14000: Loss = 0.000639, Avg Loss of last 100 steps = 0.001105 (Best: 0.000698, Patience: 510/5000)
    Step 15000: Loss = 0.000581, Avg Loss of last 100 steps = 0.000567 (Best: 0.000561, Patience: 211/5000)
    Step 16000: Loss = 0.001285, Avg Loss of last 100 steps = 0.000603 (Best: 0.000531, Patience: 160/5000)
    Step 17000: Loss = 0.000809, Avg Loss of last 100 steps = 0.001447 (Best: 0.000515, Patience: 559/5000)
    Step 18000: Loss = 0.000512, Avg Loss of last 100 steps = 0.001057 (Best: 0.000491, Patience: 671/5000)
    Step 19000: Loss = 0.000581, Avg Loss of last 100 steps = 0.001439 (Best: 0.000487, Patience: 341/5000)
    Step 20000: Loss = 0.000420, Avg Loss of last 100 steps = 0.001193 (Best: 0.000449, Patience: 89/5000)
    Step 21000: Loss = 0.001878, Avg Loss of last 100 steps = 0.001341 (Best: 0.000438, Patience: 612/5000)
    Step 22000: Loss = 0.000609, Avg Loss of last 100 steps = 0.001257 (Best: 0.000438, Patience: 1612/5000)
    Step 23000: Loss = 0.000410, Avg Loss of last 100 steps = 0.000415 (Best: 0.000412, Patience: 16/5000)
    Step 24000: Loss = 0.004305, Avg Loss of last 100 steps = 0.001056 (Best: 0.000412, Patience: 1016/5000)
    Step 25000: Loss = 0.000462, Avg Loss of last 100 steps = 0.001160 (Best: 0.000409, Patience: 730/5000)
    Step 26000: Loss = 0.001566, Avg Loss of last 100 steps = 0.000962 (Best: 0.000409, Patience: 1730/5000)
    Step 27000: Loss = 0.000920, Avg Loss of last 100 steps = 0.001454 (Best: 0.000403, Patience: 749/5000)
    Step 28000: Loss = 0.001622, Avg Loss of last 100 steps = 0.000674 (Best: 0.000401, Patience: 129/5000)
    Step 29000: Loss = 0.000381, Avg Loss of last 100 steps = 0.001210 (Best: 0.000401, Patience: 1129/5000)
    Step 30000: Loss = 0.001022, Avg Loss of last 100 steps = 0.001065 (Best: 0.000392, Patience: 729/5000)
    Step 31000: Loss = 0.000320, Avg Loss of last 100 steps = 0.001294 (Best: 0.000390, Patience: 298/5000)
    Step 32000: Loss = 0.000434, Avg Loss of last 100 steps = 0.000537 (Best: 0.000379, Patience: 373/5000)
    Step 33000: Loss = 0.000800, Avg Loss of last 100 steps = 0.001449 (Best: 0.000377, Patience: 678/5000)
    Step 34000: Loss = 0.000396, Avg Loss of last 100 steps = 0.001041 (Best: 0.000377, Patience: 1678/5000)
    Step 35000: Loss = 0.000492, Avg Loss of last 100 steps = 0.001117 (Best: 0.000377, Patience: 2678/5000)
    Step 36000: Loss = 0.000429, Avg Loss of last 100 steps = 0.000885 (Best: 0.000377, Patience: 3678/5000)
    Step 37000: Loss = 0.000382, Avg Loss of last 100 steps = 0.001139 (Best: 0.000366, Patience: 769/5000)
    Step 38000: Loss = 0.000378, Avg Loss of last 100 steps = 0.000472 (Best: 0.000366, Patience: 1769/5000)
    Step 39000: Loss = 0.000504, Avg Loss of last 100 steps = 0.000961 (Best: 0.000354, Patience: 295/5000)
    Step 40000: Loss = 0.000378, Avg Loss of last 100 steps = 0.000702 (Best: 0.000354, Patience: 1295/5000)
    Step 41000: Loss = 0.000836, Avg Loss of last 100 steps = 0.000658 (Best: 0.000354, Patience: 2295/5000)
    Step 42000: Loss = 0.000985, Avg Loss of last 100 steps = 0.000699 (Best: 0.000354, Patience: 3295/5000)
    Step 43000: Loss = 0.000602, Avg Loss of last 100 steps = 0.000845 (Best: 0.000351, Patience: 667/5000)
    Step 44000: Loss = 0.001011, Avg Loss of last 100 steps = 0.000536 (Best: 0.000351, Patience: 1667/5000)
    Step 45000: Loss = 0.000470, Avg Loss of last 100 steps = 0.000977 (Best: 0.000345, Patience: 724/5000)
    Step 46000: Loss = 0.000393, Avg Loss of last 100 steps = 0.000528 (Best: 0.000345, Patience: 1724/5000)
    Step 47000: Loss = 0.000521, Avg Loss of last 100 steps = 0.000775 (Best: 0.000344, Patience: 345/5000)
    Step 48000: Loss = 0.000352, Avg Loss of last 100 steps = 0.000876 (Best: 0.000344, Patience: 1345/5000)
    Step 49000: Loss = 0.000349, Avg Loss of last 100 steps = 0.001452 (Best: 0.000339, Patience: 206/5000)
    Step 50000: Loss = 0.000288, Avg Loss of last 100 steps = 0.001059 (Best: 0.000329, Patience: 485/5000)
    Step 51000: Loss = 0.000424, Avg Loss of last 100 steps = 0.000369 (Best: 0.000329, Patience: 1485/5000)
    Step 52000: Loss = 0.000821, Avg Loss of last 100 steps = 0.000642 (Best: 0.000329, Patience: 2485/5000)
    Step 53000: Loss = 0.002832, Avg Loss of last 100 steps = 0.000417 (Best: 0.000329, Patience: 3485/5000)
    Step 54000: Loss = 0.000314, Avg Loss of last 100 steps = 0.000430 (Best: 0.000293, Patience: 509/5000)
    Step 55000: Loss = 0.000430, Avg Loss of last 100 steps = 0.000506 (Best: 0.000286, Patience: 44/5000)
    Step 56000: Loss = 0.000264, Avg Loss of last 100 steps = 0.000615 (Best: 0.000271, Patience: 463/5000)
    Step 57000: Loss = 0.000406, Avg Loss of last 100 steps = 0.000910 (Best: 0.000243, Patience: 70/5000)
    Step 58000: Loss = 0.000897, Avg Loss of last 100 steps = 0.000598 (Best: 0.000238, Patience: 393/5000)
    Step 59000: Loss = 0.000273, Avg Loss of last 100 steps = 0.000631 (Best: 0.000238, Patience: 1393/5000)
    Step 60000: Loss = 0.000347, Avg Loss of last 100 steps = 0.000579 (Best: 0.000235, Patience: 450/5000)
    Step 61000: Loss = 0.000557, Avg Loss of last 100 steps = 0.000542 (Best: 0.000235, Patience: 1450/5000)
    Step 62000: Loss = 0.000504, Avg Loss of last 100 steps = 0.001028 (Best: 0.000224, Patience: 332/5000)
    Step 63000: Loss = 0.000212, Avg Loss of last 100 steps = 0.000599 (Best: 0.000211, Patience: 877/5000)
    Step 64000: Loss = 0.000507, Avg Loss of last 100 steps = 0.000790 (Best: 0.000211, Patience: 1877/5000)
    Step 65000: Loss = 0.000208, Avg Loss of last 100 steps = 0.001033 (Best: 0.000188, Patience: 232/5000)
    Step 66000: Loss = 0.000307, Avg Loss of last 100 steps = 0.000298 (Best: 0.000188, Patience: 1232/5000)
    Step 67000: Loss = 0.000229, Avg Loss of last 100 steps = 0.000544 (Best: 0.000188, Patience: 2232/5000)
    Step 68000: Loss = 0.000204, Avg Loss of last 100 steps = 0.000868 (Best: 0.000188, Patience: 3232/5000)
    Step 69000: Loss = 0.000177, Avg Loss of last 100 steps = 0.000381 (Best: 0.000163, Patience: 381/5000)
    Step 70000: Loss = 0.000187, Avg Loss of last 100 steps = 0.000800 (Best: 0.000163, Patience: 1381/5000)
    Step 71000: Loss = 0.000145, Avg Loss of last 100 steps = 0.000402 (Best: 0.000155, Patience: 458/5000)
    Step 72000: Loss = 0.000146, Avg Loss of last 100 steps = 0.000518 (Best: 0.000155, Patience: 1458/5000)
    Step 73000: Loss = 0.000251, Avg Loss of last 100 steps = 0.000771 (Best: 0.000144, Patience: 643/5000)
    Step 74000: Loss = 0.000713, Avg Loss of last 100 steps = 0.000511 (Best: 0.000144, Patience: 1643/5000)
    Step 75000: Loss = 0.000210, Avg Loss of last 100 steps = 0.000617 (Best: 0.000144, Patience: 2643/5000)
    Step 76000: Loss = 0.000391, Avg Loss of last 100 steps = 0.000430 (Best: 0.000138, Patience: 234/5000)
    Step 77000: Loss = 0.000126, Avg Loss of last 100 steps = 0.000518 (Best: 0.000138, Patience: 1234/5000)
    Step 78000: Loss = 0.000134, Avg Loss of last 100 steps = 0.000485 (Best: 0.000138, Patience: 2234/5000)
    Step 79000: Loss = 0.001871, Avg Loss of last 100 steps = 0.000368 (Best: 0.000112, Patience: 666/5000)
    Step 80000: Loss = 0.000094, Avg Loss of last 100 steps = 0.000478 (Best: 0.000095, Patience: 304/5000)
    Step 81000: Loss = 0.000176, Avg Loss of last 100 steps = 0.000414 (Best: 0.000095, Patience: 1304/5000)
    Step 82000: Loss = 0.000447, Avg Loss of last 100 steps = 0.000522 (Best: 0.000095, Patience: 2304/5000)
    Step 83000: Loss = 0.000139, Avg Loss of last 100 steps = 0.000116 (Best: 0.000095, Patience: 3304/5000)
    Step 84000: Loss = 0.000456, Avg Loss of last 100 steps = 0.000502 (Best: 0.000095, Patience: 4304/5000)
    Step 85000: Loss = 0.000079, Avg Loss of last 100 steps = 0.000547 (Best: 0.000088, Patience: 395/5000)
    Step 86000: Loss = 0.000086, Avg Loss of last 100 steps = 0.000149 (Best: 0.000088, Patience: 1395/5000)
    Step 87000: Loss = 0.000080, Avg Loss of last 100 steps = 0.001098 (Best: 0.000083, Patience: 627/5000)
    Step 88000: Loss = 0.000454, Avg Loss of last 100 steps = 0.000582 (Best: 0.000083, Patience: 1627/5000)
    Step 89000: Loss = 0.000216, Avg Loss of last 100 steps = 0.000717 (Best: 0.000083, Patience: 2627/5000)
    Step 90000: Loss = 0.000301, Avg Loss of last 100 steps = 0.000294 (Best: 0.000077, Patience: 566/5000)
    Step 91000: Loss = 0.000085, Avg Loss of last 100 steps = 0.000123 (Best: 0.000077, Patience: 1566/5000)
    Step 92000: Loss = 0.000258, Avg Loss of last 100 steps = 0.000288 (Best: 0.000077, Patience: 2566/5000)
    Step 93000: Loss = 0.002003, Avg Loss of last 100 steps = 0.000277 (Best: 0.000077, Patience: 3566/5000)
    Step 94000: Loss = 0.000703, Avg Loss of last 100 steps = 0.000106 (Best: 0.000077, Patience: 4566/5000)
    
    Early stopping at step 94434
    Best loss: 0.000077, Current loss: 0.001368
    


    
![png](temp_files/temp_18_1.png)
    



    
![png](temp_files/temp_18_2.png)
    



    
![png](temp_files/temp_18_3.png)
    


    
    ======================================================================
      ξ = 0.1
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.020092   0.046864      0.046326    0.000010  78331
       Deeper Network 0.006958   0.018811      0.015695    0.000040  53719
        Wider Network 0.096459   0.214783      0.214751    0.000502  91929
    Weighted loss ICs 0.063816   0.160104      0.149360    0.001368  94435
    
    ======================================================================
      ξ = 0.25
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.000975   0.002510      0.003174    0.000010  78331
       Deeper Network 0.003228   0.007698      0.009581    0.000040  53719
        Wider Network 0.011166   0.038435      0.037905    0.000502  91929
    Weighted loss ICs 0.005406   0.010775      0.016085    0.001368  94435
    
    ======================================================================
      ξ = 0.4
    ======================================================================
                Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
        Standard Tanh 0.001134   0.003203      0.004233    0.000010  78331
       Deeper Network 0.002819   0.007685      0.010543    0.000040  53719
        Wider Network 0.009084   0.033834      0.038767    0.000502  91929
    Weighted loss ICs 0.005963   0.015780      0.022140    0.001368  94435
    
    

I notice now that from step 40000 I start to dont have a lot of decrease in the loss function for the first model, but the deeper model is still decresing the loss function, as expected, since is deeper we can expect he have more activation function so he can simulate better complex functions.
The wider model stopped to decrese the loss function after 10000/20000 steps because the Universal Approximation Theorem state that in principle a wider neural network can approximate the function but only if is infinitly wide.
We can also see how the loss became a lot more wobbly, sign that the batch size increase wasn't enough, we should probably implement a dropout or a normalization or decrease the learning rate because it's to aggressive. 

# BONUS EXPERIMENT
By reading the documentation I then realized there is an extra model to try, I quote:
>He et al (2020) propose a two-step training approach in which the loss function is minimized first by the Adam algorithm with a predefined stop condition, then by the L-BFGS-B optimizer. According to the aforementioned paper, for cases with a little amount of training data and/or residual points, L-BFGS-B, performs better with a faster rate of convergence and reduced computing cost.
 >— *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.3, Page 24*



```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# hibrid training loop
def train_pinn_hybrid(model, steps=5000, patience=5000, min_delta=1e-6):

    # I first optimize via Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)

    # initial conditions targets (specified in the test on the ML4SC website)
    x0_target = 0.7
    v0_target = 1.2
    
    # windows for the average loss calculation
    window = 100
    loss_window = []
    
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0
    
    print("start training (with Adam)...")

    for step in range(steps):
        optimizer_adam.zero_grad()
        
        # sample random points for z and xi in given domains (specified in the test on the ML4SC website)
        batch_size = 1000
        z_col = torch.rand(batch_size, 1) * 20.0
        xi_col = torch.rand(batch_size, 1) * (0.4 - 0.1) + 0.1
        # at each of this random points we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col, xi_col)
        
        # initial condition loss (boundary loss)
        # Sample random xi for the boundary condition
        xi_bc = torch.rand(batch_size // 4, 1) * (0.4 - 0.1) + 0.1  # we pick a random xi again, no matter what xi I took, the oscillator always start in the same spot
        z_bc = torch.zeros_like(xi_bc)  # z is always 0 here
        z_bc.requires_grad = True  # we still need to track z if I want to track how much fast x change with z, so dx/dz

        # guess of the position at t=0:
        x_bc = model(z_bc, xi_bc)
        # then compute velocity at t=0:
        dx_bc = torch.autograd.grad(x_bc, z_bc, torch.ones_like(x_bc), create_graph=True)[0]
        
        # then calculate the penalty for position and velocity:
        loss_ic_x = torch.mean((x_bc - x0_target)**2)
        loss_ic_v = torch.mean((dx_bc - v0_target)**2)
        
                                                    # total loss, infact without the penalty for position and velocity
        loss = loss_physics + loss_ic_x + loss_ic_v # I would have that x=0 and v=0 for every time step minimize the loss and  
                                                    # so is the solution but that cannot be since of ICs different from zero

        loss.backward()         
        optimizer_adam.step()   

        current_loss = loss.item()
        loss_history.append(current_loss)
        
        # early stopping criteria: if the average loss doesnt improve at least of min_delta, for a number of consecutive steps equal to the patience, it stops early.
        loss_window.append(current_loss)
        if len(loss_window) > window:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)
        
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at step {step}")
            print(f"Best avg loss: {best_loss:.6f}, Current avg loss: {avg_loss:.6f}")
            break
        
        if step % 1000 == 0:
            print(f"Step {step}: Loss = {current_loss:.6f}, Avg Loss of last 100 steps = {avg_loss:.6f} (Best: {best_loss:.6f}, Patience: {patience_counter}/{patience})")            

    # after I optimized with Adam I do a fine tuning with L-BFGS
    print("continue training (with L-BFGS Fine-tuning)...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), 
                                  lr=1.0, 
                                  history_size=50, 
                                  max_iter=5000, 
                                  line_search_fn="strong_wolfe")
    
    # second sample, this sample is not in a loop, otherwise the LBFGS would get
    # confuse by the change of the distribution of points when I change steps 
    batch_size_lbfgs = 2000
    z_col_fixed = torch.rand(batch_size_lbfgs, 1) * 20.0
    xi_col_fixed = torch.rand(batch_size_lbfgs, 1) * (0.4 - 0.1) + 0.1
    
    xi_bc_fixed = torch.rand(batch_size_lbfgs // 4, 1) * (0.4 - 0.1) + 0.1
    z_bc_fixed = torch.zeros_like(xi_bc_fixed)
    z_bc_fixed.requires_grad = True 

    def closure():
        optimizer_lbfgs.zero_grad()
        
        # at each of this random points (NOW FIXED) we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col_fixed, xi_col_fixed)
        
        # guess of the position at t=0:
        x_bc = model(z_bc_fixed, xi_bc_fixed)
        # then compute velocity at t=0:
        dx_bc = torch.autograd.grad(x_bc, z_bc_fixed, torch.ones_like(x_bc), create_graph=True)[0]
        
        # then calculate the penalty for position and velocity:
        loss_ic_x = torch.mean((x_bc - x0_target)**2)
        loss_ic_v = torch.mean((dx_bc - v0_target)**2)        

                                                          # total loss, infact without the penalty for position and velocity
        total_loss = loss_physics + loss_ic_x + loss_ic_v # I would have that x=0 and v=0 for every time step minimize the loss and  
                                                          # so is the solution but that cannot be since of ICs different from zero

        total_loss.backward()
        
        # Append to history for plotting
        loss_history.append(total_loss.item())
        return total_loss

    optimizer_lbfgs.step(closure)
    print("training complete.")
    return loss_history
    
def summary_tables_single(model, history, model_name="Adam + L-BFGS PINN", test_xis=[0.1, 0.25, 0.4]):
    z_test = torch.linspace(0, 20, 500).view(-1, 1)
    for xi_val in test_xis:
        xi_test = torch.full_like(z_test, xi_val)
        exact = analytical_solution(z_test.numpy(), xi_val)
        with torch.no_grad():
            pred = model(z_test, xi_test).numpy()
        err = np.abs(exact - pred)
        df = pd.DataFrame([{
            "Model": model_name,
            "MAE": np.mean(err),
            "Max Error": np.max(err),
            "L2 Rel Error": np.sqrt(np.sum((exact - pred)**2) / np.sum(exact**2)),
            "Final Loss": history[-1],
            "Steps": len(history)
        }])
        print(f"\n{'='*70}")
        print(f"  ξ = {xi_val}")
        print(f"{'='*70}")
        print(df.to_string(index=False, float_format="%.6f"))
    print()


# initializzation and training via class calling
pinn = OscillatorPINN()
history = train_pinn_hybrid(pinn) 

plot_predictions(pinn, "Adam + L-BFGS PINN")
plot_loss(history)
plot_error(pinn, "Adam + L-BFGS PINN")
summary_tables_single(pinn, history) # make a summary table where I have some metrics as max, mar, L2 
```

    start training (with Adam)...
    Step 0: Loss = 1.822458, Avg Loss of last 100 steps = 1.822458 (Best: 1.822458, Patience: 0/5000)
    Step 1000: Loss = 0.003223, Avg Loss of last 100 steps = 0.003805 (Best: 0.003805, Patience: 0/5000)
    Step 2000: Loss = 0.001191, Avg Loss of last 100 steps = 0.001416 (Best: 0.001396, Patience: 15/5000)
    Step 3000: Loss = 0.000891, Avg Loss of last 100 steps = 0.001105 (Best: 0.001017, Patience: 123/5000)
    Step 4000: Loss = 0.000909, Avg Loss of last 100 steps = 0.000719 (Best: 0.000690, Patience: 56/5000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    


    
![png](temp_files/temp_21_1.png)
    



    
![png](temp_files/temp_21_2.png)
    



    
![png](temp_files/temp_21_3.png)
    


    
    ======================================================================
      ξ = 0.1
    ======================================================================
                 Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
    Adam + L-BFGS PINN 0.010888   0.033974      0.028191    0.000004   8023
    
    ======================================================================
      ξ = 0.25
    ======================================================================
                 Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
    Adam + L-BFGS PINN 0.001139   0.003365      0.003824    0.000004   8023
    
    ======================================================================
      ξ = 0.4
    ======================================================================
                 Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
    Adam + L-BFGS PINN 0.002155   0.004945      0.007675    0.000004   8023
    
    

what we did before this last experiment is not wasted: we can try to adapt the architecture found in the previous experiments to this mixed ADAM and L-BFGS-B way of optimizing the loss.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# hibrid training loop
def train_hybrid(model_name, activation, lambda_ic=1.0, hidden_dim=50, layers=3, adam_steps=5000, adam_lr=1e-3, batch_size=1000, lbfgs_max_iter=5000, lbfgs_batch_size=2000, patience=2000, min_delta=1e-6):

    print(f"\n{'='*70}")
    print(f"  Hybrid Adam + L-BFGS Experiment: {model_name}")
    print(f"{'='*70}")

    model = BasePINN(activation=activation, hidden_dim=hidden_dim, hidden_layers=layers)
    
    # initial conditions targets (specified in the test on the ML4SC website)
    x0_target = 0.7
    v0_target = 1.2


    # windows for the average loss calculation
    window = 100
    loss_window = []
    
    loss_history = []
    best_loss = float('inf')
    patience_counter = 0

    # I first optimize via Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=adam_lr)

    print("start training (with Adam)...")

    for step in range(adam_steps):
        optimizer_adam.zero_grad()

        # sample random points for z and xi in given domains (specified in the test on the ML4SC website)
        z_col = torch.rand(batch_size, 1) * 20.0
        xi_col = torch.rand(batch_size, 1) * (0.4 - 0.1) + 0.1
        # at each of this random points we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col, xi_col)

        # initial condition loss (boundary loss)
        # Sample random xi for the boundary condition
        xi_bc = torch.rand(batch_size // 4, 1) * (0.4 - 0.1) + 0.1  # we pick a random xi again, no matter what xi I took, the oscillator always start in the same spot
        z_bc = torch.zeros_like(xi_bc)  # z is always 0 here
        z_bc.requires_grad = True  # we still need to track z if I want to track how much fast x change with z, so dx/dz
        
        # guess of the position at t=0:
        x_bc = model(z_bc, xi_bc)

        # then compute the velocity at t=0:
        dx_bc = torch.autograd.grad(
            x_bc, z_bc, 
            torch.ones_like(x_bc), 
            create_graph=True
        )[0]

        # than calculate the penalty for position and velocity:
        loss_ic_val = torch.mean((x_bc - x0_target)**2) + torch.mean((dx_bc - v0_target)**2)
        
                                                        # total loss, infact without the penalty for position and velocity        
        loss = loss_physics + (lambda_ic * loss_ic_val) # I would have that x=0 and v=0 for every time step minimize the loss and 
                                                        # so is the solution but that cannot be since of ICs different from zero       
        loss.backward()
        optimizer_adam.step()

        current_loss = loss.item()
        loss_history.append(loss.item())

        # early stopping criteria: if the average loss doesnt improve at least of min_delta, for a number of consecutive steps equal to the patience, it stops early.
        loss_window.append(current_loss)
        if len(loss_window) > window:
            loss_window.pop(0)
        avg_loss = sum(loss_window) / len(loss_window)
        
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at step {step}")
            print(f"Best avg loss: {best_loss:.6f}, Current avg loss: {avg_loss:.6f}")
            break
        
        if step % 1000 == 0:
            print(f"Step {step}: Loss = {current_loss:.6f}, Avg Loss of last 100 steps = {avg_loss:.6f} (Best: {best_loss:.6f}, Patience: {patience_counter}/{patience})")            

    # after I optimized with Adam I do a fine tuning with L-BFGS
    print("continue training (with L-BFGS Fine-tuning)...")

    optimizer_lbfgs = optim.LBFGS(model.parameters(),
                                  lr=1.0,
                                  history_size=50,
                                  max_iter=lbfgs_max_iter,
                                  line_search_fn="strong_wolfe")
    
    # second sample, this sample is not in a loop, otherwise the LBFGS would get
    # confuse by the change of the distribution of points when I change steps 
    batch_size_lbfgs = 2000
    z_col_fixed = torch.rand(batch_size_lbfgs, 1) * 20.0
    xi_col_fixed = torch.rand(batch_size_lbfgs, 1) * (0.4 - 0.1) + 0.1
    
    xi_bc_fixed = torch.rand(batch_size_lbfgs // 4, 1) * (0.4 - 0.1) + 0.1
    z_bc_fixed = torch.zeros_like(xi_bc_fixed)
    z_bc_fixed.requires_grad = True 


    lbfgs_iter = [0]  # mutable counter for closure

    def closure():
        optimizer_lbfgs.zero_grad()

        # at each of this random points (NOW FIXED) we calculate the loss/physics violation
        loss_physics = physics_loss(model, z_col_fixed, xi_col_fixed)

        # guess of the position at t=0:
        x_bc = model(z_bc_fixed, xi_bc_fixed)
        # then compute velocity at t=0:
        dx_bc = torch.autograd.grad(x_bc, z_bc_fixed, torch.ones_like(x_bc), create_graph=True)[0]

        # then calculate the penalty for position and velocity:
        loss_ic_x = torch.mean((x_bc - x0_target)**2)
        loss_ic_v = torch.mean((dx_bc - v0_target)**2)        

                                                          # total loss, infact without the penalty for position and velocity
        total_loss = loss_physics + loss_ic_x + loss_ic_v # I would have that x=0 and v=0 for every time step minimize the loss and  
                                                          # so is the solution but that cannot be since of ICs different from zero

        total_loss.backward()
        
        # Append to history for plotting
        loss_history.append(total_loss.item())
        return total_loss

    optimizer_lbfgs.step(closure)
    print("training complete.")
    return model, loss_history


# define the experiments
hybrid_experiments = [
    {"name": "Standard Tanh (Adam+L-BFGS)", "act": nn.Tanh, "lambda": 1.0, "dim": 50, "layers": 3, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "Deeper Network (Adam+L-BFGS)", "act": nn.Tanh, "lambda": 1.0, "dim": 50, "layers": 6, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "Wider Network (Adam+L-BFGS)", "act": nn.Tanh, "lambda": 1.0, "dim": 200, "layers": 1, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "Weighted loss ICs", "act": nn.Tanh, "lambda": 10.0, "dim": 50, "layers": 3, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000}
]


def run_hybrid_experiments(experiments):
    results = {}
    for exp in experiments:
        model, hist = train_hybrid(
            model_name=exp["name"],
            activation=exp["act"],
            lambda_ic=exp["lambda"],
            hidden_dim=exp["dim"],
            layers=exp["layers"],
            adam_steps=exp["adam_steps"],
            adam_lr=exp["adam_lr"],
            batch_size=exp["bs"],
            lbfgs_max_iter=exp["lbfgs_max_iter"],
            lbfgs_batch_size=exp["lbfgs_bs"]
        )
        results[exp["name"]] = {"model": model, "hist": hist}
    return results

 
hybrid_results = run_hybrid_experiments(hybrid_experiments) # initializzation and training
plot_all_predictions(hybrid_results) # plotting results for specific xi values
plot_all_losses(hybrid_results) # plot loss curve
plot_all_errors(hybrid_results) # plot the error analysis
summary_tables(hybrid_results) # make a summary table where I have some metrics as max, mar, L2 

```

    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: Standard Tanh (Adam+L-BFGS)
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 1.822458, Avg Loss of last 100 steps = 1.822458 (Best: 1.822458, Patience: 0/2000)
    Step 1000: Loss = 0.003223, Avg Loss of last 100 steps = 0.003805 (Best: 0.003805, Patience: 0/2000)
    Step 2000: Loss = 0.001191, Avg Loss of last 100 steps = 0.001416 (Best: 0.001396, Patience: 15/2000)
    Step 3000: Loss = 0.000891, Avg Loss of last 100 steps = 0.001105 (Best: 0.001017, Patience: 123/2000)
    Step 4000: Loss = 0.000909, Avg Loss of last 100 steps = 0.000719 (Best: 0.000690, Patience: 56/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: Deeper Network (Adam+L-BFGS)
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 1.800887, Avg Loss of last 100 steps = 1.800887 (Best: 1.800887, Patience: 0/2000)
    Step 1000: Loss = 0.005521, Avg Loss of last 100 steps = 0.004769 (Best: 0.004599, Patience: 51/2000)
    Step 2000: Loss = 0.004379, Avg Loss of last 100 steps = 0.003156 (Best: 0.002930, Patience: 29/2000)
    Step 3000: Loss = 0.001622, Avg Loss of last 100 steps = 0.002003 (Best: 0.001711, Patience: 344/2000)
    Step 4000: Loss = 0.001889, Avg Loss of last 100 steps = 0.001429 (Best: 0.001429, Patience: 0/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: Wider Network (Adam+L-BFGS)
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 1.769087, Avg Loss of last 100 steps = 1.769087 (Best: 1.769087, Patience: 0/2000)
    Step 1000: Loss = 0.744621, Avg Loss of last 100 steps = 0.760122 (Best: 0.760122, Patience: 0/2000)
    Step 2000: Loss = 0.157984, Avg Loss of last 100 steps = 0.185207 (Best: 0.185207, Patience: 0/2000)
    Step 3000: Loss = 0.072484, Avg Loss of last 100 steps = 0.071255 (Best: 0.071255, Patience: 0/2000)
    Step 4000: Loss = 0.024462, Avg Loss of last 100 steps = 0.028192 (Best: 0.028192, Patience: 0/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: Weighted loss ICs
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 18.199539, Avg Loss of last 100 steps = 18.199539 (Best: 18.199539, Patience: 0/2000)
    Step 1000: Loss = 0.028534, Avg Loss of last 100 steps = 0.028983 (Best: 0.028983, Patience: 0/2000)
    Step 2000: Loss = 0.005950, Avg Loss of last 100 steps = 0.007279 (Best: 0.007279, Patience: 0/2000)
    Step 3000: Loss = 0.004091, Avg Loss of last 100 steps = 0.005270 (Best: 0.005191, Patience: 133/2000)
    Step 4000: Loss = 0.003954, Avg Loss of last 100 steps = 0.004083 (Best: 0.004083, Patience: 0/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    


    
![png](temp_files/temp_23_1.png)
    



    
![png](temp_files/temp_23_2.png)
    



    
![png](temp_files/temp_23_3.png)
    


    
    ======================================================================
      ξ = 0.1
    ======================================================================
                           Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Standard Tanh (Adam+L-BFGS) 0.010888   0.033974      0.028191    0.000004   8023
    Deeper Network (Adam+L-BFGS) 0.041372   0.097072      0.098695    0.000038   7461
     Wider Network (Adam+L-BFGS) 0.059199   0.130843      0.135921    0.000063   7994
               Weighted loss ICs 0.013102   0.039137      0.031077    0.000007   8193
    
    ======================================================================
      ξ = 0.25
    ======================================================================
                           Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Standard Tanh (Adam+L-BFGS) 0.001139   0.003365      0.003824    0.000004   8023
    Deeper Network (Adam+L-BFGS) 0.002805   0.011017      0.010159    0.000038   7461
     Wider Network (Adam+L-BFGS) 0.005454   0.014598      0.017500    0.000063   7994
               Weighted loss ICs 0.001383   0.003562      0.004401    0.000007   8193
    
    ======================================================================
      ξ = 0.4
    ======================================================================
                           Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Standard Tanh (Adam+L-BFGS) 0.002155   0.004945      0.007675    0.000004   8023
    Deeper Network (Adam+L-BFGS) 0.002839   0.014947      0.014155    0.000038   7461
     Wider Network (Adam+L-BFGS) 0.004991   0.015705      0.018800    0.000063   7994
               Weighted loss ICs 0.003327   0.009138      0.012419    0.000007   8193
    
    

other possible models can be modification on the number of layers and neuron, I quote and hightlight:
> Another example of how the differential problem affects network architecture can be found
> **in Kharazmi et al (2021b) for their hp-VPINN. The architecture is implemented with four layers and twenty neurons per layer**, but for an advection equation with a double discontinuity of the exact solution, they use an eight-layered deep network. For a
constrained approach, by utilizing a specific portion of the NN to satisfy the
required boundary conditions,
> **Zhu et al (2021) use five hidden layers and 250 neurons per layer to constitute the fully connected neural network**. Bringing
the number of layers higher, **in PINNeik (Waheed et al, 2021), a DNN with ten
hidden layers containing twenty neurons each is utilized**, with a locally adaptive inverse tangent function as the activation function for all hidden layers
except the final layer, which has a linear activation function.
> **He et al (2020) examines the effect of neural network size on state estimation accuracy. They begin by experimenting with various hidden layer sizes ranging from three to five, while maintaining a value of 32 neurons per layer**. Then they set the number of hidden layers to three, the activation function to hyperbolic tangent,
while varying the number of neurons in each hidden layer. 
>
> — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.1, Pages 12*

let's try those architecture with the old pipeline with only Adam optimized and the new pipeline with best optimizer so far that we have founded (that is ADAM + L-BFGS):


```python
# define the experiments
experiments = [
    {"name": "Kharazmi Deep Network", "act": nn.Tanh, "lambda": 1.0, "dim": 20, "layers": 4, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000},
    # {"name": "ReLU", "act": nn.ReLU, "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 2000, "bs":1000},
    {"name": "Zhu Deeper Network", "act": nn.Tanh, "lambda": 1.0, "dim": 250, "layers": 5, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000},
    {"name": "He 3 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32, "layers": 3, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000},
    {"name": "He 4 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32, "layers": 4, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000},
    {"name": "He 5 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32, "layers": 5, "number of steps": 100000, "bs":1000, "lr":1e-3, "patience":5000}
]

results = run_experiments(experiments) # initializzation and training
plot_all_predictions(results) # plotting results for specific xi values
plot_all_losses(results) # plot loss curve
plot_all_errors(results) # plot the error analysis
summary_tables(results) # make a summary table where I have some metrics as max, mar, L2 
```

    --- starting Experiment: Kharazmi Deep Network (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.850432 (Best: 1.850432, Patience: 0/10000)
    Step 1000: Loss = 0.005505 (Best: 0.004728, Patience: 29/10000)
    Step 2000: Loss = 0.001932 (Best: 0.001701, Patience: 14/10000)
    Step 3000: Loss = 0.001341 (Best: 0.001083, Patience: 103/10000)
    Step 4000: Loss = 0.001089 (Best: 0.000835, Patience: 103/10000)
    Step 5000: Loss = 0.001222 (Best: 0.000815, Patience: 464/10000)
    Step 6000: Loss = 0.002040 (Best: 0.000746, Patience: 796/10000)
    Step 7000: Loss = 0.001036 (Best: 0.000740, Patience: 564/10000)
    Step 8000: Loss = 0.001064 (Best: 0.000695, Patience: 143/10000)
    Step 9000: Loss = 0.000913 (Best: 0.000614, Patience: 508/10000)
    Step 10000: Loss = 0.000856 (Best: 0.000608, Patience: 83/10000)
    Step 11000: Loss = 0.000868 (Best: 0.000569, Patience: 84/10000)
    Step 12000: Loss = 0.000868 (Best: 0.000569, Patience: 1084/10000)
    Step 13000: Loss = 0.000630 (Best: 0.000563, Patience: 499/10000)
    Step 14000: Loss = 0.000865 (Best: 0.000533, Patience: 102/10000)
    Step 15000: Loss = 0.000699 (Best: 0.000533, Patience: 1102/10000)
    Step 16000: Loss = 0.000699 (Best: 0.000533, Patience: 2102/10000)
    Step 17000: Loss = 0.000771 (Best: 0.000533, Patience: 3102/10000)
    Step 18000: Loss = 0.000713 (Best: 0.000533, Patience: 4102/10000)
    Step 19000: Loss = 0.001178 (Best: 0.000503, Patience: 978/10000)
    Step 20000: Loss = 0.000742 (Best: 0.000503, Patience: 1978/10000)
    Step 21000: Loss = 0.000792 (Best: 0.000465, Patience: 579/10000)
    Step 22000: Loss = 0.000845 (Best: 0.000465, Patience: 1579/10000)
    Step 23000: Loss = 0.000782 (Best: 0.000465, Patience: 2579/10000)
    Step 24000: Loss = 0.000630 (Best: 0.000465, Patience: 3579/10000)
    Step 25000: Loss = 0.000812 (Best: 0.000465, Patience: 4579/10000)
    Step 26000: Loss = 0.000613 (Best: 0.000457, Patience: 196/10000)
    Step 27000: Loss = 0.000655 (Best: 0.000457, Patience: 1196/10000)
    Step 28000: Loss = 0.000737 (Best: 0.000451, Patience: 217/10000)
    Step 29000: Loss = 0.001142 (Best: 0.000451, Patience: 1217/10000)
    Step 30000: Loss = 0.000698 (Best: 0.000442, Patience: 511/10000)
    Step 31000: Loss = 0.000613 (Best: 0.000427, Patience: 321/10000)
    Step 32000: Loss = 0.000679 (Best: 0.000427, Patience: 1321/10000)
    Step 33000: Loss = 0.000649 (Best: 0.000427, Patience: 2321/10000)
    Step 34000: Loss = 0.000630 (Best: 0.000427, Patience: 3321/10000)
    Step 35000: Loss = 0.000602 (Best: 0.000427, Patience: 4321/10000)
    Step 36000: Loss = 0.000827 (Best: 0.000427, Patience: 5321/10000)
    Step 37000: Loss = 0.000559 (Best: 0.000427, Patience: 6321/10000)
    Step 38000: Loss = 0.000778 (Best: 0.000427, Patience: 7321/10000)
    Step 39000: Loss = 0.000657 (Best: 0.000427, Patience: 8321/10000)
    Step 40000: Loss = 0.000851 (Best: 0.000397, Patience: 195/10000)
    Step 41000: Loss = 0.000584 (Best: 0.000397, Patience: 1195/10000)
    Step 42000: Loss = 0.000562 (Best: 0.000397, Patience: 2195/10000)
    Step 43000: Loss = 0.000537 (Best: 0.000397, Patience: 3195/10000)
    Step 44000: Loss = 0.000474 (Best: 0.000376, Patience: 569/10000)
    Step 45000: Loss = 0.000649 (Best: 0.000365, Patience: 146/10000)
    Step 46000: Loss = 0.000872 (Best: 0.000365, Patience: 1146/10000)
    Step 47000: Loss = 0.000517 (Best: 0.000344, Patience: 397/10000)
    Step 48000: Loss = 0.000436 (Best: 0.000331, Patience: 593/10000)
    Step 49000: Loss = 0.000586 (Best: 0.000331, Patience: 1593/10000)
    Step 50000: Loss = 0.000439 (Best: 0.000331, Patience: 2593/10000)
    Step 51000: Loss = 0.000546 (Best: 0.000325, Patience: 608/10000)
    Step 52000: Loss = 0.000500 (Best: 0.000325, Patience: 1608/10000)
    Step 53000: Loss = 0.000468 (Best: 0.000321, Patience: 341/10000)
    Step 54000: Loss = 0.000429 (Best: 0.000314, Patience: 308/10000)
    Step 55000: Loss = 0.000438 (Best: 0.000314, Patience: 1308/10000)
    Step 56000: Loss = 0.000446 (Best: 0.000281, Patience: 770/10000)
    Step 57000: Loss = 0.000615 (Best: 0.000262, Patience: 948/10000)
    Step 58000: Loss = 0.000474 (Best: 0.000262, Patience: 1948/10000)
    Step 59000: Loss = 0.000758 (Best: 0.000252, Patience: 168/10000)
    Step 60000: Loss = 0.000331 (Best: 0.000241, Patience: 906/10000)
    Step 61000: Loss = 0.000416 (Best: 0.000241, Patience: 1906/10000)
    Step 62000: Loss = 0.000253 (Best: 0.000225, Patience: 199/10000)
    Step 63000: Loss = 0.000425 (Best: 0.000225, Patience: 1199/10000)
    Step 64000: Loss = 0.000235 (Best: 0.000213, Patience: 521/10000)
    Step 65000: Loss = 0.000330 (Best: 0.000203, Patience: 624/10000)
    Step 66000: Loss = 0.000321 (Best: 0.000188, Patience: 332/10000)
    Step 67000: Loss = 0.000343 (Best: 0.000177, Patience: 354/10000)
    Step 68000: Loss = 0.000232 (Best: 0.000164, Patience: 98/10000)
    Step 69000: Loss = 0.000270 (Best: 0.000159, Patience: 709/10000)
    Step 70000: Loss = 0.000327 (Best: 0.000142, Patience: 47/10000)
    Step 71000: Loss = 0.000182 (Best: 0.000138, Patience: 343/10000)
    Step 72000: Loss = 0.000210 (Best: 0.000128, Patience: 232/10000)
    Step 73000: Loss = 0.000282 (Best: 0.000127, Patience: 160/10000)
    Step 74000: Loss = 0.000223 (Best: 0.000120, Patience: 165/10000)
    Step 75000: Loss = 0.000143 (Best: 0.000120, Patience: 1165/10000)
    Step 76000: Loss = 0.000149 (Best: 0.000103, Patience: 541/10000)
    Step 77000: Loss = 0.000134 (Best: 0.000102, Patience: 5/10000)
    Step 78000: Loss = 0.000158 (Best: 0.000101, Patience: 339/10000)
    Step 79000: Loss = 0.000147 (Best: 0.000101, Patience: 1339/10000)
    Step 80000: Loss = 0.000150 (Best: 0.000101, Patience: 2339/10000)
    Step 81000: Loss = 0.000198 (Best: 0.000101, Patience: 3339/10000)
    Step 82000: Loss = 0.000120 (Best: 0.000101, Patience: 4339/10000)
    Step 83000: Loss = 0.000133 (Best: 0.000101, Patience: 5339/10000)
    Step 84000: Loss = 0.000320 (Best: 0.000094, Patience: 26/10000)
    Step 85000: Loss = 0.000192 (Best: 0.000094, Patience: 1026/10000)
    Step 86000: Loss = 0.000246 (Best: 0.000094, Patience: 2026/10000)
    Step 87000: Loss = 0.000195 (Best: 0.000090, Patience: 341/10000)
    Step 88000: Loss = 0.000254 (Best: 0.000090, Patience: 1341/10000)
    Step 89000: Loss = 0.000134 (Best: 0.000090, Patience: 2341/10000)
    Step 90000: Loss = 0.000115 (Best: 0.000090, Patience: 3341/10000)
    Step 91000: Loss = 0.000131 (Best: 0.000089, Patience: 404/10000)
    Step 92000: Loss = 0.000181 (Best: 0.000087, Patience: 517/10000)
    Step 93000: Loss = 0.000111 (Best: 0.000087, Patience: 1517/10000)
    Step 94000: Loss = 0.000117 (Best: 0.000081, Patience: 69/10000)
    Step 95000: Loss = 0.000470 (Best: 0.000081, Patience: 1069/10000)
    Step 96000: Loss = 0.000125 (Best: 0.000081, Patience: 2069/10000)
    Step 97000: Loss = 0.000107 (Best: 0.000081, Patience: 3069/10000)
    Step 98000: Loss = 0.000116 (Best: 0.000076, Patience: 343/10000)
    Step 99000: Loss = 0.000137 (Best: 0.000076, Patience: 1343/10000)
    --- starting Experiment: Zhu Deeper Network (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.996960 (Best: 1.996960, Patience: 0/10000)
    Step 1000: Loss = 0.002354 (Best: 0.001780, Patience: 7/10000)
    Step 2000: Loss = 0.003802 (Best: 0.000991, Patience: 134/10000)
    Step 3000: Loss = 0.000982 (Best: 0.000826, Patience: 2/10000)
    Step 4000: Loss = 0.000681 (Best: 0.000677, Patience: 116/10000)
    Step 5000: Loss = 0.000647 (Best: 0.000508, Patience: 473/10000)
    Step 6000: Loss = 0.003051 (Best: 0.000297, Patience: 29/10000)
    Step 7000: Loss = 0.000668 (Best: 0.000195, Patience: 65/10000)
    Step 8000: Loss = 0.000606 (Best: 0.000149, Patience: 209/10000)
    Step 9000: Loss = 0.557124 (Best: 0.000084, Patience: 108/10000)
    Step 10000: Loss = 0.000354 (Best: 0.000084, Patience: 1108/10000)
    Step 11000: Loss = 0.000197 (Best: 0.000084, Patience: 2108/10000)
    Step 12000: Loss = 0.000334 (Best: 0.000084, Patience: 3108/10000)
    Step 13000: Loss = 0.000878 (Best: 0.000079, Patience: 515/10000)
    Step 14000: Loss = 0.000379 (Best: 0.000059, Patience: 311/10000)
    Step 15000: Loss = 0.001127 (Best: 0.000059, Patience: 1311/10000)
    Step 16000: Loss = 0.000349 (Best: 0.000050, Patience: 858/10000)
    Step 17000: Loss = 0.001349 (Best: 0.000043, Patience: 896/10000)
    Step 18000: Loss = 0.000822 (Best: 0.000042, Patience: 550/10000)
    Step 19000: Loss = 0.000797 (Best: 0.000033, Patience: 824/10000)
    Step 20000: Loss = 0.000868 (Best: 0.000033, Patience: 1824/10000)
    Step 21000: Loss = 0.000317 (Best: 0.000033, Patience: 2824/10000)
    Step 22000: Loss = 0.000134 (Best: 0.000033, Patience: 3824/10000)
    Step 23000: Loss = 0.000418 (Best: 0.000033, Patience: 4824/10000)
    Step 24000: Loss = 0.001687 (Best: 0.000033, Patience: 5824/10000)
    Step 25000: Loss = 0.000094 (Best: 0.000031, Patience: 54/10000)
    Step 26000: Loss = 0.000520 (Best: 0.000027, Patience: 112/10000)
    Step 27000: Loss = 0.000527 (Best: 0.000027, Patience: 1112/10000)
    Step 28000: Loss = 0.000095 (Best: 0.000027, Patience: 2112/10000)
    Step 29000: Loss = 0.000174 (Best: 0.000020, Patience: 205/10000)
    Step 30000: Loss = 0.000138 (Best: 0.000020, Patience: 1205/10000)
    Step 31000: Loss = 0.000127 (Best: 0.000015, Patience: 773/10000)
    Step 32000: Loss = 0.000288 (Best: 0.000015, Patience: 1773/10000)
    Step 33000: Loss = 0.000904 (Best: 0.000005, Patience: 573/10000)
    Step 34000: Loss = 0.000226 (Best: 0.000005, Patience: 1573/10000)
    Step 35000: Loss = 0.000141 (Best: 0.000005, Patience: 2573/10000)
    Step 36000: Loss = 0.001537 (Best: 0.000005, Patience: 3573/10000)
    Step 37000: Loss = 0.000125 (Best: 0.000005, Patience: 4573/10000)
    Step 38000: Loss = 0.000029 (Best: 0.000005, Patience: 5573/10000)
    Step 39000: Loss = 0.000137 (Best: 0.000005, Patience: 6573/10000)
    Step 40000: Loss = 0.000153 (Best: 0.000005, Patience: 7573/10000)
    Step 41000: Loss = 0.000182 (Best: 0.000005, Patience: 8573/10000)
    Step 42000: Loss = 0.000160 (Best: 0.000005, Patience: 9573/10000)
    
    Early stopping at step 42427
    Best loss: 0.000005, Current loss: 0.000067
    --- starting Experiment: He 3 layer, 32 neurons (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.961239 (Best: 1.961239, Patience: 0/10000)
    Step 1000: Loss = 0.005354 (Best: 0.004785, Patience: 2/10000)
    Step 2000: Loss = 0.002818 (Best: 0.002056, Patience: 57/10000)
    Step 3000: Loss = 0.001830 (Best: 0.001493, Patience: 62/10000)
    Step 4000: Loss = 0.001260 (Best: 0.000973, Patience: 4/10000)
    Step 5000: Loss = 0.000871 (Best: 0.000666, Patience: 161/10000)
    Step 6000: Loss = 0.000755 (Best: 0.000490, Patience: 224/10000)
    Step 7000: Loss = 0.000759 (Best: 0.000412, Patience: 31/10000)
    Step 8000: Loss = 0.000430 (Best: 0.000342, Patience: 219/10000)
    Step 9000: Loss = 0.000407 (Best: 0.000314, Patience: 63/10000)
    Step 10000: Loss = 0.000490 (Best: 0.000314, Patience: 1063/10000)
    Step 11000: Loss = 0.000372 (Best: 0.000275, Patience: 747/10000)
    Step 12000: Loss = 0.000347 (Best: 0.000275, Patience: 1747/10000)
    Step 13000: Loss = 0.000363 (Best: 0.000250, Patience: 161/10000)
    Step 14000: Loss = 0.000297 (Best: 0.000218, Patience: 466/10000)
    Step 15000: Loss = 0.000256 (Best: 0.000212, Patience: 311/10000)
    Step 16000: Loss = 0.000212 (Best: 0.000167, Patience: 6/10000)
    Step 17000: Loss = 0.000228 (Best: 0.000164, Patience: 248/10000)
    Step 18000: Loss = 0.000196 (Best: 0.000141, Patience: 361/10000)
    Step 19000: Loss = 0.000164 (Best: 0.000127, Patience: 233/10000)
    Step 20000: Loss = 0.000170 (Best: 0.000117, Patience: 58/10000)
    Step 21000: Loss = 0.000526 (Best: 0.000099, Patience: 117/10000)
    Step 22000: Loss = 0.000186 (Best: 0.000092, Patience: 95/10000)
    Step 23000: Loss = 0.000159 (Best: 0.000089, Patience: 624/10000)
    Step 24000: Loss = 0.000133 (Best: 0.000078, Patience: 134/10000)
    Step 25000: Loss = 0.000094 (Best: 0.000071, Patience: 3/10000)
    Step 26000: Loss = 0.000085 (Best: 0.000063, Patience: 15/10000)
    Step 27000: Loss = 0.000101 (Best: 0.000060, Patience: 47/10000)
    Step 28000: Loss = 0.000085 (Best: 0.000054, Patience: 272/10000)
    Step 29000: Loss = 0.000101 (Best: 0.000054, Patience: 1272/10000)
    Step 30000: Loss = 0.000155 (Best: 0.000054, Patience: 2272/10000)
    Step 31000: Loss = 0.000077 (Best: 0.000051, Patience: 659/10000)
    Step 32000: Loss = 0.000138 (Best: 0.000047, Patience: 385/10000)
    Step 33000: Loss = 0.000073 (Best: 0.000047, Patience: 1385/10000)
    Step 34000: Loss = 0.000058 (Best: 0.000046, Patience: 867/10000)
    Step 35000: Loss = 0.000080 (Best: 0.000038, Patience: 46/10000)
    Step 36000: Loss = 0.000090 (Best: 0.000038, Patience: 1046/10000)
    Step 37000: Loss = 0.000051 (Best: 0.000028, Patience: 126/10000)
    Step 38000: Loss = 0.000067 (Best: 0.000028, Patience: 1126/10000)
    Step 39000: Loss = 0.000166 (Best: 0.000028, Patience: 2126/10000)
    Step 40000: Loss = 0.000047 (Best: 0.000028, Patience: 3126/10000)
    Step 41000: Loss = 0.000274 (Best: 0.000028, Patience: 4126/10000)
    Step 42000: Loss = 0.000075 (Best: 0.000028, Patience: 5126/10000)
    Step 43000: Loss = 0.000046 (Best: 0.000028, Patience: 6126/10000)
    Step 44000: Loss = 0.000108 (Best: 0.000028, Patience: 7126/10000)
    Step 45000: Loss = 0.000047 (Best: 0.000028, Patience: 8126/10000)
    Step 46000: Loss = 0.000145 (Best: 0.000028, Patience: 9126/10000)
    
    Early stopping at step 46874
    Best loss: 0.000028, Current loss: 0.000104
    --- starting Experiment: He 4 layer, 32 neurons (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.881602 (Best: 1.881602, Patience: 0/10000)
    Step 1000: Loss = 0.004058 (Best: 0.003161, Patience: 60/10000)
    Step 2000: Loss = 0.001974 (Best: 0.001384, Patience: 11/10000)
    Step 3000: Loss = 0.001041 (Best: 0.000925, Patience: 92/10000)
    Step 4000: Loss = 0.000913 (Best: 0.000706, Patience: 271/10000)
    Step 5000: Loss = 0.000751 (Best: 0.000568, Patience: 28/10000)
    Step 6000: Loss = 0.000555 (Best: 0.000454, Patience: 201/10000)
    Step 7000: Loss = 0.000524 (Best: 0.000347, Patience: 31/10000)
    Step 8000: Loss = 0.000450 (Best: 0.000347, Patience: 1031/10000)
    Step 9000: Loss = 0.000515 (Best: 0.000325, Patience: 11/10000)
    Step 10000: Loss = 0.001128 (Best: 0.000307, Patience: 714/10000)
    Step 11000: Loss = 0.000434 (Best: 0.000269, Patience: 94/10000)
    Step 12000: Loss = 0.000296 (Best: 0.000244, Patience: 265/10000)
    Step 13000: Loss = 0.000292 (Best: 0.000244, Patience: 1265/10000)
    Step 14000: Loss = 0.000340 (Best: 0.000239, Patience: 856/10000)
    Step 15000: Loss = 0.000369 (Best: 0.000214, Patience: 703/10000)
    Step 16000: Loss = 0.000289 (Best: 0.000202, Patience: 251/10000)
    Step 17000: Loss = 0.000256 (Best: 0.000187, Patience: 142/10000)
    Step 18000: Loss = 0.000380 (Best: 0.000166, Patience: 520/10000)
    Step 19000: Loss = 0.000936 (Best: 0.000163, Patience: 660/10000)
    Step 20000: Loss = 0.000277 (Best: 0.000146, Patience: 392/10000)
    Step 21000: Loss = 0.000189 (Best: 0.000138, Patience: 403/10000)
    Step 22000: Loss = 0.000238 (Best: 0.000125, Patience: 12/10000)
    Step 23000: Loss = 0.000233 (Best: 0.000117, Patience: 904/10000)
    Step 24000: Loss = 0.000157 (Best: 0.000117, Patience: 1904/10000)
    Step 25000: Loss = 0.000213 (Best: 0.000114, Patience: 18/10000)
    Step 26000: Loss = 0.000151 (Best: 0.000109, Patience: 552/10000)
    Step 27000: Loss = 0.000154 (Best: 0.000099, Patience: 566/10000)
    Step 28000: Loss = 0.000165 (Best: 0.000099, Patience: 1566/10000)
    Step 29000: Loss = 0.000172 (Best: 0.000097, Patience: 834/10000)
    Step 30000: Loss = 0.000214 (Best: 0.000097, Patience: 1834/10000)
    Step 31000: Loss = 0.000232 (Best: 0.000092, Patience: 359/10000)
    Step 32000: Loss = 0.000452 (Best: 0.000092, Patience: 1359/10000)
    Step 33000: Loss = 0.000165 (Best: 0.000083, Patience: 337/10000)
    Step 34000: Loss = 0.000145 (Best: 0.000083, Patience: 1337/10000)
    Step 35000: Loss = 0.000403 (Best: 0.000083, Patience: 2337/10000)
    Step 36000: Loss = 0.000140 (Best: 0.000083, Patience: 3337/10000)
    Step 37000: Loss = 0.000121 (Best: 0.000077, Patience: 71/10000)
    Step 38000: Loss = 0.000169 (Best: 0.000077, Patience: 1071/10000)
    Step 39000: Loss = 0.000096 (Best: 0.000077, Patience: 2071/10000)
    Step 40000: Loss = 0.000230 (Best: 0.000073, Patience: 379/10000)
    Step 41000: Loss = 0.000126 (Best: 0.000073, Patience: 1379/10000)
    Step 42000: Loss = 0.000313 (Best: 0.000073, Patience: 2379/10000)
    Step 43000: Loss = 0.000148 (Best: 0.000069, Patience: 886/10000)
    Step 44000: Loss = 0.000085 (Best: 0.000069, Patience: 1886/10000)
    Step 45000: Loss = 0.000102 (Best: 0.000069, Patience: 2886/10000)
    Step 46000: Loss = 0.000079 (Best: 0.000069, Patience: 3886/10000)
    Step 47000: Loss = 0.000127 (Best: 0.000064, Patience: 341/10000)
    Step 48000: Loss = 0.000363 (Best: 0.000056, Patience: 241/10000)
    Step 49000: Loss = 0.000254 (Best: 0.000056, Patience: 1241/10000)
    Step 50000: Loss = 0.000082 (Best: 0.000056, Patience: 2241/10000)
    Step 51000: Loss = 0.000146 (Best: 0.000056, Patience: 3241/10000)
    Step 52000: Loss = 0.000183 (Best: 0.000052, Patience: 135/10000)
    Step 53000: Loss = 0.000081 (Best: 0.000050, Patience: 186/10000)
    Step 54000: Loss = 0.001183 (Best: 0.000050, Patience: 1186/10000)
    Step 55000: Loss = 0.000098 (Best: 0.000045, Patience: 693/10000)
    Step 56000: Loss = 0.000061 (Best: 0.000043, Patience: 190/10000)
    Step 57000: Loss = 0.000117 (Best: 0.000042, Patience: 689/10000)
    Step 58000: Loss = 0.000094 (Best: 0.000038, Patience: 95/10000)
    Step 59000: Loss = 0.000148 (Best: 0.000037, Patience: 238/10000)
    Step 60000: Loss = 0.000064 (Best: 0.000033, Patience: 39/10000)
    Step 61000: Loss = 0.000052 (Best: 0.000033, Patience: 1039/10000)
    Step 62000: Loss = 0.000062 (Best: 0.000031, Patience: 212/10000)
    Step 63000: Loss = 0.000041 (Best: 0.000030, Patience: 441/10000)
    Step 64000: Loss = 0.000052 (Best: 0.000027, Patience: 272/10000)
    Step 65000: Loss = 0.000150 (Best: 0.000025, Patience: 531/10000)
    Step 66000: Loss = 0.000050 (Best: 0.000021, Patience: 365/10000)
    Step 67000: Loss = 0.000226 (Best: 0.000021, Patience: 1365/10000)
    Step 68000: Loss = 0.000034 (Best: 0.000021, Patience: 2365/10000)
    Step 69000: Loss = 0.000038 (Best: 0.000018, Patience: 940/10000)
    Step 70000: Loss = 0.000406 (Best: 0.000018, Patience: 1940/10000)
    Step 71000: Loss = 0.000095 (Best: 0.000017, Patience: 57/10000)
    Step 72000: Loss = 0.000122 (Best: 0.000017, Patience: 1057/10000)
    Step 73000: Loss = 0.000608 (Best: 0.000014, Patience: 209/10000)
    Step 74000: Loss = 0.000017 (Best: 0.000014, Patience: 1209/10000)
    Step 75000: Loss = 0.000138 (Best: 0.000014, Patience: 2209/10000)
    Step 76000: Loss = 0.000043 (Best: 0.000011, Patience: 665/10000)
    Step 77000: Loss = 0.000474 (Best: 0.000010, Patience: 59/10000)
    Step 78000: Loss = 0.000013 (Best: 0.000010, Patience: 1059/10000)
    Step 79000: Loss = 0.000056 (Best: 0.000010, Patience: 2059/10000)
    Step 80000: Loss = 0.000014 (Best: 0.000008, Patience: 263/10000)
    Step 81000: Loss = 0.000131 (Best: 0.000008, Patience: 1263/10000)
    Step 82000: Loss = 0.000026 (Best: 0.000008, Patience: 2263/10000)
    Step 83000: Loss = 0.000497 (Best: 0.000008, Patience: 3263/10000)
    Step 84000: Loss = 0.000019 (Best: 0.000007, Patience: 698/10000)
    Step 85000: Loss = 0.000015 (Best: 0.000007, Patience: 1698/10000)
    Step 86000: Loss = 0.000062 (Best: 0.000006, Patience: 764/10000)
    Step 87000: Loss = 0.000044 (Best: 0.000005, Patience: 306/10000)
    Step 88000: Loss = 0.000015 (Best: 0.000005, Patience: 1306/10000)
    Step 89000: Loss = 0.000023 (Best: 0.000005, Patience: 2306/10000)
    Step 90000: Loss = 0.000009 (Best: 0.000004, Patience: 691/10000)
    Step 91000: Loss = 0.000020 (Best: 0.000004, Patience: 1691/10000)
    Step 92000: Loss = 0.000019 (Best: 0.000004, Patience: 2691/10000)
    Step 93000: Loss = 0.000028 (Best: 0.000004, Patience: 3691/10000)
    Step 94000: Loss = 0.000112 (Best: 0.000004, Patience: 4691/10000)
    Step 95000: Loss = 0.000009 (Best: 0.000004, Patience: 5691/10000)
    Step 96000: Loss = 0.000011 (Best: 0.000004, Patience: 6691/10000)
    Step 97000: Loss = 0.000027 (Best: 0.000004, Patience: 7691/10000)
    Step 98000: Loss = 0.000034 (Best: 0.000004, Patience: 8691/10000)
    Step 99000: Loss = 0.000006 (Best: 0.000002, Patience: 581/10000)
    --- starting Experiment: He 5 layer, 32 neurons (loss lambda_ICs=1.0) ---
    Step 0: Loss = 1.901654 (Best: 1.901654, Patience: 0/10000)
    Step 1000: Loss = 0.006270 (Best: 0.005038, Patience: 35/10000)
    Step 2000: Loss = 0.004489 (Best: 0.003363, Patience: 24/10000)
    Step 3000: Loss = 0.002913 (Best: 0.002278, Patience: 156/10000)
    Step 4000: Loss = 0.001708 (Best: 0.001436, Patience: 10/10000)
    Step 5000: Loss = 0.001749 (Best: 0.000992, Patience: 89/10000)
    Step 6000: Loss = 0.000786 (Best: 0.000668, Patience: 30/10000)
    Step 7000: Loss = 0.002163 (Best: 0.000543, Patience: 427/10000)
    Step 8000: Loss = 0.000666 (Best: 0.000500, Patience: 8/10000)
    Step 9000: Loss = 0.000824 (Best: 0.000440, Patience: 161/10000)
    Step 10000: Loss = 0.000705 (Best: 0.000440, Patience: 1161/10000)
    Step 11000: Loss = 0.000528 (Best: 0.000440, Patience: 2161/10000)
    Step 12000: Loss = 0.000571 (Best: 0.000419, Patience: 478/10000)
    Step 13000: Loss = 0.000490 (Best: 0.000399, Patience: 442/10000)
    Step 14000: Loss = 0.000702 (Best: 0.000385, Patience: 346/10000)
    Step 15000: Loss = 0.000820 (Best: 0.000368, Patience: 187/10000)
    Step 16000: Loss = 0.000445 (Best: 0.000368, Patience: 1187/10000)
    Step 17000: Loss = 0.000571 (Best: 0.000312, Patience: 79/10000)
    Step 18000: Loss = 0.000490 (Best: 0.000312, Patience: 1079/10000)
    Step 19000: Loss = 0.000637 (Best: 0.000312, Patience: 2079/10000)
    Step 20000: Loss = 0.000763 (Best: 0.000312, Patience: 3079/10000)
    Step 21000: Loss = 0.000482 (Best: 0.000299, Patience: 14/10000)
    Step 22000: Loss = 0.000510 (Best: 0.000299, Patience: 1014/10000)
    Step 23000: Loss = 0.000461 (Best: 0.000282, Patience: 710/10000)
    Step 24000: Loss = 0.000644 (Best: 0.000277, Patience: 774/10000)
    Step 25000: Loss = 0.000357 (Best: 0.000248, Patience: 118/10000)
    Step 26000: Loss = 0.000318 (Best: 0.000248, Patience: 1118/10000)
    Step 27000: Loss = 0.000461 (Best: 0.000226, Patience: 811/10000)
    Step 28000: Loss = 0.000251 (Best: 0.000226, Patience: 1811/10000)
    Step 29000: Loss = 0.000337 (Best: 0.000218, Patience: 64/10000)
    Step 30000: Loss = 0.000558 (Best: 0.000199, Patience: 448/10000)
    Step 31000: Loss = 0.000391 (Best: 0.000188, Patience: 720/10000)
    Step 32000: Loss = 0.000266 (Best: 0.000181, Patience: 524/10000)
    Step 33000: Loss = 0.000255 (Best: 0.000157, Patience: 497/10000)
    Step 34000: Loss = 0.000200 (Best: 0.000151, Patience: 212/10000)
    Step 35000: Loss = 0.000361 (Best: 0.000148, Patience: 147/10000)
    Step 36000: Loss = 0.000271 (Best: 0.000138, Patience: 552/10000)
    Step 37000: Loss = 0.000152 (Best: 0.000129, Patience: 378/10000)
    Step 38000: Loss = 0.000203 (Best: 0.000129, Patience: 1378/10000)
    Step 39000: Loss = 0.000324 (Best: 0.000111, Patience: 883/10000)
    Step 40000: Loss = 0.000418 (Best: 0.000111, Patience: 1883/10000)
    Step 41000: Loss = 0.000181 (Best: 0.000110, Patience: 705/10000)
    Step 42000: Loss = 0.000125 (Best: 0.000094, Patience: 895/10000)
    Step 43000: Loss = 0.000217 (Best: 0.000094, Patience: 1895/10000)
    Step 44000: Loss = 0.000445 (Best: 0.000088, Patience: 162/10000)
    Step 45000: Loss = 0.000414 (Best: 0.000084, Patience: 95/10000)
    Step 46000: Loss = 0.000134 (Best: 0.000079, Patience: 980/10000)
    Step 47000: Loss = 0.000083 (Best: 0.000070, Patience: 474/10000)
    Step 48000: Loss = 0.000141 (Best: 0.000066, Patience: 47/10000)
    Step 49000: Loss = 0.000308 (Best: 0.000063, Patience: 453/10000)
    Step 50000: Loss = 0.000071 (Best: 0.000058, Patience: 807/10000)
    Step 51000: Loss = 0.000106 (Best: 0.000056, Patience: 659/10000)
    Step 52000: Loss = 0.000091 (Best: 0.000046, Patience: 403/10000)
    Step 53000: Loss = 0.000126 (Best: 0.000046, Patience: 1403/10000)
    Step 54000: Loss = 0.000140 (Best: 0.000043, Patience: 330/10000)
    Step 55000: Loss = 0.000203 (Best: 0.000041, Patience: 180/10000)
    Step 56000: Loss = 0.000038 (Best: 0.000037, Patience: 534/10000)
    Step 57000: Loss = 0.000607 (Best: 0.000035, Patience: 253/10000)
    Step 58000: Loss = 0.000034 (Best: 0.000024, Patience: 140/10000)
    Step 59000: Loss = 0.000060 (Best: 0.000024, Patience: 1140/10000)
    Step 60000: Loss = 0.000141 (Best: 0.000020, Patience: 573/10000)
    Step 61000: Loss = 0.000450 (Best: 0.000018, Patience: 828/10000)
    Step 62000: Loss = 0.000266 (Best: 0.000013, Patience: 301/10000)
    Step 63000: Loss = 0.000073 (Best: 0.000010, Patience: 387/10000)
    Step 64000: Loss = 0.000023 (Best: 0.000010, Patience: 1387/10000)
    Step 65000: Loss = 0.000020 (Best: 0.000009, Patience: 839/10000)
    Step 66000: Loss = 0.000025 (Best: 0.000009, Patience: 1839/10000)
    Step 67000: Loss = 0.000017 (Best: 0.000008, Patience: 72/10000)
    Step 68000: Loss = 0.000037 (Best: 0.000008, Patience: 1072/10000)
    Step 69000: Loss = 0.000045 (Best: 0.000008, Patience: 2072/10000)
    Step 70000: Loss = 0.000009 (Best: 0.000008, Patience: 3072/10000)
    Step 71000: Loss = 0.000024 (Best: 0.000007, Patience: 187/10000)
    Step 72000: Loss = 0.000010 (Best: 0.000007, Patience: 1187/10000)
    Step 73000: Loss = 0.000039 (Best: 0.000007, Patience: 2187/10000)
    Step 74000: Loss = 0.000436 (Best: 0.000007, Patience: 3187/10000)
    Step 75000: Loss = 0.000027 (Best: 0.000006, Patience: 932/10000)
    Step 76000: Loss = 0.000097 (Best: 0.000006, Patience: 1932/10000)
    Step 77000: Loss = 0.000143 (Best: 0.000006, Patience: 2932/10000)
    Step 78000: Loss = 0.000078 (Best: 0.000006, Patience: 3932/10000)
    Step 79000: Loss = 0.000034 (Best: 0.000006, Patience: 4932/10000)
    Step 80000: Loss = 0.000105 (Best: 0.000006, Patience: 5932/10000)
    Step 81000: Loss = 0.000089 (Best: 0.000004, Patience: 354/10000)
    Step 82000: Loss = 0.000023 (Best: 0.000004, Patience: 1354/10000)
    Step 83000: Loss = 0.000008 (Best: 0.000004, Patience: 2354/10000)
    Step 84000: Loss = 0.000385 (Best: 0.000004, Patience: 3354/10000)
    Step 85000: Loss = 0.000037 (Best: 0.000004, Patience: 4354/10000)
    Step 86000: Loss = 0.000541 (Best: 0.000004, Patience: 5354/10000)
    Step 87000: Loss = 0.000025 (Best: 0.000003, Patience: 614/10000)
    Step 88000: Loss = 0.000004 (Best: 0.000003, Patience: 1614/10000)
    Step 89000: Loss = 0.000010 (Best: 0.000003, Patience: 2614/10000)
    Step 90000: Loss = 0.000106 (Best: 0.000003, Patience: 3614/10000)
    Step 91000: Loss = 0.000165 (Best: 0.000003, Patience: 4614/10000)
    Step 92000: Loss = 0.000010 (Best: 0.000003, Patience: 5614/10000)
    Step 93000: Loss = 0.000105 (Best: 0.000003, Patience: 6614/10000)
    Step 94000: Loss = 0.000014 (Best: 0.000003, Patience: 7614/10000)
    Step 95000: Loss = 0.000015 (Best: 0.000003, Patience: 8614/10000)
    Step 96000: Loss = 0.000249 (Best: 0.000002, Patience: 945/10000)
    Step 97000: Loss = 0.000026 (Best: 0.000002, Patience: 1945/10000)
    Step 98000: Loss = 0.000089 (Best: 0.000002, Patience: 2945/10000)
    Step 99000: Loss = 0.000005 (Best: 0.000002, Patience: 3945/10000)
    


    
![png](temp_files/temp_25_1.png)
    



    
![png](temp_files/temp_25_2.png)
    



    
![png](temp_files/temp_25_3.png)
    


    
    ======================================================================
      ξ = 0.1
    ======================================================================
                     Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Kharazmi Deep Network 0.070252   0.176493      0.167557    0.000138 100000
        Zhu Deeper Network 0.015125   0.036268      0.034660    0.000067  42428
    He 3 layer, 32 neurons 0.046279   0.121854      0.111146    0.000104  46875
    He 4 layer, 32 neurons 0.011161   0.023828      0.025334    0.000009 100000
    He 5 layer, 32 neurons 0.009764   0.021599      0.021470    0.000407 100000
    
    ======================================================================
      ξ = 0.2
    ======================================================================
                     Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Kharazmi Deep Network 0.004409   0.021154      0.015987    0.000138 100000
        Zhu Deeper Network 0.002794   0.008220      0.008327    0.000067  42428
    He 3 layer, 32 neurons 0.008316   0.015252      0.022299    0.000104  46875
    He 4 layer, 32 neurons 0.000723   0.002083      0.002096    0.000009 100000
    He 5 layer, 32 neurons 0.004156   0.014364      0.012627    0.000407 100000
    
    ======================================================================
      ξ = 0.3
    ======================================================================
                     Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Kharazmi Deep Network 0.003561   0.010488      0.012564    0.000138 100000
        Zhu Deeper Network 0.002031   0.004012      0.006641    0.000067  42428
    He 3 layer, 32 neurons 0.006890   0.013408      0.021405    0.000104  46875
    He 4 layer, 32 neurons 0.000523   0.001771      0.001885    0.000009 100000
    He 5 layer, 32 neurons 0.003115   0.011370      0.010949    0.000407 100000
    
    ======================================================================
      ξ = 0.4
    ======================================================================
                     Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Kharazmi Deep Network 0.004135   0.012471      0.016364    0.000138 100000
        Zhu Deeper Network 0.002550   0.007550      0.009255    0.000067  42428
    He 3 layer, 32 neurons 0.006716   0.012936      0.020954    0.000104  46875
    He 4 layer, 32 neurons 0.000650   0.001956      0.002474    0.000009 100000
    He 5 layer, 32 neurons 0.003417   0.014179      0.012604    0.000407 100000
    
    


```python
hybrid_experiments = [
    {"name": "Kharazmi Deep Network", "act": nn.Tanh, "lambda": 1.0, "dim": 20, "layers": 4, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    # {"name": "ReLU", "act": nn.ReLU, "lambda": 1.0, "dim": 50, "layers": 3, "number of steps": 2000, "bs":1000},
    {"name": "Zhu Deeper Network", "act": nn.Tanh, "lambda": 1.0, "dim": 250, "layers": 5, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "He 3 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32, "layers": 3, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "He 4 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32, "layers": 4, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000},
    {"name": "He 5 layer, 32 neurons", "act": nn.Tanh, "lambda": 1.0, "dim": 32, "layers": 5, "adam_steps": 5000, "adam_lr": 1e-3, "bs": 1000, "lbfgs_max_iter": 5000, "lbfgs_bs": 2000}
]

hybrid_results = run_hybrid_experiments(hybrid_experiments) # initializzation and training
plot_all_predictions(hybrid_results) # plotting results for specific xi values
plot_all_losses(hybrid_results) # plot loss curve
plot_all_errors(hybrid_results) # plot the error analysis
summary_tables(hybrid_results) # make a summary table where I have some metrics as max, mar, L2 

```

    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: Kharazmi Deep Network
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 2.266744, Avg Loss of last 100 steps = 2.266744 (Best: 2.266744, Patience: 0/2000)
    Step 1000: Loss = 0.005032, Avg Loss of last 100 steps = 0.005862 (Best: 0.005862, Patience: 0/2000)
    Step 2000: Loss = 0.003175, Avg Loss of last 100 steps = 0.003317 (Best: 0.003313, Patience: 2/2000)
    Step 3000: Loss = 0.001593, Avg Loss of last 100 steps = 0.001537 (Best: 0.001537, Patience: 18/2000)
    Step 4000: Loss = 0.000865, Avg Loss of last 100 steps = 0.000963 (Best: 0.000957, Patience: 5/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: Zhu Deeper Network
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 1.893569, Avg Loss of last 100 steps = 1.893569 (Best: 1.893569, Patience: 0/2000)
    Step 1000: Loss = 0.010017, Avg Loss of last 100 steps = 0.005647 (Best: 0.004088, Patience: 168/2000)
    Step 2000: Loss = 0.002204, Avg Loss of last 100 steps = 0.002863 (Best: 0.002491, Patience: 772/2000)
    Step 3000: Loss = 0.001625, Avg Loss of last 100 steps = 0.003117 (Best: 0.002276, Patience: 200/2000)
    Step 4000: Loss = 0.007354, Avg Loss of last 100 steps = 0.002340 (Best: 0.001519, Patience: 591/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: He 3 layer, 32 neurons
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 2.077172, Avg Loss of last 100 steps = 2.077172 (Best: 2.077172, Patience: 0/2000)
    Step 1000: Loss = 0.006511, Avg Loss of last 100 steps = 0.006782 (Best: 0.006782, Patience: 0/2000)
    Step 2000: Loss = 0.002244, Avg Loss of last 100 steps = 0.002117 (Best: 0.002118, Patience: 2/2000)
    Step 3000: Loss = 0.001505, Avg Loss of last 100 steps = 0.001466 (Best: 0.001464, Patience: 10/2000)
    Step 4000: Loss = 0.001425, Avg Loss of last 100 steps = 0.001261 (Best: 0.001250, Patience: 164/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: He 4 layer, 32 neurons
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 1.743219, Avg Loss of last 100 steps = 1.743219 (Best: 1.743219, Patience: 0/2000)
    Step 1000: Loss = 0.003222, Avg Loss of last 100 steps = 0.003683 (Best: 0.003683, Patience: 0/2000)
    Step 2000: Loss = 0.001968, Avg Loss of last 100 steps = 0.002002 (Best: 0.001880, Patience: 73/2000)
    Step 3000: Loss = 0.001079, Avg Loss of last 100 steps = 0.001147 (Best: 0.001130, Patience: 94/2000)
    Step 4000: Loss = 0.000699, Avg Loss of last 100 steps = 0.000722 (Best: 0.000660, Patience: 70/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    
    ======================================================================
      Hybrid Adam + L-BFGS Experiment: He 5 layer, 32 neurons
    ======================================================================
    start training (with Adam)...
    Step 0: Loss = 2.221924, Avg Loss of last 100 steps = 2.221924 (Best: 2.221924, Patience: 0/2000)
    Step 1000: Loss = 0.005125, Avg Loss of last 100 steps = 0.004767 (Best: 0.004767, Patience: 0/2000)
    Step 2000: Loss = 0.002334, Avg Loss of last 100 steps = 0.002429 (Best: 0.002429, Patience: 0/2000)
    Step 3000: Loss = 0.001062, Avg Loss of last 100 steps = 0.001313 (Best: 0.001223, Patience: 37/2000)
    Step 4000: Loss = 0.001032, Avg Loss of last 100 steps = 0.001049 (Best: 0.000995, Patience: 158/2000)
    continue training (with L-BFGS Fine-tuning)...
    training complete.
    


    
![png](temp_files/temp_26_1.png)
    



    
![png](temp_files/temp_26_2.png)
    



    
![png](temp_files/temp_26_3.png)
    


    
    ======================================================================
      ξ = 0.1
    ======================================================================
                     Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Kharazmi Deep Network 0.067317   0.183702      0.161115    0.000094   6920
        Zhu Deeper Network 0.014031   0.040138      0.032102    0.000007   7847
    He 3 layer, 32 neurons 0.019057   0.051869      0.047623    0.000015   8208
    He 4 layer, 32 neurons 0.019325   0.050518      0.044330    0.000010   7565
    He 5 layer, 32 neurons 0.010578   0.034118      0.027177    0.000008   8243
    
    ======================================================================
      ξ = 0.25
    ======================================================================
                     Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Kharazmi Deep Network 0.003081   0.011493      0.011249    0.000094   6920
        Zhu Deeper Network 0.000915   0.002447      0.002875    0.000007   7847
    He 3 layer, 32 neurons 0.001900   0.005729      0.005915    0.000015   8208
    He 4 layer, 32 neurons 0.001183   0.006241      0.004647    0.000010   7565
    He 5 layer, 32 neurons 0.000870   0.003229      0.003051    0.000008   8243
    
    ======================================================================
      ξ = 0.4
    ======================================================================
                     Model      MAE  Max Error  L2 Rel Error  Final Loss  Steps
     Kharazmi Deep Network 0.006109   0.018586      0.024316    0.000094   6920
        Zhu Deeper Network 0.002950   0.006630      0.010580    0.000007   7847
    He 3 layer, 32 neurons 0.003469   0.008426      0.012194    0.000015   8208
    He 4 layer, 32 neurons 0.002562   0.005955      0.009261    0.000010   7565
    He 5 layer, 32 neurons 0.002112   0.004516      0.007479    0.000008   8243
    
    

or we can adopt different models respect to feedforward networks as convolutional neural networks, recurrent neural networks or bayesian neural network, for more informations on the review of this methods:
> — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 2.1.2, 2.1.3 e 2.1.4, from Page 14 to 18*

I hope this project was enough to make me a GSoC 2026 candidate to your lab. I have been studying neural networks for more than one year, and I am diving into the most complex problems and trying to do my best. My degree is basically in Complex Systems, and even though I studied hard in every topic and aced every exam, I think neural networks resonate much more with me compared to, for examples, pandemic models or problems regarding the electrodynamics of continuous media. You can look to my projects for internship and master thesis, or to my certifications from DeepLearning.AI, or to my notes and grades of my masters and I think you will conclude that I have passion for these topics. Also, I am at Naples University at the mathematics department as professor Salvatore Cuomo, we could work together for this project and we could also use his expertise in this research. I tried my best to explain each decision with logic or with the literature review. 

I would like to talk with you more about the project or the papers attached to the website. Even without the GSoC scholarship, I hope we can still work together; until my graduation in the summer, I am basically free. Having a high level scientist at a high level organization as Alabama University, Fermilab or Florida State University as mentor would be an enormous gift that would help me to improve a higher speed. 

More papers working on this PINN problems and ODEs can be see here: 
> — *S. Cuomo et al., [Scientific Machine Learning through Physics-Informed Neural Networks](https://doi.org/10.48550/arXiv.2201.05624), Chapter 3.1, Pages 29 & 30*



```python

```
