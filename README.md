# CellCultureBayesianOptimization
Bayesian Optimization algorithm for Cell Culture Media

5.1.	CodeFiles directory
Step 1:  Define the design factors for the problem, the type of the design factor (continuous, categorical), the number of continuous (Nx) and categorical variables (Nc), the number of categories per categorical variable (C_list/ C), the bounds for the design factors, the type of optimization problem (CoCa vs Co, Unconstrained vs Constrained), Number of initial data points (initN), Number of experiments to be generated in each iteration (batch_size), Measurement noise (Meas_Noise), exploration-exploitation trade-off constant (trade_off)

This information is stored in the dictionary – data_param

Note: Scaling the data to have bounds between [0, 1] is used for numerical stability for the optimizer. The actual upper bound for the design variables is then multiplied -posterior to generate the experimental data. 

Step 2: Generate the initial design. If starting from scratch use design_initial_experiments() function is used to generate initial data. The jupyter notebooks PBMC_TestFile.ipynb and Kphaffi_TestFile.ipynb illustrates the use of this function and the required inputs.

Step 3: After the designed experiments are performed, data is provided back to the algorithm to generate the next set of experiments using design_experiments(). Provided data must be a concatenation of all previous rounds. 

NOTE: In both steps 2 and 3, the background information related to optimization are stored in a .pkl file and preferred filenames are to be provided. This is particularly relevant for categorical continuous optimizations where each iteration's probability distribution and weights are recorded and updated.


5.2.	The Analysis_Kphaffi directory has all the Jupyter Notebook files to recreate the plots and includes the sequential data generated during this work. 

5.3.	The Analysis_PBMC directory has all the Jupyter Notebook and prism files used to analyze and generate the plots and the data generated during the work. 

NOTE: It is to be noted that owing to the stochasticity of the approach, the same experiments will likely not be generated every time the code is executed. We have provided the sequential set of data created during this work. 

