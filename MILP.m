%% Example MILP formulation for Drosophila HD network

%load params and set other params

load('/Users/dysprague/fixed-point-finder/data/fly_RNN_params.mat')

k_phi = 1;
k_y = 1;
k_z = 15;
dt = 0.01;

input_HD = 0;

alpha_tilde = 10;
alpha_HD_leak = alpha_tilde + 0.5*(k_y/k_phi)*(1/(k_phi+k_y));
I_ext_coeff = alpha;

w_HD_HD = alpha_tilde + 1/(k_phi+k_y);
w_HD_AVplus = sqrt(2)*k_y/(k_y+k_phi);
w_HD_AVminus = sqrt(2)*k_y/(k_y+k_phi);
w_HD_Del7 = 1/(k_y+k_phi);
w_AVplus_HD = 1;
w_AVminus_HD = 1;
w_Del7_HD = 1;

tau_AVplus = 0.01;
tau_AVmins = 0.01;
tau_Del7 = 0.001;


%%   Ax + B*y = -Iext      and    C*u + D*y = 0,  with u = relu(x)

A = w_HD_HD*W_HD_HD - alpha_HD_leak*eye(size(W_HD_HD));
B = w_HD_Del7*W_HD_Del7;
C = w_Del7_HD*W_Del7_HD;
D = W_Del7_Del7-eye(size(W_Del7_Del7));

%% where y is known to be nonnegative (so relu(y)=y)

% Dimensions (example values, adjust as needed)
n_x = size(A,2);  % Dimension of x (and u)
n_y = size(B,2);  % Dimension of y

% Define decision variables ordering:
%   x:   indices 1:n_x       (pre-activation for x)
%   u:   indices n_x+1:2*n_x   (u = relu(x))
%   y:   indices 2*n_x+1:2*n_x+n_y  (y variables, known to be >=0)
%   z:   indices 2*n_x+n_y+1:end   (binary for each x component)
nvars = 2*n_x + n_y + n_x;  % total variables = 3*n_x + n_y

% Define index sets:
idx_x  = 1:n_x;
idx_u  = n_x+1 : 2*n_x;
idx_y  = 2*n_x+1 : 2*n_x+n_y;
idx_z  = 2*n_x+n_y+1 : nvars;

%% Variable bounds
% Bounds for x (pre-activation); these must be provided.
L_x = -20 * ones(n_x,1);  % example lower bounds (must be <0)
U_x =  40* ones(n_x,1);  % example upper bounds (must be >0)

% For u = relu(x), we know u>=0. We set an upper bound; here we use U_x.
L_u = zeros(n_x,1);
U_u = U_x;

% For y, since y>=0 we can set lower bounds 0.
L_y = 0.2*ones(n_y,1);
%L_y = zeros(n_y,1);
% Set an upper bound for y (if known) or use inf.
U_y = inf(n_y,1);

% For binary variables z: they are 0/1.
L_z = zeros(n_x,1);
U_z = ones(n_x,1);

% Assemble lower and upper bounds for the full decision vector
lb = [L_x; L_u; L_y; L_z];
ub = [U_x; U_u; U_y; U_z];

% Specify that the last n_x variables (the z's) are integer (binary)
intcon = idx_z;

%% Build MILP constraints

% --- Equality constraints ---
% Equation (1): A*x + B*y = -Iext.
% Note: I is the identity matrix of size n_x.
Aeq1 = zeros(n_x, nvars);
Aeq1(:, idx_x) = A;
Aeq1(:, idx_y) = B;
beq1 = -I_ext_coeff*(cos(transpose(phi_0_r_HD) - input_HD));
%beq1 = zeros(n_x,1);

% Equation (2): C*u + D*y = 0.
% Here I_y is the identity matrix of size n_y.
Aeq2 = zeros(n_y, nvars);
Aeq2(:, idx_u) = C;
Aeq2(:, idx_y) = D;
beq2 = zeros(n_y, 1);

% Combine equality constraints:
Aeq = [Aeq1; Aeq2];
beq = [beq1; beq2];

% --- Inequality constraints for the ReLU on x ---
% We enforce u = relu(x) via a big-M formulation.

% Constraint 1: u >= x  <=>  x - u <= 0.
nReLU = n_x;  % one per x
Aineq1 = zeros(nReLU, nvars);
Aineq1(:, idx_x) = eye(nReLU);
Aineq1(:, idx_u) =  -eye(nReLU);
bineq1 = zeros(nReLU, 1);

% Constraint 2: u >= 0 is already ensured by lower bounds on u.

% Constraint 3: u <= x - L_x*(1 - z).
% Rearranged: u - x + L_x*z <= -L_x.
Aineq2 = zeros(nReLU, nvars);
Aineq2(:, idx_u) =  eye(nReLU);
Aineq2(:, idx_x) = -eye(nReLU);
% For the z term, each constraint uses the corresponding L_x:
Aineq2(:, idx_z) = diag(L_x);  % note: L_x is negative so the RHS becomes -L_x
bineq2 = -L_x;

% Constraint 4: u <= U_x*z.
Aineq3 = zeros(nReLU, nvars);
Aineq3(:, idx_u) = eye(nReLU);
Aineq3(:, idx_z) = -diag(U_x);
bineq3 = zeros(nReLU,1);

% Optionally, add tighter constraints on x:
% Constraint 5: x <= U_x*z.
Aineq4 = zeros(nReLU, nvars);
Aineq4(:, idx_x) = eye(nReLU);
Aineq4(:, idx_z) = -diag(U_x);
bineq4 = zeros(nReLU,1);

% Constraint 6: x >= L_x*(1 - z) 
%  can be written as: -x - L_x*z <= -L_x.
Aineq5 = zeros(nReLU, nvars);
Aineq5(:, idx_x) = -eye(nReLU);
Aineq5(:, idx_z) = -diag(L_x);
bineq5 = -L_x;

% Combine all inequality constraints:
Aineq = [Aineq1; Aineq2; Aineq3; Aineq4; Aineq5];
bineq = [bineq1; bineq2; bineq3; bineq4; bineq5];

%% Objective function
% If this is a feasibility problem you can use a zero objective.
f = zeros(nvars, 1);
penalty = 1e-3;
f(idx_u) = penalty;

%% Solve the MILP using intlinprog
options = optimoptions('intlinprog','Display','iter');
[x_sol, fval, exitflag, output] = intlinprog(f, intcon, Aineq, bineq, Aeq, beq, lb, ub, options);

%% Extract solution components
x_pre = x_sol(idx_x);
u_relu = x_sol(idx_u);
y_sol = x_sol(idx_y);
z_bin = x_sol(idx_z);

%% Check results
% You can verify that u_relu is close to max(0, x_pre) and that the equality constraints are satisfied.

drHD_dt = I_ext_coeff*(cos(transpose(phi_0_r_HD) - input_HD)) + A*x_pre + B*y_sol;
%drHD_dt = A*x_pre + B*y_sol;
drDel7_dt = C*max(0,x_pre) + D*y_sol;