function [diff, G, G_dense, ss] = stationary(x, G, G_dense, param)

%% AGGREGATES
r = x(1); if r > param.rho || r < -0.1, diff = NaN(1); return; end

K = (param.alpha * exp(param.Zmean) / (r + param.delta)) ^ (1/(1-param.alpha)) * param.L;
w = (1-param.alpha) * exp(param.Zmean) * K.^param.alpha .* param.L.^(-param.alpha);
Y = exp(param.Zmean) * K^param.alpha * param.L^(1-param.alpha);


%% VFI
G.income = r * G.a + w .* G.z;

% Boundary conditions:
left_bound  = param.u1(G.income);
right_bound = param.u1(G.income);

BC{1}.left.type = 'VNB'; BC{1}.right.type = 'VNF';
BC{1}.left.f  = @(points) sparse_project(left_bound , points, G);
BC{1}.right.f = @(points) sparse_project(right_bound, points, G);
BC{2}.left.type = '0'; BC{2}.right.type = '0';
G = gen_FD(G, BC);
G_dense = gen_FD(G_dense, BC);

% Initialize guess V0
if ~isfield(G,'V0'), G.V0 = param.u(G.income) / param.rho; end

% Solve VFI
[V, hjb] = VFI_DSS(G, [], param);


%% OUTPUT VF AS NEXT GUESS
G.V0 = V;


%% KOLMOGOROV FORWARD
mu_dense  = G.BH_dense * hjb.s;

g = KF(mu_dense, G_dense, param);


%% MARKET CLEARING
KH = sum(sum( G_dense.a .* g * G_dense.dx));
C  = sum(sum( (G.BH_dense * hjb.c) .* g * G_dense.dx));
S  = sum(sum( (G.BH_dense * hjb.s) .* g * G_dense.dx));

excess_supply  = Y - C - param.delta*KH;
excess_capital = K - KH;

diff = excess_capital;

ss.V = V; ss.g = g; ss.c = hjb.c; ss.s = hjb.s; ss.mass = sum(g * G_dense.dx);
ss.K = K; ss.C = C; ss.S = S; ss.r = r; ss.Y = Y; ss.w = w; ss.A = hjb.A;
ss.excess_supply = excess_supply; ss.excess_capital = excess_capital; 

end

