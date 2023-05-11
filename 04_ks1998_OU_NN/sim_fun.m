function sim = sim_fun(G, G_dense, agg, V, ss, param)

% Assumption: you initialize *everything* at the ss allocation

%% SIMULATE OU-TFP PROCESS
rng(12345);
Z_shock = randn(param.N,1);

Z_t = param.Zmean*ones(param.N, 1);
for n = 1:param.N-1
    Z_t(n+1) = Z_t(n) - param.thetaZ*Z_t(n) * param.dt + param.sigmaZ * sqrt(param.dt) * Z_shock(n);
end

GZ_t = (Z_t - param.min(3)) / (param.max(3) - param.min(3));


%% LAW OF MOTION (LOM) SIMULATION
K_lom     = zeros(param.N,1);
K_lom(1)  = param.K0;
GK_lom    = zeros(param.N,1);
GK_lom(1) = (K_lom(1) - param.min(4)) ./ (param.max(4) - param.min(4));

muK_H = agg.H_comp * agg.muK;
muK_lom = zeros(param.N,1);
muK_lom(1) = agg.muK(ismember(agg.grid, 0.5*ones(1,agg.d), 'rows'), :);

for n = 1:param.N-1
    K_lom(n+1)  = K_lom(n) + muK_lom(n)*param.dt;
    K_lom(n+1)  = max(K_lom(n+1), param.min(4));
    K_lom(n+1)  = min(K_lom(n+1), param.max(4));
    GK_lom(n+1) = (K_lom(n+1) - param.min(4)) ./ (param.max(4) - param.min(4));
    
    BH = H_basis_small([GZ_t(n+1), GK_lom(n+1)], agg.grid, agg.lvl, agg.h);
    muK_lom(n+1) = BH * muK_H;
end

if any(any([sum(GK_lom == 1); sum(GK_lom == 0)])) > 0
    fprintf('K-region not large enough. \n');
    disp([sum(GK_lom == 1); sum(GK_lom == 0)]);
end


%% KF SIMULATION 
Az_dense = FD_operator(G_dense, param.theta_z * (param.zmean - G_dense.z), param.sig_z*ones(G_dense.J, 1), 2);

V_t = cell(param.N,1);
g_t = cell(param.N,1);
g_t{1} = ss.g;

X_t  = zeros(param.N,agg.d);
K_t  = zeros(param.N,1); 
GK_t = zeros(param.N,1);
GX_t = zeros(param.N,agg.d);

K_t(1)    = param.K0;
X_t(1,:)  = [Z_t(1), K_t(1)];
GK_t(1)   = (K_t(1) - param.min(4)) ./ (param.max(4) - param.min(4));
GX_t(1,:) = [GZ_t(1), GK_t(1)];

r_t = zeros(param.N, 1);
w_t = zeros(param.N, 1);
Y_t = zeros(param.N, 1);

BH_dense_agg = H_basis(G_dense.grid, G_dense.lvl, G.grid(:, 1:param.d_idio), G.lvl(:, 1:param.d_idio));
VH = G.H_comp * V;

for n = 1:param.N-1
    
    r_t(n) = param.alpha     .* exp(Z_t(n)) .* K_t(n) .^(param.alpha-1) .* param.L.^(1-param.alpha) - param.delta;
    w_t(n) = (1-param.alpha) .* exp(Z_t(n)) .* K_t(n) .^(param.alpha)   .* param.L.^( -param.alpha);
    Y_t(n) = exp(Z_t(n)) .* K_t(n).^param.alpha .* param.L.^(1-param.alpha);
    
    BH_agg = H_basis_small(GX_t(n, :), G.grid(:, param.d_idio+1:G.d), G.lvl(:, param.d_idio+1:G.d), G.h(:, param.d_idio+1:G.d));
    V_t{n} = BH_dense_agg * (BH_agg' .* VH);
    
    G_dense.income = r_t(n) * G_dense.a + w_t(n) * G_dense.z;
    
%     clear BC;
%     left_bound  = param.u1(G_dense.income);
%     right_bound = param.u1(G_dense.income);
%     for j = 1:param.discrete_types
%         BC{1}.left.type = 'VNB'; BC{1}.right.type = 'VNF';
%         BC{1}.left.f  = @(points) sparse_project(left_bound(:, j),  points, GDense);
%         BC{1}.right.f = @(points) sparse_project(right_bound(:, j), points, GDense);
%         G_dense = gen_FD(G_dense, BC, num2str(j));
%     end

    hjb = HJB(V_t{n}, G_dense, param);
        
    Aa_dense = FD_operator(G_dense, hjb.s, zeros(G_dense.J, 1), 1);
    
    A = Aa_dense + Az_dense;
    AT = A';
    
    g_t{n+1} = (speye(G_dense.J) - AT*param.dt)\g_t{n}; 
    %g_t{n+1} = g_t{n} + param.dt * AT * g_t{n};
    
    mass = sum(sum(g_t{n+1} * G_dense.dx));
    if max(abs(mass - 1)) > 1e-5, fprintf('KF simulation not mass-preserving.'); end
    
    K_t(n+1) = g_t{n+1}' * G_dense.a * G_dense.dx;

    % Impose reflecting boundary (KF should do this automatically)
    K_t(n+1) = max(K_t(n+1, :), param.min(4));
    K_t(n+1) = min(K_t(n+1, :), param.max(4));
        
    X_t(n+1, :) = [Z_t(n+1), K_t(n+1)];
    
    GK_t(n+1) = (K_t(n+1) - param.min(4)) ./ (param.max(4) - param.min(4));
    GX_t(n+1, :) = [GZ_t(n+1), GK_t(n+1)];

end


%% OUTPUT
sim.N = param.N; 
sim.t = param.t;
sim.dt = param.dt;
sim.n_data = param.n_data;
sim.X = X_t;
sim.K = K_t;
sim.K_lom = K_lom;
sim.GK = GK_t; 
sim.GK_lom = GK_lom;


end

