function [V, hjb] = VFI(G, agg, param)

V = G.V0;

% EXOGENOUS OPERATORS
Az = FD_operator(G, param.theta_z * (param.zmean - G.z), param.sig_z*ones(G.J,1), 2);
AZ = FD_operator(G, agg.muZ(agg.agg2full, :), agg.sigZ(agg.agg2full, :), 3);
AK = FD_operator(G, agg.muK(agg.agg2full, :), agg.sigK(agg.agg2full, :), 4);

for iter = 1:param.maxit

% COMPUTE POLICY FUNCTIONS
hjb = HJB(V, G, param);
if any(any(isnan(hjb.c))), V = NaN(1); return; end

% ASSEMBLE FD OPERATOR MATRIX
Aa = FD_operator(G, hjb.s, zeros(G.J, 1), 1);

A = Aa + Az + AZ + AK;

B = (1/param.Delta + param.rho)*speye(G.J) - A;
b = hjb.u + V / param.Delta;

% SOLVE LINEAR SYSTEM
% V_new = B\b;
[V_new, flag] = gmres(B, b, [], param.crit/10, 2000, [], [], V);

% UPDATE
V_change = V_new - V;
V = V_new;

dist = max(max(abs(V_change)));
if dist < param.crit, break; end

% if mod(iter,1) == 0, fprintf('VFI %.3i,  Distance: %.6d,  Hash: %.7f \n', iter, dist, mean(mean(V))); end
if ~isreal(V), fprintf('Complex values in VFI: terminating process.'); V = NaN(1); return; end

end

if iter == param.maxit, fprintf('VFI did not converge. Remaining Gap: %.2d\n', iter, dist); V = NaN(1); return; end

end