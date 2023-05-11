function [muK, perf] = neural_net(sim,agg,linear_muK,param)

% training data for the NN
h = diff(sim.K(sim.n_data))./sim.dt;
%X = sim.X(sim.n_data(1:end-1),:);

% rescale inputs, important same scale simulated data and agg.X
%XX = rescale([sim.X(sim.n_data(1:end-1),:);agg.X],-1,1);
XX = normalize([sim.X(sim.n_data(1:end-1),:);agg.X]);

X1 = XX(1:length(h),:);
X2 = XX(length(h) + 1: end,:);

rng(1234)
% train the model
model = fitrnet(X1,h,'LayerSizes',[param.LayerSizes], ...
    'Activations',param.Activations,'Standardize',false, ...
    'LossTolerance',1e-10,'IterationLimit',10000, ...
    'GradientTolerance',1e-8,'StepTolerance',1e-8);

% y = predict(model,sim.X(sim.n_data(1:end-1),:));
% perf = (max(y-h));
perf = NaN(1);

% predictions of the model of drift
muK = predict(model,X2);

end