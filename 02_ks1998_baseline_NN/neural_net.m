function [muK, perf] = neural_net(sim,agg,linear_muK,param)

% training data for the NN
h = diff(sim.K(sim.n_data))./sim.dt;
%X = sim.X(sim.n_data(1:end-1),:);

% rescale inputs, important same scale simulated data and agg.X
%XX = rescale([sim.X(sim.n_data(1:end-1),:);agg.X],-1,1);
XX = normalize([sim.X(sim.n_data(1:end-1),:); agg.X]);

X1 = XX(1:length(h),:);        % data to train
X2 = XX(length(h) + 1: end,:); % data to predict mu_K

rng(1234)

% train the model
model = fitrnet(X1, h, 'LayerSizes', [param.LayerSizes], ...
    'Activations', param.Activations, 'Standardize', false, ...
    'LossTolerance', 1e-10, 'IterationLimit', 10000, ...
    'GradientTolerance', 1e-8, 'StepTolerance', 1e-8);

y = predict(model, X1);
perf = max(abs( log(y) - log(h) ));
%perf = NaN(1); % not used

% predictions of the model of drift
muK = predict(model, X2);


%% Optimze hyperparameters
param.optimize = 0;
if param.optimize
    
gird_LayerSizes{1} = 10:4:40;
grid_Activations = {'relu', 'tanh', 'sigmoid', 'none'};

best.RMSE = perf; disp(perf);
best.layer_1 = param.LayerSizes;
best.activ = param.Activations;

for layer_1 = gird_LayerSizes{1}
    for activ = grid_Activations
        model_ = fitrnet(X1, h, 'LayerSizes', [layer_1], ...
            'Activations', activ, 'Standardize', false, ...
            'LossTolerance', 1e-10, 'IterationLimit', 10000, ...
            'GradientTolerance', 1e-8, 'StepTolerance', 1e-8);

        y = predict(model_, X1);
        perf = max(abs( log(y) - log(h) ));
        if perf < best.RMSE
            best.layer_1 = layer_1;
            best.activ = activ;
            best.RMSE = perf;
        end
    end
end

% train the model
model = fitrnet(X1, h, 'LayerSizes', [best.layer_1], ...
    'Activations', best.activ, 'Standardize', false, ...
    'LossTolerance', 1e-10, 'IterationLimit', 10000, ...
    'GradientTolerance', 1e-8, 'StepTolerance', 1e-8);

y = predict(model, X1);
perf = max(abs( log(y) - log(h) ));
%perf = NaN(1); % not used

% predictions of the model of drift
muK = predict(model, X2);

disp(best);

end

end
