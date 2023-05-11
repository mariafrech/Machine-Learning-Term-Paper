% Simulation:
lStyle  = {'-', '-.'};
figure('visible', 'on'); hold on;
l1 = plot(param.t, sim.K,     'LineStyle', lStyle{1}, 'Color', colorPalette(1, :), 'LineWidth', 2.5);
l2 = plot(param.t, sim.K_lom, 'LineStyle', lStyle{2}, 'Color', colorPalette(2, :), 'LineWidth', 2);
hold off; xlabel('Simulation horizon (years)'); ylabel('Capital');
set(gcf, 'units', 'inches', 'position', [5.5, 4.5, 10, 5.5])
legend([l1, l2], {'Capital: simulation', 'Capital: forecast'}, ...
    'Location', 'SouthOutside', 'NumColumns', 2, 'FontSize', 12, 'Interpreter', 'Latex', 'box', 'off');
set(gcf, 'renderer', 'Painter');
exportgraphics(gcf, './output/simulation_dh.eps');

close all;

%% Compute 1-quarter forecast error
load data_OLS
forecast_error_OLS = sim_OLS.K(param.n_data) - sim_OLS.K_lom(param.n_data);
forecast_error_NN  = sim.K(param.n_data)     - sim.K_lom(param.n_data);
hold on;
l1 = histogram(forecast_error_OLS, 'EdgeAlpha',0.5); % 'LineWidth', 2, 'DisplayStyle','stairs');
l2 = histogram(forecast_error_NN,  'EdgeAlpha',0.5); % 'LineWidth', 2, 'DisplayStyle','stairs');
hold off;
xlim([-1e-2, 1e-2])
xlabel('Forecasting error','FontSize', 12, 'Interpreter', 'Latex');
ylabel('Density','FontSize', 12, 'Interpreter', 'Latex');
legend([l1, l2], {'OLS', 'NN'}, 'Interpreter', 'Latex', 'box', 'off', 'Location', 'SouthEast');
%title('Krusell-Smith with 2-state process','Interpreter', 'Latex', 'FontSize', 15);
set(gcf, 'renderer', 'Painter');
exportgraphics(gcf, './output/hist_error_2state.eps');

%% PLM

close all
scatter3(agg.Z, agg.K, agg.muK)
xlabel('Technological shock $Z_t$','FontSize', 12, 'Interpreter', 'Latex');
ylabel('Capital K_t','FontSize', 12, 'Interpreter', 'Latex');
title('PLM approximated with a NN','Interpreter', 'Latex', 'FontSize', 15);
set(gcf, 'renderer', 'Painter');
exportgraphics(gcf, './output/PLM.eps');

