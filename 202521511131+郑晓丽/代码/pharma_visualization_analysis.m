%% Exploratory visualization for the pharma/biotech feature table
% This script creates descriptive figures used in course_paper.tex.

clear; clc; close all;

scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    rootDir = pwd;
else
    rootDir = fileparts(scriptPath);
end

outDir = fullfile(rootDir, 'analysis_outputs');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

featureFile = fullfile(outDir, 'feature_table_year_therapy_area.csv');
clusterFile = fullfile(outDir, 'clustered_results.csv');

T = readtable(featureFile, 'TextType', 'string');
T.year = toDouble(T.year);
T.therapy_area = string(T.therapy_area);

numericVars = ["trial_count", "phase3_trial_count", "success_rate", ...
    "approval_count", "blockbuster_count", "total_peak_sales_usd_bn_est", ...
    "disease_burden_global_millions", "funding_total_usd_bn", ...
    "pharma_rd_spend_total_usd_bn"];

for j = 1:numel(numericVars)
    T.(numericVars(j)) = toDouble(T.(numericVars(j)));
end

makeYearlyTrendFigure(T, outDir);
makeTherapyActivityFigure(T, outDir);
makeBurdenCommercialMismatchFigure(T, outDir);

if exist(clusterFile, 'file') == 2
    C = readtable(clusterFile, 'TextType', 'string');
    C.year = toDouble(C.year);
    C.cluster = toDouble(C.cluster);
    C.therapy_area = string(C.therapy_area);
    makeClusterCompositionFigure(C, outDir);
end

fprintf('Exploratory visualization figures saved to: %s\n', outDir);

function makeYearlyTrendFigure(T, outDir)
    [G, year] = findgroups(T.year);
    trialSum = splitapply(@sumNoNan, T.trial_count, G);
    approvalSum = splitapply(@sumNoNan, T.approval_count, G);
    salesSum = splitapply(@sumNoNan, T.total_peak_sales_usd_bn_est, G);

    fig = figure('Color', 'w', 'Position', [100 100 980 560]);
    yyaxis left;
    plot(year, trialSum, '-o', 'LineWidth', 2.0, 'MarkerSize', 5);
    hold on;
    plot(year, approvalSum, '-s', 'LineWidth', 2.0, 'MarkerSize', 5);
    ylabel('Count');
    ylim([0, max([trialSum; approvalSum]) * 1.18]);

    yyaxis right;
    plot(year, salesSum, '-^', 'LineWidth', 2.0, 'MarkerSize', 5);
    ylabel('Total peak sales estimate (USD bn)');
    ylim([0, max(salesSum) * 1.18]);

    grid on; box on;
    xlabel('Year');
    title('Yearly trends of clinical activity, approvals, and commercial potential');
    legend({'Clinical trials', 'Drug approvals', 'Peak sales estimate'}, ...
        'Location', 'northwest');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
    exportFigure(fig, fullfile(outDir, 'viz_yearly_trends.png'));
    close(fig);
end

function makeTherapyActivityFigure(T, outDir)
    [G, area] = findgroups(T.therapy_area);
    meanTrials = splitapply(@meanNoNan, T.trial_count, G);
    meanApprovals = splitapply(@meanNoNan, T.approval_count, G);
    meanPhase3 = splitapply(@meanNoNan, T.phase3_trial_count, G);

    activityScore = meanTrials + meanApprovals;
    [~, idx] = sort(activityScore, 'descend');
    area = area(idx);
    values = [meanTrials(idx), meanPhase3(idx), meanApprovals(idx)];

    fig = figure('Color', 'w', 'Position', [100 100 1120 580]);
    bar(values, 'grouped');
    grid on; box on;
    xticks(1:numel(area));
    xticklabels(strrep(area, '_', '\_'));
    xtickangle(35);
    ylabel('Average count per year');
    title('Average R&D and approval activity by therapy area');
    legend({'Trials', 'Phase 3 trials', 'Approvals'}, 'Location', 'northwest');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 10.5);
    exportFigure(fig, fullfile(outDir, 'viz_therapy_area_activity.png'));
    close(fig);
end

function makeBurdenCommercialMismatchFigure(T, outDir)
    [G, area] = findgroups(T.therapy_area);
    burden = splitapply(@meanNoNan, T.disease_burden_global_millions, G);
    approvals = splitapply(@meanNoNan, T.approval_count, G);
    trials = splitapply(@meanNoNan, T.trial_count, G);
    sales = splitapply(@meanNoNan, T.total_peak_sales_usd_bn_est, G);

    bubbleSize = 55 + normalizeSize(sales) * 520;

    fig = figure('Color', 'w', 'Position', [100 100 920 650]);
    scatter(burden, approvals, bubbleSize, trials, 'filled', ...
        'MarkerFaceAlpha', 0.72, 'MarkerEdgeColor', [0.2 0.2 0.2]);
    grid on; box on;
    colormap(turbo);
    cb = colorbar;
    cb.Label.String = 'Average trial count';
    xlabel('Average global disease burden (DALYs, millions)');
    ylabel('Average approval count');
    title('Disease burden versus approval activity by therapy area');

    for i = 1:numel(area)
        text(burden(i) + 4, approvals(i), strrep(area(i), "_", "\_"), ...
            'FontName', 'Times New Roman', 'FontSize', 9);
    end

    set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);
    exportFigure(fig, fullfile(outDir, 'viz_burden_commercial_mismatch.png'));
    close(fig);
end

function makeClusterCompositionFigure(C, outDir)
    [clusterList, ~, cIdx] = unique(C.cluster);
    [areaList, ~, aIdx] = unique(C.therapy_area);
    counts = accumarray([cIdx, aIdx], 1, [numel(clusterList), numel(areaList)], @sum, 0);

    [~, order] = sort(sum(counts, 1), 'descend');
    areaList = areaList(order);
    counts = counts(:, order);

    fig = figure('Color', 'w', 'Position', [100 100 1020 560]);
    bar(clusterList, counts, 'stacked');
    grid on; box on;
    xlabel('Cluster');
    ylabel('Sample count');
    title('Therapy area composition within K-Means clusters');
    legend(strrep(areaList, "_", "\_"), 'Location', 'eastoutside');
    set(gca, 'FontName', 'Times New Roman', 'FontSize', 10.5);
    exportFigure(fig, fullfile(outDir, 'viz_cluster_composition.png'));
    close(fig);
end

function y = normalizeSize(x)
    x = double(x);
    finite = isfinite(x);
    y = zeros(size(x));
    if any(finite)
        xmin = min(x(finite));
        xmax = max(x(finite));
        if xmax > xmin
            y(finite) = (x(finite) - xmin) ./ (xmax - xmin);
        end
    end
end

function x = toDouble(v)
    if isnumeric(v)
        x = double(v);
    elseif islogical(v)
        x = double(v);
    else
        x = str2double(string(v));
    end
    x = x(:);
end

function y = meanNoNan(x)
    x = double(x);
    x = x(isfinite(x));
    if isempty(x)
        y = NaN;
    else
        y = mean(x);
    end
end

function y = sumNoNan(x)
    x = double(x);
    x = x(isfinite(x));
    if isempty(x)
        y = 0;
    else
        y = sum(x);
    end
end

function exportFigure(fig, filePath)
    try
        exportgraphics(fig, filePath, 'Resolution', 200);
    catch
        saveas(fig, filePath);
    end
end
