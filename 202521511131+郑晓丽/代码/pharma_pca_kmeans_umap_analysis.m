%% PCA + K-Means + UMAP analysis for pharma/biotech data
% Run this script from the folder that contains the five CSV files.
% Output tables and figures will be written to analysis_outputs/.

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

rngSeed = 42;
rng(rngSeed, 'twister');

kClusters = 3;
kMeansReplicates = 50;
pcaVarianceTargetPct = 85;
useIndustryBackground = true;  % funding and company financials are year-level features.

%% 1. Read source data
trials = readtable(fullfile(rootDir, 'clinical_trials.csv'), 'TextType', 'string');
approvals = readtable(fullfile(rootDir, 'drug_approvals.csv'), 'TextType', 'string');
burden = readtable(fullfile(rootDir, 'disease_burden.csv'), 'TextType', 'string');
funding = readtable(fullfile(rootDir, 'biotech_funding.csv'), 'TextType', 'string');
financials = readtable(fullfile(rootDir, 'pharma_companies_financials.csv'), 'TextType', 'string');

%% 2. Build year x therapy area feature table
trialAgg = aggregateTrials(trials);
approvalAgg = aggregateApprovals(approvals);
burdenAgg = aggregateDiseaseBurden(burden);
fundingAgg = aggregateFunding(funding);
financialAgg = aggregateFinancials(financials);

sampleKeys = makeSampleKeys(trials, approvals, burdenAgg);

featureTbl = sampleKeys;
featureTbl = leftJoin(featureTbl, trialAgg, {'year', 'therapy_area'});
featureTbl = leftJoin(featureTbl, approvalAgg, {'year', 'therapy_area'});
featureTbl = leftJoin(featureTbl, burdenAgg, {'year', 'therapy_area'});
featureTbl = leftJoin(featureTbl, fundingAgg, {'year'});
featureTbl = leftJoin(featureTbl, financialAgg, {'year'});
featureTbl = sortrows(featureTbl, {'year', 'therapy_area'});

baseFeatureVars = [
    "trial_count"
    "phase3_trial_count"
    "success_rate"
    "failure_rate"
    "avg_enrollment_n"
    "avg_duration_months"
    "avg_stock_impact_pct"
    "approval_count"
    "blockbuster_count"
    "mega_blockbuster_count"
    "avg_peak_sales_usd_bn_est"
    "total_peak_sales_usd_bn_est"
    "biologic_approval_count"
    "small_molecule_approval_count"
    "disease_burden_global_millions"
    "disease_burden_regional_sum_millions"
    "mapped_disease_count"
];

industryFeatureVars = [
    "funding_deal_count"
    "funding_megadeal_count"
    "funding_total_usd_bn"
    "funding_avg_value_usd_bn"
    "pharma_company_count"
    "pharma_revenue_total_usd_bn"
    "pharma_operating_margin_avg_pct"
    "pharma_rd_spend_total_usd_bn"
    "pharma_pipeline_total_est"
];

if useIndustryBackground
    featureVars = [baseFeatureVars; industryFeatureVars];
else
    featureVars = baseFeatureVars;
end

zeroFillVars = [
    "trial_count"
    "phase3_trial_count"
    "approval_count"
    "blockbuster_count"
    "mega_blockbuster_count"
    "total_peak_sales_usd_bn_est"
    "biologic_approval_count"
    "small_molecule_approval_count"
    "disease_burden_global_millions"
    "disease_burden_regional_sum_millions"
    "mapped_disease_count"
    "funding_deal_count"
    "funding_megadeal_count"
    "funding_total_usd_bn"
    "pharma_revenue_total_usd_bn"
    "pharma_rd_spend_total_usd_bn"
    "pharma_pipeline_total_est"
];

[featureTbl, X] = tableToNumericMatrix(featureTbl, featureVars, zeroFillVars);
[Xz, featureMean, featureStd] = zscoreMatrix(X);

featureStats = table(featureVars, featureMean(:), featureStd(:), ...
    'VariableNames', {'feature', 'mean_before_standardization', 'std_before_standardization'});

%% 3. PCA
[coeff, score, latent, ~, explained] = pca(Xz);
cumExplained = cumsum(explained);

numPcForCluster = find(cumExplained >= pcaVarianceTargetPct, 1, 'first');
if isempty(numPcForCluster)
    numPcForCluster = min(3, size(score, 2));
end
numPcForCluster = min(max(numPcForCluster, 2), min(5, size(score, 2)));

clusterInput = score(:, 1:numPcForCluster);

%% 4. K-Means clustering on PCA scores
rng(rngSeed, 'twister');
clusterId = kmeans(clusterInput, kClusters, ...
    'Replicates', kMeansReplicates, ...
    'MaxIter', 1000, ...
    'Display', 'off');

%% 5. UMAP visualization
% This script will use run_umap if it is installed. If your MATLAB version
% has a built-in umap function, it will try that next. If neither exists,
% it creates a PCA fallback plot and prints a warning.
[embedding2D, embeddingMethod] = computeEmbedding2D(Xz, rngSeed);

%% 6. Save result tables
analysisTbl = featureTbl;

nPcToSave = min(5, size(score, 2));
for j = 1:nPcToSave
    analysisTbl.(sprintf('PC%d', j)) = score(:, j);
end

analysisTbl.cluster = clusterId;
analysisTbl.embedding_1 = embedding2D(:, 1);
analysisTbl.embedding_2 = embedding2D(:, 2);
analysisTbl.embedding_method = repmat(embeddingMethod, height(analysisTbl), 1);

explainedTbl = table((1:numel(explained))', latent, explained, cumExplained, ...
    'VariableNames', {'PC', 'eigenvalue', 'explained_pct', 'cumulative_explained_pct'});

loadingTbl = table(featureVars, 'VariableNames', {'feature'});
for j = 1:nPcToSave
    loadingTbl.(sprintf('PC%d_loading', j)) = coeff(:, j);
end

clusterProfileTbl = makeClusterProfile(analysisTbl, featureVars);

writetable(featureTbl, fullfile(outDir, 'feature_table_year_therapy_area.csv'));
writetable(featureStats, fullfile(outDir, 'feature_standardization_stats.csv'));
writetable(explainedTbl, fullfile(outDir, 'pca_explained_variance.csv'));
writetable(loadingTbl, fullfile(outDir, 'pca_loadings.csv'));
writetable(analysisTbl, fullfile(outDir, 'clustered_results.csv'));
writetable(clusterProfileTbl, fullfile(outDir, 'cluster_feature_means.csv'));

%% 7. Save figures
plotPcaScree(explained, outDir);
plotPcaScatter(score, explained, clusterId, outDir);
plotEmbeddingScatter(embedding2D, embeddingMethod, clusterId, outDir);
plotClusterHeatmap(clusterProfileTbl, featureVars, outDir);

fprintf('\nAnalysis complete.\n');
fprintf('Output folder: %s\n', outDir);
fprintf('Samples: %d year-therapy_area rows\n', height(analysisTbl));
fprintf('Features used: %d\n', numel(featureVars));
fprintf('PCA components used for K-Means: %d\n', numPcForCluster);
fprintf('Embedding method: %s\n\n', char(embeddingMethod));

%% Local functions
function trialAgg = aggregateTrials(trials)
    trials.year = toDouble(trials.year);
    trials.therapy_area = normalizeKey(trials.therapy_area);
    trials.phase = string(trials.phase);
    trials.is_success = toDouble(trials.is_success);
    trials.is_failure = toDouble(trials.is_failure);
    trials.enrollment_n = toDouble(trials.enrollment_n);
    trials.duration_months = toDouble(trials.duration_months);
    trials.estimated_stock_impact_pct = toDouble(trials.estimated_stock_impact_pct);

    [G, year, therapy_area] = findgroups(trials.year, trials.therapy_area);
    trial_count = splitapply(@numel, string(trials.trial_id), G);
    phase3_trial_count = splitapply(@sumNoNan, double(contains(lower(trials.phase), "phase 3")), G);
    success_rate = splitapply(@meanNoNan, trials.is_success, G);
    failure_rate = splitapply(@meanNoNan, trials.is_failure, G);
    avg_enrollment_n = splitapply(@meanNoNan, trials.enrollment_n, G);
    avg_duration_months = splitapply(@meanNoNan, trials.duration_months, G);
    avg_stock_impact_pct = splitapply(@meanNoNan, trials.estimated_stock_impact_pct, G);

    trialAgg = table(year, therapy_area, trial_count, phase3_trial_count, ...
        success_rate, failure_rate, avg_enrollment_n, avg_duration_months, ...
        avg_stock_impact_pct);
end

function approvalAgg = aggregateApprovals(approvals)
    approvals.year = toDouble(approvals.year);
    approvals.therapy_area = normalizeKey(approvals.therapy_area);
    approvals.drug_type = normalizeKey(approvals.drug_type);
    approvals.peak_sales_usd_bn_est = toDouble(approvals.peak_sales_usd_bn_est);
    approvals.is_blockbuster = toDouble(approvals.is_blockbuster);
    approvals.is_mega_blockbuster = toDouble(approvals.is_mega_blockbuster);

    [G, year, therapy_area] = findgroups(approvals.year, approvals.therapy_area);
    approval_count = splitapply(@numel, string(approvals.approval_id), G);
    blockbuster_count = splitapply(@sumNoNan, approvals.is_blockbuster, G);
    mega_blockbuster_count = splitapply(@sumNoNan, approvals.is_mega_blockbuster, G);
    avg_peak_sales_usd_bn_est = splitapply(@meanNoNan, approvals.peak_sales_usd_bn_est, G);
    total_peak_sales_usd_bn_est = splitapply(@sumNoNan, approvals.peak_sales_usd_bn_est, G);
    biologic_approval_count = splitapply(@sumNoNan, double(approvals.drug_type == "biologic"), G);
    small_molecule_approval_count = splitapply(@sumNoNan, double(approvals.drug_type == "small_mol"), G);

    approvalAgg = table(year, therapy_area, approval_count, blockbuster_count, ...
        mega_blockbuster_count, avg_peak_sales_usd_bn_est, ...
        total_peak_sales_usd_bn_est, biologic_approval_count, ...
        small_molecule_approval_count);
end

function burdenAgg = aggregateDiseaseBurden(burden)
    burden.year = toDouble(burden.year);
    burden.disease = string(burden.disease);
    burden.therapy_area = mapDiseaseToTherapyArea(burden.disease);
    burden.global_dalys_millions = toDouble(burden.global_dalys_millions);
    burden.dalys_millions = toDouble(burden.dalys_millions);

    burden = burden(burden.therapy_area ~= "", :);

    [G1, diseaseYear, diseaseName, diseaseArea] = findgroups( ...
        burden.year, burden.disease, burden.therapy_area);

    disease_global_millions = splitapply(@meanNoNan, burden.global_dalys_millions, G1);
    disease_regional_sum_millions = splitapply(@sumNoNan, burden.dalys_millions, G1);

    burdenByDisease = table(diseaseYear, diseaseName, diseaseArea, ...
        disease_global_millions, disease_regional_sum_millions, ...
        'VariableNames', {'year', 'disease', 'therapy_area', ...
        'disease_global_millions', 'disease_regional_sum_millions'});

    [G2, year, therapy_area] = findgroups(burdenByDisease.year, burdenByDisease.therapy_area);
    disease_burden_global_millions = splitapply(@sumNoNan, burdenByDisease.disease_global_millions, G2);
    disease_burden_regional_sum_millions = splitapply(@sumNoNan, burdenByDisease.disease_regional_sum_millions, G2);
    mapped_disease_count = splitapply(@numel, burdenByDisease.disease, G2);

    burdenAgg = table(year, therapy_area, disease_burden_global_millions, ...
        disease_burden_regional_sum_millions, mapped_disease_count);
end

function fundingAgg = aggregateFunding(funding)
    funding.year = toDouble(funding.year);
    funding.value_usd_bn = toDouble(funding.value_usd_bn);
    funding.is_megadeal = toDouble(funding.is_megadeal);

    [G, year] = findgroups(funding.year);
    funding_deal_count = splitapply(@numel, string(funding.deal_id), G);
    funding_megadeal_count = splitapply(@sumNoNan, funding.is_megadeal, G);
    funding_total_usd_bn = splitapply(@sumNoNan, funding.value_usd_bn, G);
    funding_avg_value_usd_bn = splitapply(@meanNoNan, funding.value_usd_bn, G);

    fundingAgg = table(year, funding_deal_count, funding_megadeal_count, ...
        funding_total_usd_bn, funding_avg_value_usd_bn);
end

function financialAgg = aggregateFinancials(financials)
    financials.year = toDouble(financials.year);
    financials.revenue_usd_bn = toDouble(financials.revenue_usd_bn);
    financials.operating_margin_pct = toDouble(financials.operating_margin_pct);
    financials.rd_spend_usd_bn = toDouble(financials.rd_spend_usd_bn);
    financials.pipeline_size_est = toDouble(financials.pipeline_size_est);

    [G, year] = findgroups(financials.year);
    pharma_company_count = splitapply(@numel, string(financials.company_name), G);
    pharma_revenue_total_usd_bn = splitapply(@sumNoNan, financials.revenue_usd_bn, G);
    pharma_operating_margin_avg_pct = splitapply(@meanNoNan, financials.operating_margin_pct, G);
    pharma_rd_spend_total_usd_bn = splitapply(@sumNoNan, financials.rd_spend_usd_bn, G);
    pharma_pipeline_total_est = splitapply(@sumNoNan, financials.pipeline_size_est, G);

    financialAgg = table(year, pharma_company_count, pharma_revenue_total_usd_bn, ...
        pharma_operating_margin_avg_pct, pharma_rd_spend_total_usd_bn, ...
        pharma_pipeline_total_est);
end

function sampleKeys = makeSampleKeys(trials, approvals, burdenAgg)
    trialYears = toDouble(trials.year);
    approvalYears = toDouble(approvals.year);
    burdenYears = burdenAgg.year;
    years = unique([trialYears; approvalYears; burdenYears]);
    years = years(isfinite(years));
    years = (min(years):max(years))';

    areas = unique([ ...
        normalizeKey(trials.therapy_area); ...
        normalizeKey(approvals.therapy_area); ...
        burdenAgg.therapy_area], 'stable');
    areas = areas(areas ~= "");

    [yearGrid, areaGrid] = ndgrid(years, 1:numel(areas));
    year = yearGrid(:);
    therapy_area = areas(areaGrid(:));

    sampleKeys = table(year, therapy_area);
end

function out = leftJoin(leftTbl, rightTbl, keys)
    if height(rightTbl) == 0
        out = leftTbl;
        return;
    end
    out = outerjoin(leftTbl, rightTbl, 'Keys', keys, 'MergeKeys', true, 'Type', 'left');
end

function [tbl, X] = tableToNumericMatrix(tbl, featureVars, zeroFillVars)
    X = zeros(height(tbl), numel(featureVars));

    for j = 1:numel(featureVars)
        varName = char(featureVars(j));
        data = toDouble(tbl.(varName));
        data(~isfinite(data)) = NaN;

        missing = isnan(data);
        if any(missing)
            if any(featureVars(j) == zeroFillVars)
                data(missing) = 0;
            else
                replacement = median(data(~missing));
                if isempty(replacement) || isnan(replacement)
                    replacement = 0;
                end
                data(missing) = replacement;
            end
        end

        tbl.(varName) = data;
        X(:, j) = data;
    end
end

function [Xz, mu, sigma] = zscoreMatrix(X)
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma == 0 | ~isfinite(sigma)) = 1;
    Xz = (X - mu) ./ sigma;
    Xz(~isfinite(Xz)) = 0;
end

function profileTbl = makeClusterProfile(tbl, featureVars)
    [G, cluster] = findgroups(tbl.cluster);
    sample_count = splitapply(@numel, tbl.cluster, G);
    profileTbl = table(cluster, sample_count);

    for j = 1:numel(featureVars)
        sourceName = char(featureVars(j));
        outName = ['mean_' sourceName];
        profileTbl.(outName) = splitapply(@meanNoNan, tbl.(sourceName), G);
    end
end

function [embedding, method] = computeEmbedding2D(Xz, rngSeed)
    embedding = [];
    method = "";

    if exist('run_umap', 'file') == 2
        try
            rng(rngSeed, 'twister');
            embedding = run_umap(Xz, ...
                'n_components', 2, ...
                'n_neighbors', 15, ...
                'min_dist', 0.1, ...
                'metric', 'euclidean', ...
                'verbose', 'none');
            embedding = cleanEmbedding(embedding);
            method = "run_umap";
        catch ME
            warning('run_umap was found but failed: %s', ME.message);
            embedding = [];
        end
    end

    if isempty(embedding) && exist('umap', 'file') == 2
        try
            rng(rngSeed, 'twister');
            embedding = umap(Xz, ...
                'NumComponents', 2, ...
                'NumNeighbors', 15, ...
                'MinDistance', 0.1);
            embedding = cleanEmbedding(embedding);
            method = "umap";
        catch ME1
            try
                embedding = umap(Xz);
                embedding = cleanEmbedding(embedding);
                method = "umap";
            catch ME2
                warning('umap was found but failed: %s | %s', ME1.message, ME2.message);
                embedding = [];
            end
        end
    end

    if isempty(embedding)
        try
            embedding = pythonUmapEmbedding(Xz, rngSeed);
            method = "Python umap-learn";
        catch ME
            warning('Python umap-learn was not available: %s', ME.message);
            embedding = [];
        end
    end

    if isempty(embedding)
        [~, fallbackScore] = pca(Xz);
        embedding = fallbackScore(:, 1:min(2, size(fallbackScore, 2)));
        method = "PCA fallback; install run_umap or umap-learn for true UMAP";
        warning('No working UMAP function was found. Install run_umap, MATLAB umap, or Python umap-learn for true UMAP.');
    end

    if size(embedding, 2) < 2
        embedding(:, 2) = 0;
    elseif size(embedding, 2) > 2
        embedding = embedding(:, 1:2);
    end
end

function embedding = pythonUmapEmbedding(Xz, rngSeed)
    umapModule = py.importlib.import_module('umap');
    np = py.importlib.import_module('numpy');

    reducer = umapModule.UMAP(pyargs( ...
        'n_components', int32(2), ...
        'n_neighbors', int32(15), ...
        'min_dist', 0.1, ...
        'metric', 'euclidean', ...
        'random_state', int32(rngSeed)));

    Xpy = np.array(Xz);
    Ypy = reducer.fit_transform(Xpy);
    embedding = pyListToMatrix(Ypy.tolist());
end

function M = pyListToMatrix(pyList)
    rows = cell(pyList);
    nRows = numel(rows);
    firstRow = cell(rows{1});
    nCols = numel(firstRow);
    M = zeros(nRows, nCols);

    for r = 1:nRows
        row = cell(rows{r});
        for c = 1:nCols
            M(r, c) = double(row{c});
        end
    end
end

function embedding = cleanEmbedding(embedding)
    if iscell(embedding)
        embedding = embedding{1};
    end
    if istable(embedding)
        embedding = table2array(embedding);
    end
    embedding = double(embedding);
end

function plotPcaScree(explained, outDir)
    n = min(10, numel(explained));
    fig = figure('Color', 'w', 'Position', [100 100 900 520]);
    bar(1:n, explained(1:n), 'FaceColor', [0.22 0.45 0.70]);
    hold on;
    plot(1:n, cumsum(explained(1:n)), '-o', ...
        'Color', [0.85 0.33 0.10], 'LineWidth', 1.8, 'MarkerFaceColor', [0.85 0.33 0.10]);
    grid on;
    xlabel('Principal component');
    ylabel('Explained variance (%)');
    title('PCA explained variance');
    legend({'Single PC', 'Cumulative'}, 'Location', 'best');
    saveFigure(fig, fullfile(outDir, 'pca_scree.png'));
    close(fig);
end

function plotPcaScatter(score, explained, clusterId, outDir)
    if size(score, 2) < 2
        score(:, 2) = 0;
    end

    fig = figure('Color', 'w', 'Position', [100 100 900 620]);
    scatter(score(:, 1), score(:, 2), 46, clusterId, 'filled');
    grid on; box on;
    colormap(lines(max(clusterId)));
    cb = colorbar;
    cb.Label.String = 'K-Means cluster';
    xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
    ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
    title('PCA scores colored by K-Means cluster');
    saveFigure(fig, fullfile(outDir, 'pca_scatter_clusters.png'));
    close(fig);
end

function plotEmbeddingScatter(embedding, method, clusterId, outDir)
    fig = figure('Color', 'w', 'Position', [100 100 900 620]);
    scatter(embedding(:, 1), embedding(:, 2), 46, clusterId, 'filled');
    grid on; box on;
    colormap(lines(max(clusterId)));
    cb = colorbar;
    cb.Label.String = 'K-Means cluster';
    xlabel('Embedding 1');
    ylabel('Embedding 2');
    title(sprintf('2D embedding colored by K-Means cluster (%s)', char(method)));
    saveFigure(fig, fullfile(outDir, 'umap_scatter_clusters.png'));
    close(fig);
end

function plotClusterHeatmap(clusterProfileTbl, featureVars, outDir)
    meanVars = "mean_" + featureVars;
    M = zeros(height(clusterProfileTbl), numel(meanVars));
    for j = 1:numel(meanVars)
        M(:, j) = clusterProfileTbl.(char(meanVars(j)));
    end

    [Mz, ~, ~] = zscoreMatrix(M);

    fig = figure('Color', 'w', 'Position', [100 100 1200 520]);
    imagesc(Mz);
    colormap(parula);
    colorbar;
    xlabel('Feature');
    ylabel('Cluster');
    yticks(1:height(clusterProfileTbl));
    yticklabels(string(clusterProfileTbl.cluster));
    xticks(1:numel(featureVars));
    xticklabels(strrep(featureVars, '_', '\_'));
    xtickangle(45);
    title('Cluster feature profile (standardized means)');
    saveFigure(fig, fullfile(outDir, 'cluster_feature_heatmap.png'));
    close(fig);
end

function saveFigure(fig, filePath)
    try
        exportgraphics(fig, filePath, 'Resolution', 180);
    catch
        saveas(fig, filePath);
    end
end

function area = mapDiseaseToTherapyArea(disease)
    d = lower(strtrim(string(disease)));
    area = strings(size(d));

    area(d == "cardiovascular disease" | d == "stroke") = "cardiovascular";
    area(d == "cancer") = "oncology";
    area(d == "diabetes" | d == "kidney disease") = "metabolic";
    area(d == "chronic respiratory") = "respiratory";
    area(d == "neurological disorders" | d == "alzheimer's & dementias") = "neurology";
    area(d == "mental disorders" | d == "self-harm") = "psychiatry";
    area(d == "hiv/aids" | d == "tuberculosis" | d == "malaria" | ...
        d == "covid-19" | d == "lower respiratory infections" | ...
        d == "diarrheal diseases") = "infectious";
    area(d == "liver disease") = "gastrointestinal";
end

function s = normalizeKey(v)
    s = lower(strtrim(string(v)));
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
