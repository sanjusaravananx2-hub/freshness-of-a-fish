function fishFreshnessML()
    try
        fig = figure('Name', 'Fish Freshness Analysis', ...
                    'Position', [300, 300, 500, 400], ...  
                    'MenuBar', 'none', ...
                    'NumberTitle', 'off');
                    
        while true
            options = {'Train Model', 'Predict Single Image', 'Exit'};

            delete(findobj(fig, 'Type', 'uicontrol'));
   
            uicontrol('Style', 'text', ...
                     'String', 'Fish Freshness Analysis', ...
                     'Position', [100, 350, 300, 30], ...
                     'FontSize', 16, ...
                     'FontWeight', 'bold');

            btnHeight = 50; 
            btnWidth = 300;  
            spacing = 20;
            startY = 280;
            
            for i = 1:length(options)
                uicontrol('Style', 'pushbutton', ...
                         'String', options{i}, ...
                         'Position', [100, startY - (i-1)*(btnHeight+spacing), btnWidth, btnHeight], ...
                         'FontSize', 14, ... 
                         'Callback', @(src,event)makeChoice(i));
            end
            
            
            uiwait(fig);
            
           
            if ~ishandle(fig)
                break;
            end
        end
    catch ME
        errordlg(['Error: ' ME.message], 'Error');
    end
    

    function makeChoice(choice)
        switch choice
            case 1
                trainModel();
            case 2
                predictFreshness();
            case 3
                if ishandle(fig)
                    close(fig);
                end
                return;
        end
        uiresume(fig);
    end
end
function trainModel()
    folder = uigetdir('', 'Select folder containing training images');
    if folder == 0
        return;
    end
    
    categories = {'fresh', 'moderate', 'spoiled'};
    allFeatures = [];
    allLabels = [];
    
    h = waitbar(0, 'Processing training images...');
    totalImages = 0;
    
    for i = 1:length(categories)
        imgPath = fullfile(folder, categories{i});
        if ~exist(imgPath, 'dir')
            error('Category folder %s not found. Please create all required folders.', categories{i});
        end
        images = dir(fullfile(imgPath, '*.jpg'));
        totalImages = totalImages + length(images);
    end
    
    processedImages = 0;
    failedImages = 0;
    
    for i = 1:length(categories)
        category = categories{i};
        fprintf('Processing %s images...\n', category);
        
        imgPath = fullfile(folder, category);
        images = dir(fullfile(imgPath, '*.jpg'));
        
        for j = 1:length(images)
            imgFile = fullfile(imgPath, images(j).name);
            try
                features = extractFeatures(imgFile);
                
                if isempty(allFeatures)
                    allFeatures = features;
                    allLabels = i;
                else
                    if length(features) == size(allFeatures, 2)
                        allFeatures = [allFeatures; features];
                        allLabels = [allLabels; i];
                    else
                        warning('Skipping image %s: Inconsistent feature dimensions', images(j).name);
                        failedImages = failedImages + 1;
                        continue;
                    end
                end
                
                processedImages = processedImages + 1;
                waitbar(processedImages/totalImages, h, ...
                    sprintf('Processing images... %d/%d', processedImages, totalImages));
                
            catch ME
                warning('Failed to process image %s: %s', images(j).name, ME.message);
                failedImages = failedImages + 1;
            end
        end
    end
    
    close(h);
    
    if isempty(allFeatures)
        error('No valid images could be processed. Please check your dataset.');
    end
    
    fprintf('\nProcessing completed:\n');
    fprintf('Successfully processed: %d images\n', processedImages);
    fprintf('Failed to process: %d images\n', failedImages);
    
    fprintf('Training SVM model...\n');
    
    [scaledFeatures, mu, sigma] = zscore(allFeatures);
    
  
    rng(1); % For reproducibility
    template = templateSVM('KernelFunction', 'rbf', ...
        'KernelScale', 'auto', ...
        'Standardize', true, ...
        'BoxConstraint', 10);
    
    SVMModel = fitcecoc(scaledFeatures, allLabels, ...
        'Learners', template, ...
        'ClassNames', [1,2,3], ...
        'Coding', 'onevsall');
    
    % Save the trained model and scaling parameters
    save('fishFreshnessModel.mat', 'SVMModel', 'mu', 'sigma');
    fprintf('Model trained and saved successfully!\n');
end

% [Previous functions remain the same until extractFeatures]

function features = extractFeatures(imgFile)
    % Read and preprocess image
    try
        img = imread(imgFile);
        
        if size(img, 3) ~= 3
            error('Image must be RGB');
        end
        
        % Resize image for consistency
        img = imresize(img, [224 224]);
        
        % Convert to double
        img = im2double(img);
        
        % Initialize feature vector
        features = [];
        
        % 1. Color Features
        % RGB features
        meanRGB = squeeze(mean(mean(img, 1), 2))';
        stdRGB = squeeze(std(std(img, 0, 1), 0, 2))';
        
        % HSV features
        hsvImg = rgb2hsv(img);
        meanHSV = squeeze(mean(mean(hsvImg, 1), 2))';
        stdHSV = squeeze(std(std(hsvImg, 0, 1), 0, 2))';
        
        % Lab color space features
        labImg = rgb2lab(img);
        meanLab = squeeze(mean(mean(labImg, 1), 2))';
        stdLab = squeeze(std(std(labImg, 0, 1), 0, 2))';
        
        % 2. Texture Features
        grayImg = rgb2gray(img);
        
        % GLCM features with multiple offsets
        offsets = [0 1; -1 1; -1 0; -1 -1];
        glcmFeatures = zeros(1, 16); % 4 properties × 4 directions
        
        idx = 1;
        for k = 1:size(offsets, 1)
            glcm = graycomatrix(im2uint8(grayImg), 'Offset', offsets(k,:), 'NumLevels', 8);
            stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
            glcmFeatures(idx:idx+3) = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
            idx = idx + 4;
        end
        
        % Custom LBP features
        lbpFeatures = computeLBPFeatures(grayImg);
        
        % 3. Shape Features
        % Edge density
        edges = edge(grayImg, 'Canny');
        edgeDensity = sum(edges(:)) / numel(edges);
        
        % Region properties
        bw = imbinarize(grayImg, 'adaptive', 'Sensitivity', 0.4);
        stats = regionprops(bw, 'Area', 'Perimeter', 'Eccentricity', 'Solidity');
        if ~isempty(stats)
            shapeFeatures = [stats(1).Area/numel(bw), stats(1).Perimeter/numel(bw), ...
                stats(1).Eccentricity, stats(1).Solidity];
        else
            shapeFeatures = zeros(1, 4);
        end
        
        % Combine all features
        features = [
            meanRGB, stdRGB, ...                % RGB features (6)
            meanHSV, stdHSV, ...                % HSV features (6)
            meanLab, stdLab, ...                % Lab features (6)
            glcmFeatures, ...                   % GLCM features (16)
            lbpFeatures, ...                    % LBP features (10)
            edgeDensity, ...                    % Edge density (1)
            shapeFeatures                       % Shape features (4)
        ];
        
        % Ensure features is a row vector
        features = features(:)';
        
        % Verify feature vector length
        expectedLength = 49; % Total number of features
        if length(features) ~= expectedLength
            error('Invalid feature vector length: expected %d, got %d', ...
                expectedLength, length(features));
        end
        
    catch ME
        error('Error processing image %s: %s', imgFile, ME.message);
    end
end

function lbpFeatures = computeLBPFeatures(grayImg)
    % Compute basic LBP features without requiring the Computer Vision Toolbox
    
    % Initialize output
    lbpFeatures = zeros(1, 10);
    
    % Get image dimensions
    [rows, cols] = size(grayImg);
    
    % Initialize LBP image
    lbpImage = zeros(rows-2, cols-2);
    
    % Compute LBP
    for i = 2:rows-1
        for j = 2:cols-1
            % Get 3x3 neighborhood
            neighborhood = grayImg(i-1:i+1, j-1:j+1);
            center = neighborhood(2,2);
            
            % Compare with center and create binary pattern
            pattern = neighborhood(:) >= center;
            pattern(5) = []; % Remove center pixel
            
            % Convert binary pattern to decimal
            lbpValue = sum(pattern .* (2.^(0:7)'));
            
            % Store in LBP image
            lbpImage(i-1,j-1) = lbpValue;
        end
    end
    
    % Compute histogram of LBP values (normalized)
    histBins = 256;
    histogram = histcounts(lbpImage, histBins, 'Normalization', 'probability');
    
    % Extract statistical features from LBP histogram
    lbpFeatures(1) = mean(histogram);
    lbpFeatures(2) = std(histogram);
    lbpFeatures(3) = skewness(histogram);
    lbpFeatures(4) = kurtosis(histogram);
    lbpFeatures(5) = entropy(histogram);
    
    % Additional features from LBP image
    lbpFeatures(6) = mean(lbpImage(:));
    lbpFeatures(7) = std(lbpImage(:));
    lbpFeatures(8) = median(lbpImage(:));
    lbpFeatures(9) = max(lbpImage(:));
    lbpFeatures(10) = min(lbpImage(:));
end

% [Rest of the code remains the same]

function predictFreshness()
    if ~exist('fishFreshnessModel.mat', 'file')
        errordlg('No trained model found. Please train the model first.', 'Error');
        return;
    end
    
    % Load the trained model and scaling parameters
    load('fishFreshnessModel.mat', 'SVMModel', 'mu', 'sigma');
    
    [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png', 'Image files (*.jpg,*.jpeg,*.png)'}, ...
        'Select fish image for analysis');
    if isequal(filename, 0)
        return;
    end
    
    imgFile = fullfile(pathname, filename);
    
    try
        % Extract features
        features = extractFeatures(imgFile);
        
        % MODIFIED FEATURE SCALING CODE GOES HERE
        % Ensure exact match of feature dimensions
        if length(features) ~= length(mu)
            error('Feature vector length mismatch. Expected %d, got %d', length(mu), length(features));
        end
        
        % Safe scaling with error checking
        scaledFeatures = zeros(size(features));
        for i = 1:length(features)
            if sigma(i) ~= 0
                scaledFeatures(i) = (features(i) - mu(i)) / sigma(i);
            else
                scaledFeatures(i) = 0;  % Avoid division by zero
            end
        end
        
        % Debugging: Print scaled features
        disp('Scaled Features:');
        disp(scaledFeatures);
        
        % Predict class
        [predictedClass, scores] = predict(SVMModel, scaledFeatures);
        
        % Calculate confidence
        confidenceScores = abs(scores);
        maxScore = max(confidenceScores);
        confidence = (maxScore / sum(confidenceScores)) * 100;
        
        % Display results
        categories = {'Fresh', 'Moderately Fresh', 'Spoiled'};
        msg = sprintf('Prediction: %s\nConfidence: %.1f%%', ...
            categories{predictedClass}, confidence);
        msgbox(msg, 'Analysis Result');
        
    catch ME
        errordlg(['Error analyzing image: ' ME.message], 'Error');
        disp('Error Report:');
        disp(getReport(ME, 'extended'));
    end
end

function visualizeFeatures()
    % Load previously extracted features or prompt for feature extraction

    % Attempt to load existing feature data
    if exist('fishFeatures.mat', 'file')
        load('fishFeatures.mat', 'allFeatures', 'categories', 'categoryNames');
    else
        % If no saved features, run feature extraction
        choice = questdlg('No saved features found. Would you like to extract features?', ...
            'Feature Extraction', 'Yes', 'No', 'Yes');
        
        if strcmp(choice, 'No')
            return;
        end
        
        % Prompt for folder with images
        folder = uigetdir('', 'Select folder containing training images');
        if folder == 0
            return;
        end
        
        % Initialize categories and feature storage
        categoryNames = {'fresh', 'moderate', 'spoiled'};
        allFeatures = [];
        categories = [];
        
        % Process images and extract features
        for i = 1:length(categoryNames)
            imgPath = fullfile(folder, categoryNames{i});
            images = dir(fullfile(imgPath, '*.jpg'));
            
            for j = 1:length(images)
                try
                    imgFile = fullfile(imgPath, images(j).name);
                    features = extractFeatures(imgFile);
                    
                    % Accumulate features
                    allFeatures = [allFeatures; features];
                    categories = [categories; i];
                catch ME
                    warning('Failed to process image %s: %s', images(j).name, ME.message);
                end
            end
        end
        
        % Save features for future use
        save('fishFeatures.mat', 'allFeatures', 'categories', 'categoryNames');
    end
    
    % Ensure categories is a column vector
    categories = categories(:);
    
    % Feature names (based on extractFeatures function)
    featureNames = {
        'Mean R', 'Mean G', 'Mean B', 'Std R', 'Std G', 'Std B', ...           % RGB features (6)
        'Mean H', 'Mean S', 'Mean V', 'Std H', 'Std S', 'Std V', ...            % HSV features (6)
        'Mean L', 'Mean A', 'Mean B', 'Std L', 'Std A', 'Std B', ...            % Lab features (6)
        'GLCM Contrast 1', 'GLCM Corr 1', 'GLCM Energy 1', 'GLCM Homo 1', ...  % GLCM features (16)
        'GLCM Contrast 2', 'GLCM Corr 2', 'GLCM Energy 2', 'GLCM Homo 2', ...
        'GLCM Contrast 3', 'GLCM Corr 3', 'GLCM Energy 3', 'GLCM Homo 3', ...
        'GLCM Contrast 4', 'GLCM Corr 4', 'GLCM Energy 4', 'GLCM Homo 4', ...
        'LBP Mean', 'LBP Std', 'LBP Skew', 'LBP Kurt', 'LBP Entropy', ...      % LBP features (10)
        'LBP Image Mean', 'LBP Image Std', 'LBP Median', ...
        'LBP Image Max', 'LBP Image Min', ...
        'Edge Density', ...                                                    % Edge density (1)
        'Area Ratio', 'Perimeter Ratio', 'Eccentricity', 'Solidity'             % Shape features (4)
    };
    
    % Create figure with two subplots
    figure('Position', [100, 100, 1500, 800], 'Name', 'Fish Image Feature Analysis');
    
    % 1. Bar Graph of Mean Feature Values by Category
    subplot(1,2,1);
    meanFeatures = zeros(length(categoryNames), size(allFeatures, 2));
    
    for i = 1:length(categoryNames)
        catFeatures = allFeatures(categories == i, :);
        meanFeatures(i, :) = mean(catFeatures, 1);
    end
    
    % Select top 10 most distinct features
    [~, topFeatureIndices] = maxk(std(meanFeatures), 10);
    
    bar(meanFeatures(:, topFeatureIndices));
    title('Top 10 Mean Features by Fish Freshness Category');
    xlabel('Category');
    ylabel('Feature Value');
    legend(categoryNames, 'Location', 'best');
    xticks(1:length(categoryNames));
    xticklabels(categoryNames);
    
    % 2. Boxplot of Top Features
       
    % Ensure categories is a column vector
categories = categories(:);

% Verify dimensions before plotting
if size(allFeatures, 1) ~= length(categories)
    warning('Dimension mismatch: Adjusting data');
    % Truncate or pad to match
    minLength = min(size(allFeatures, 1), length(categories));
    allFeatures = allFeatures(1:minLength, :);
    categories = categories(1:minLength);
end

% Select top 10 most distinct features
[~, topFeatureIndices] = maxk(std(allFeatures), 10);

% Boxplot with additional error checking
try
    boxplot(allFeatures(:, topFeatureIndices), categories, ...
        'Labels', categoryNames, ...
        'Symbol', 'o', 'Whisker', 1.5);
catch ME
    disp('Error in boxplot:');
    disp(getReport(ME));
end

    subplot(1,2,2);
    % Additional error checking
    if size(allFeatures, 1) ~= length(categories)
        error('Number of feature rows (%d) does not match number of category labels (%d)', ...
            size(allFeatures, 1), length(categories));
    end
    
    % Create boxplot with explicit category labels
    boxplot(allFeatures(:, topFeatureIndices), categories, ...
        'Labels', categoryNames, ...
        'Symbol', 'o', 'Whisker', 1.5);
    title('Distribution of Top Features Across Categories');
    xlabel('Category');
    ylabel('Feature Value');
    
    % Adjust layout
    sgtitle('Fish Image Feature Analysis', 'FontSize', 16);
    
    % Optional: Save the figure
    saveas(gcf, 'fish_features_visualization.png');
    
    % Provide statistical summary
    fprintf('Feature Analysis Summary:\n');
    for i = 1:length(topFeatureIndices)
        featIdx = topFeatureIndices(i);
        fprintf('%s:\n', featureNames{featIdx});
        for j = 1:length(categoryNames)
            catFeatures = allFeatures(categories == j, featIdx);
            fprintf('  %s - Mean: %.4f, Std: %.4f\n', ...
                categoryNames{j}, mean(catFeatures), std(catFeatures));
        end
        fprintf('\n');
    end
    
    % Show the saved image
    msgbox('Feature visualization saved as fish_features_visualization.png', 'Visualization Complete');
end
fishFreshnessML()

