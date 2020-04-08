function [output] = ANNmapping_with_testing_to_python(path,layers,epochLimit)
    myVars = {'Time','u1','du1','u2','du2',...
                'x1','dx1','d2x1',...
                'x3','dx3','d2x3',...
                'x5','dx5','d2x5',...
                'fT1','dfT1','d2fT1',...
                'fT2','dfT2','d2fT2'};

    %% Defining Input/Output data for training

    tempData = load(path + "babblingTrial_outputData.mat", myVars{:});

    babblingInputData = struct;

    %{
        babblingInputData
            ..all
            ..bio
            ..kinapprox
            ..allmotor
    %}
    babblingInputData.all = return_all_sensory_states(tempData);
    babblingInputData.bio = return_bio_sensory_states(tempData);
    babblingInputData.kinapprox = return_kinapprox_sensory_states(tempData);
    babblingInputData.allmotor = return_allmotor_sensory_states(tempData);

    babblingOutputData=tempData.x1;

    %% Defining Input/Output data for experiment
    %{
        experimentalInputData
            ..<Group Name>
                ..<Movement Type>
    %}
    experimentalInputData = struct;
    experimentalInputData.all = struct;
    experimentalInputData.bio = struct;
    experimentalInputData.kinapprox = struct;
    experimentalInputData.allmotor = struct;

    experimentalOutputData = struct;

    %% sinusoidal angle/ sinusoidal stiffness
    tempData = load(...
        path(1:end-18) + "angleSin_stiffSin_outputData.mat",...
        myVars{:}...
    );
    experimentalInputData.all.angleSin_stiffSin = ...
        return_all_sensory_states(tempData);
    experimentalInputData.bio.angleSin_stiffSin = ...
        return_bio_sensory_states(tempData);
    experimentalInputData.kinapprox.angleSin_stiffSin = ...
        return_kinapprox_sensory_states(tempData);
    experimentalInputData.allmotor.angleSin_stiffSin = ...
        return_allmotor_sensory_states(tempData);

    experimentalOutputData.angleSin_stiffSin=tempData.x1;

    %% step angle/ sinusoidal stiffness
    tempData = load(...
        path(1:end-18) + "angleStep_stiffSin_outputData.mat",...
        myVars{:}...
    );
    experimentalInputData.all.angleStep_stiffSin = ...
        return_all_sensory_states(tempData);
    experimentalInputData.bio.angleStep_stiffSin = ...
        return_bio_sensory_states(tempData);
    experimentalInputData.kinapprox.angleStep_stiffSin = ...
        return_kinapprox_sensory_states(tempData);
    experimentalInputData.allmotor.angleStep_stiffSin = ...
        return_allmotor_sensory_states(tempData);

    experimentalOutputData.angleStep_stiffSin=tempData.x1;

    %% sinusoidal angle/ step stiffness
    tempData = load(...
        path(1:end-18) + "angleSin_stiffStep_outputData.mat",...
        myVars{:}...
    );
    experimentalInputData.all.angleSin_stiffStep = ...
        return_all_sensory_states(tempData);
    experimentalInputData.bio.angleSin_stiffStep = ...
        return_bio_sensory_states(tempData);
    experimentalInputData.kinapprox.angleSin_stiffStep = ...
        return_kinapprox_sensory_states(tempData);
    experimentalInputData.allmotor.angleSin_stiffStep = ...
        return_allmotor_sensory_states(tempData);

    experimentalOutputData.angleSin_stiffStep=tempData.x1;

    %% step angle/ step stiffness
    tempData = load(...
        path(1:end-18) + "angleStep_stiffStep_outputData.mat",...
        myVars{:}...
    );
    experimentalInputData.all.angleStep_stiffStep = ...
        return_all_sensory_states(tempData);
    experimentalInputData.bio.angleStep_stiffStep = ...
        return_bio_sensory_states(tempData);
    experimentalInputData.kinapprox.angleStep_stiffStep = ...
        return_kinapprox_sensory_states(tempData);
    experimentalInputData.allmotor.angleStep_stiffStep = ...
        return_allmotor_sensory_states(tempData);

    experimentalOutputData.angleStep_stiffStep=tempData.x1;

    %% Neural Network Training/Testing
    babblingDataStructure = struct;
    experimentDataStructure = struct;
    sensoryGroups = fieldnames(babblingInputData);
    movementTypes = fieldnames(experimentalInputData.all);
    for i=1:numel(sensoryGroups)
        % Defining Test/Train data
        tempBabblingInputData=babblingInputData.(sensoryGroups{i});
        tempBabblingOutputData=babblingOutputData;

        tempExperimentalInputData = experimentalInputData.(sensoryGroups{i});
        % 90% train and 10% test split
        [trainingIndex,~,testingIndex] = ...
            dividerand(size(tempBabblingInputData,2),.9,0,.1);

        %% NN - Train
        net=feedforwardnet(double(layers));
        net.trainParam.showWindow = 0;   % hide training window
        net.trainParam.epochs = double(epochLimit);
        [net, tr] = train(...
            net,...
            tempBabblingInputData(:,trainingIndex),...
            tempBabblingOutputData(trainingIndex)...
        );
        babblingDataStructure.(sensoryGroups{i}) = struct;
        babblingDataStructure.(sensoryGroups{i}).tr = tr;
        % view(net)
        close all;

        %% NN - Test
        babblingDataStructure.(sensoryGroups{i}).predictedJointAngle = ...
            net(tempBabblingInputData(:,testingIndex));
        babblingDataStructure.(sensoryGroups{i}).testRMSE = sqrt(mean(...
            (...
                tempBabblingOutputData(testingIndex)...
                - babblingDataStructure.(sensoryGroups{i}).predictedJointAngle...
            ).^2 ...
        ));

        %% NN - Experiment
        for j=1:numel(movementTypes)
            % ALL VALUES IN RADIANS
            tempExperimentOutputData = ...
                experimentalOutputData.(movementTypes{j});
            experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}) = ...
                struct;
            experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).expectedJointAngle = ...
                tempExperimentOutputData;
            experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).predictedJointAngle = ...
                net(tempExperimentalInputData.(movementTypes{j}));
            experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).rawError = ...
                experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).expectedJointAngle ...
                -experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).predictedJointAngle;
            experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).experimentRMSE = ...
                sqrt(mean(...
                    experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).rawError.^2 ...
                ));
            experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).experimentMAE = ...
                mean(abs(...
                    experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).rawError...
                ));
            experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).experimentSTD = ...
                std(...
                    experimentDataStructure.(sensoryGroups{i}).(movementTypes{j}).rawError...
                );
        end
    end
    output = struct;
    output.babbling = babblingDataStructure;
    output.experiment = experimentDataStructure;

    function [groupData] = return_all_sensory_states(dataStructure)
        groupData = [...
            dataStructure.x3; dataStructure.dx3; dataStructure.d2x3;...
            dataStructure.x5; dataStructure.dx5; dataStructure.d2x5;...
            dataStructure.fT1; dataStructure.dfT1; dataStructure.d2fT1;...
            dataStructure.fT2; dataStructure.dfT2; dataStructure.d2fT2...
        ];
    end

    function [groupData] = return_bio_sensory_states(dataStructure)
        groupData = [...
            dataStructure.x3; dataStructure.dx3;...
            dataStructure.x5; dataStructure.dx5;...
            dataStructure.fT1; dataStructure.fT2...
        ];
    end

    function [groupData] = return_kinapprox_sensory_states(dataStructure)
        groupData = [...
            dataStructure.x3; dataStructure.dx3;...
            dataStructure.x5; dataStructure.dx5...
        ];
    end

    function [groupData] = return_allmotor_sensory_states(dataStructure)
        groupData = [...
            dataStructure.x3; dataStructure.dx3; dataStructure.d2x3;...
            dataStructure.x5; dataStructure.dx5; dataStructure.d2x5...
        ];
    end
end
