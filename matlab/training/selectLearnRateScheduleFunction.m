function learnRateFunction = selectLearnRateScheduleFunction(learnRateSchedule)
    switch learnRateSchedule
        case "decay"
            learnRateFunction = @decayLearnRateSchedule;
        case "piecewise"
            learnRateFunction = @piecewiseLearnRateSchedule;
        case "constant"
            learnRateFunction = @constantLearnRateSchedule;
    end
end

function learnRate = decayLearnRateSchedule(iteration, ~, options)
    learnRate = options.initialLearnRate / (1 + options.learnRateDecay * iteration);
end

function learnRate = piecewiseLearnRateSchedule(iteration, learnRate, options)
    if mod(iteration, options.learnRateDropPeriod) == 0
        learnRate = learnRate*options.learnRateDropFactor;
    end
end

function learnRate = constantLearnRateSchedule(~, learnRate, ~)

end