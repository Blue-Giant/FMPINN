clc;
clear all
close all
M=65;
% M=51;
% opt = 'rand_disorder';
opt = 'my_disorder';

if strcmp('my_disorder', opt)
    original_index = int32(linspace(0,M*M-1,M*M));
    disorder_index = reorder_index(original_index, M);

    order_index = recover_index(disorder_index, M);
else
    original_index = int32(linspace(0,M*M*M-1,M*M*M));
    index = randperm(length(original_index));
    disorder_index = original_index(index);
    recover(index) = disorder_index; 
end

if M==51
    save('disorder_index51.mat','disorder_index');
elseif M==65
    save('disorder_index65.mat','disorder_index');
elseif M==26
    save('disorder_index26.mat','disorder_index');
end